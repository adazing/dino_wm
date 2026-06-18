"""
Generate WM-planned trajectories for BC training.

Plans to goal images with the world model (PlanWorkspace + a planner), executes
the plans in the env, and saves the resulting observation/action/state
trajectories as a PlannedTrajDataset file. Uses the same config as plan.py.

Usage:
    python generate_planned_trajectories.py model_name=<run> ckpt_base_path=<abs> \\
        planner=mpc_cem goal_source=dset goal_H=5 n_evals=10 \\
        n_planning_rounds=20 planned_out_path=./planned_pusht.pth

The saved file is read by datasets.planned_dset.PlannedTrajDataset and fed to
train_policy.py (data_source=planned or both).
"""
import os
import gym
import torch
import hydra
import numpy as np
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
from utils import cfg_to_dict, seed as set_seed
from plan import load_model, PlanWorkspace


def _to_uint8_visual(arr):
    """(T, H, W, C) in [0, 255] (np or tensor) -> uint8 tensor."""
    t = torch.as_tensor(np.asarray(arr))
    if t.dtype != torch.uint8:
        t = t.round().clamp(0, 255).to(torch.uint8)
    return t


def _pad_actions_to(act, length):
    """Pad (T, A) to (length, A) by repeating the last action."""
    if act.shape[0] >= length:
        return act[:length]
    pad = act[-1:].repeat(length - act.shape[0], 1)
    return torch.cat([act, pad], dim=0)


def generate(cfg_dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt_base_path = cfg_dict["ckpt_base_path"]
    model_path = f"{ckpt_base_path}/outputs/{cfg_dict['model_name']}/"
    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)

    set_seed(cfg_dict["seed"])
    _, dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    # generation harvests TRAINING data for the BC policy -> sample goals from the
    # train split (plan.py uses "valid" because it's *evaluating* the WM, not us).
    dset = dset["train"]

    frameskip = model_cfg.frameskip
    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    model = load_model(model_ckpt, model_cfg, num_action_repeat, device=device)

    def make_env():
        if model_cfg.env.name in ("wall", "deformable_env"):
            from env.serial_vector_env import SerialVectorEnv
            return SerialVectorEnv(
                [gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)
                 for _ in range(cfg_dict["n_evals"])]
            )
        return SubprocVectorEnv(
            [lambda: gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)
             for _ in range(cfg_dict["n_evals"])]
        )

    env = make_env()

    n_rounds = int(cfg_dict.get("n_planning_rounds", 1))
    only_successes = bool(cfg_dict.get("planned_only_successes", False))

    visual, actions_out, states_out, proprios_out, seq_lengths = [], [], [], [], []
    base_seed = cfg_dict["seed"]

    for r in range(n_rounds):
        round_cfg = dict(cfg_dict)
        round_cfg["seed"] = base_seed + r * 1000  # vary sampled targets per round

        workspace = PlanWorkspace(
            cfg_dict=round_cfg,
            wm=model,
            dset=dset,
            env=env,
            env_name=model_cfg.env.name,
            frameskip=frameskip,
            wandb_run=None,
        )
        plan_actions, action_len = workspace.planner.plan(
            obs_0=workspace.obs_0, obs_g=workspace.obs_g, actions=None
        )
        logs, successes, e_obses, e_states = workspace.evaluator.eval_actions(
            plan_actions.detach(),
            action_len,
            save_video=(r == 0),
            filename=f"planned_round{r}",
        )
        print(f"[round {r}] success_rate={logs.get('success_rate')}")

        # planner actions: (b, H, f*d) normalized -> per-step (b, H*f, d)
        per_step = rearrange(plan_actions.detach().cpu(), "b t (f d) -> b (t f) d", f=frameskip)

        b = per_step.shape[0]
        for i in range(b):
            if only_successes and not bool(np.asarray(successes)[i]):
                continue
            vis_i = _to_uint8_visual(e_obses["visual"][i])  # (T, H, W, C)
            T = vis_i.shape[0]
            act_i = _pad_actions_to(per_step[i].float(), T)
            prop = e_obses.get("proprio")  # not all envs expose proprio; policy is image-only
            prop_i = (torch.as_tensor(np.asarray(prop[i])).float()
                      if prop is not None else torch.zeros(T, 1))
            state_i = torch.as_tensor(np.asarray(e_states[i])).float()
            visual.append(vis_i)
            actions_out.append(act_i)
            states_out.append(state_i)
            proprios_out.append(prop_i)
            seq_lengths.append(int(T))

    env.close()

    out_path = os.path.abspath(cfg_dict.get("planned_out_path", "./planned_trajectories.pth"))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(
        {
            "visual": visual,
            "actions": actions_out,
            "states": states_out,
            "proprios": proprios_out,
            "seq_lengths": seq_lengths,
            "env_name": model_cfg.env.name,
            "frameskip": frameskip,
        },
        out_path,
    )
    print(f"Saved {len(seq_lengths)} planned trajectories to {out_path}")
    return out_path


@hydra.main(config_path="conf", config_name="plan")
def main(cfg: OmegaConf):
    from hydra.utils import get_original_cwd
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict["wandb_logging"] = False
    # resolve a relative planned_out_path against the dir the user ran from
    # (Hydra changes cwd to the run dir), so train_policy.py can find it.
    out = cfg_dict.get("planned_out_path") or "./planned_trajectories.pth"
    if not os.path.isabs(out):
        out = os.path.join(get_original_cwd(), out)
    cfg_dict["planned_out_path"] = out
    generate(cfg_dict)


if __name__ == "__main__":
    main()
