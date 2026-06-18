"""
Train a behavior-cloning policy (VQ-BeT or Diffusion) on a frozen patch encoder,
over one of three data sources:

    data_source=original  -> original demonstrations only
    data_source=planned   -> WM-planned trajectories only (generate_planned_trajectories.py)
    data_source=both      -> original + planned

Examples:
    accelerate launch train_policy.py policy=vqbet     data_source=original
    accelerate launch train_policy.py policy=diffusion data_source=both \\
        planned_data_path=./planned_pusht.pth
"""
import os
import gym
import hydra
import torch
import numpy as np
from pathlib import Path
from datetime import timedelta
from collections import deque
from einops import rearrange
from omegaconf import OmegaConf
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs

from utils import seed as set_seed
from env.venv import SubprocVectorEnv

from datasets.policy_dataset import (
    make_raw_policy_traj,
    split_traj_datasets,
    TrajectoryEmbeddingDataset,
    VqbetTrajectorySlicerDataset,
)

# allow ${eval:'...'} in configs (used by the diffusion policy's pred_horizon)
OmegaConf.register_new_resolver("eval", eval, replace=True)

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False


# parameter-count helpers
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_params(n):
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{n}"
        n /= 1000.0
    return f"{n:.1f}T"


# build raw trajectory datasets
def build_raw_dino_traj_datasets(dataset_cfg, num_hist, num_pred, frameskip):
    """Instantiate an env's trajectory datasets and return the train split only.

    The valid split is left out so env-eval (which samples its goals/inits from
    "valid") stays held out from BC training. default_transform outputs [0,1],
    which is what the encoders expect, so no transform override is needed."""
    _, traj_dset = hydra.utils.call(
        dataset_cfg, num_hist=num_hist, num_pred=num_pred, frameskip=frameskip
    )
    key = "train" if traj_dset.get("train") is not None else next(iter(traj_dset))
    return [traj_dset[key]]


def main(cfg):
    # accelerate / logging
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
    dist_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with="wandb" if _HAS_WANDB else None,
        kwargs_handlers=[process_group_kwargs, dist_kwargs],
    )
    device = accelerator.device
    set_seed(cfg.seed)

    use_diffusion = "diffusion" in cfg.policy["_target_"]
    gpu_batch_size = max(1, cfg.batch_size // accelerator.num_processes)

    # goal-conditioned BC: goal = a goal-image embedding (dim = encoder.output_dim),
    # stacked with obs inside the policy. Set goal_dim so cfg.policy resolves it
    # (done before wandb.init so the logged config is accurate).
    goal_conditional = bool(cfg.get("goal_conditional", False))
    cfg.env.goal_dim = int(cfg.encoder.output_dim) if goal_conditional else 0

    do_wandb = accelerator.is_main_process and _HAS_WANDB and cfg.get("wandb_logging", True)
    if do_wandb:
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                   name=str(cfg.experiment),
                   config=OmegaConf.to_container(cfg, resolve=True))

    # frozen patch encoder
    encoder = hydra.utils.instantiate(cfg.encoder)
    encoder = accelerator.prepare(encoder)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # raw dataset for the chosen data source
    original = (build_raw_dino_traj_datasets(cfg.env.dataset, cfg.num_hist, cfg.num_pred, cfg.frameskip)
                if cfg.data_source in ("original", "both") else [])
    planned = []
    if cfg.data_source in ("planned", "both"):
        assert cfg.planned_data_path is not None, "planned_data_path required for planned/both"
        from datasets.planned_dset import PlannedTrajDataset
        from datasets.img_transforms import default_transform
        # default_transform resizes to img_size and keeps [0,1]
        planned = [PlannedTrajDataset(cfg.planned_data_path, transform=default_transform(cfg.img_size))]
    # fail fast if the configured action dim doesn't match the actual data
    ref = (original or planned)[0]
    ref_base = getattr(ref, "dataset", ref)
    data_act_dim = getattr(ref_base, "action_dim", None)
    if data_act_dim is not None and int(cfg.env.act_dim) != int(data_act_dim):
        raise ValueError(
            f"cfg.env.act_dim={cfg.env.act_dim} but the dataset's action_dim={data_act_dim}. "
            f"Set act_dim correctly in conf/env/{cfg.env.name}.yaml."
        )

    dataset = make_raw_policy_traj(original, planned, cfg.data_source)

    # policy + optimizer
    cbet_model = hydra.utils.instantiate(cfg.policy).to(device)
    optimizer = cbet_model.configure_optimizers(
        weight_decay=cfg.optim.weight_decay,
        learning_rate=cfg.optim.lr,
        betas=tuple(cfg.optim.betas),
    )
    if use_diffusion:
        from normalizer import LinearNormalizer
        action_normalizer = LinearNormalizer()
        action_normalizer.fit(dataset.get_all_actions())
        cbet_model.set_normalizer(action_normalizer)

    # parameter counts
    if accelerator.is_main_process:
        enc_total, enc_train = count_parameters(encoder)
        mdl_total, mdl_train = count_parameters(cbet_model)
        print("\n" + "=" * 60 + "\nMODEL PARAMETER COUNTS\n" + "=" * 60)
        print(f"\nEncoder:\n  Total parameters:     {format_params(enc_total):>10} ({enc_total:,})")
        print(f"  Trainable parameters: {format_params(enc_train):>10} ({enc_train:,})")
        print(f"\nPolicy Model (cbet_model):\n  Total parameters:     {format_params(mdl_total):>10} ({mdl_total:,})")
        print(f"  Trainable parameters: {format_params(mdl_train):>10} ({mdl_train:,})")
        print(f"\nCombined (Encoder + Policy):\n  Total parameters:     {format_params(enc_total + mdl_total):>10} ({enc_total + mdl_total:,})")
        print(f"  Trainable parameters: {format_params(enc_train + mdl_train):>10} ({enc_train + mdl_train:,})")

    # precompute embeddings, slice, build loaders
    train_data, test_data = split_traj_datasets(
        dataset, train_fraction=cfg.train_fraction, random_seed=cfg.seed
    )
    train_data = TrajectoryEmbeddingDataset(encoder, train_data, device=cfg.embed_device)
    test_data = TrajectoryEmbeddingDataset(encoder, test_data, device=cfg.embed_device)
    slicer_kwargs = dict(window=cfg.window_size, action_window=cfg.action_window_size,
                         vqbet_get_future_action_chunk=False, goal_conditional=goal_conditional)
    train_data = VqbetTrajectorySlicerDataset(train_data, **slicer_kwargs)
    test_data = VqbetTrajectorySlicerDataset(test_data, **slicer_kwargs)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=gpu_batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=gpu_batch_size, shuffle=False, num_workers=cfg.num_workers)

    cbet_model, optimizer, train_loader, test_loader = accelerator.prepare(
        cbet_model, optimizer, train_loader, test_loader)

    goal_dim = int(cfg.env.goal_dim)

    def run_model(batch):
        obs, act = batch[0].to(device), batch[1].to(device)
        obs = rearrange(obs, "N T V P E -> N T (V P) E")
        goal = None if goal_dim == 0 else rearrange(batch[2].to(device), "N T V P E -> N T (V P) E")
        return cbet_model(obs, goal, act)

    out_dir = Path(os.getcwd())
    for epoch in range(cfg.epochs):
        # closed-loop env eval (main process only)
        if cfg.eval_on_env and (epoch + 1) % cfg.eval_on_env_freq == 0 and accelerator.is_main_process:
            try:
                metrics = eval_on_env(cfg, accelerator.unwrap_model(encoder),
                                      accelerator.unwrap_model(cbet_model), device, use_diffusion)
                print(f"eval_on_env: {metrics}")
                if do_wandb:
                    wandb.log({**{f"env/{k}": v for k, v in metrics.items()}, "epoch": epoch})
            except Exception as e:
                print(f"eval_on_env skipped: {e}")

        # validation loss + action_diff metrics
        if epoch % cfg.eval_freq == 0:
            cbet_model.eval()
            total_loss, n = 0.0, 0
            ad = {k: 0.0 for k in ("action_diff", "action_diff_tot",
                                   "action_diff_mean_res1", "action_diff_mean_res2", "action_diff_max")}
            with torch.no_grad():
                for batch in test_loader:
                    _, loss, loss_dict = run_model(batch)
                    if loss is None:
                        continue
                    total_loss += loss.item(); n += 1
                    if do_wandb:
                        wandb.log({**{f"eval/{x}": y for x, y in loss_dict.items()}, "epoch": epoch})
                    if not use_diffusion:
                        for k in ad:
                            ad[k] += loss_dict.get(k, 0.0)
            if accelerator.is_main_process:
                print(f"Test loss: {total_loss / max(1, n)}")
                if do_wandb and not use_diffusion:
                    wandb.log({**{f"eval/epoch_wise_{k}": v for k, v in ad.items()}, "epoch": epoch})

        # train
        cbet_model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            _, loss, loss_dict = run_model(batch)
            train_loss += loss.item()
            accelerator.backward(loss)
            optimizer.step()
            if use_diffusion:
                if hasattr(cbet_model, "module"):
                    accelerator.unwrap_model(cbet_model).ema_step()
                else:
                    cbet_model.ema_step()
            if do_wandb:
                wandb.log({**{f"train/{x}": y for x, y in loss_dict.items()}, "epoch": epoch})

        # VQ-BeT fits its codebook at epoch boundary
        raw_model = accelerator.unwrap_model(cbet_model)
        if hasattr(raw_model, "finish_epoch"):
            raw_model.finish_epoch()
        if accelerator.is_main_process:
            print(f"Train loss: {train_loss / max(1, len(train_loader))}")

        # save (main process)
        if epoch % cfg.save_every == 0 and accelerator.is_main_process:
            torch.save(accelerator.unwrap_model(cbet_model), out_dir / f"model_{epoch}.pt")

    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(cbet_model), out_dir / "model_final.pt")


# Closed-loop env eval: roll the policy out per step, report eval_state success
# plus coverage for envs that expose it (pusht). The sim state comes from
# info["state"]. Action ensembling follows the standard receding-horizon scheme.
@torch.no_grad()
def eval_on_env(cfg, encoder, policy, device, use_diffusion):
    n_envs = cfg.n_envs

    _, traj_dset = hydra.utils.call(
        cfg.env.dataset, num_hist=cfg.num_hist,
        num_pred=cfg.num_pred, frameskip=cfg.frameskip)
    dset = traj_dset["valid"]
    base = getattr(dset, "dataset", dset)  # init/goal states + action stats
    act_mean, act_std = base.action_mean.numpy(), base.action_std.numpy()

    # wall/deformable can't be subprocessed; use SerialVectorEnv
    if cfg.env.name in ("wall", "deformable_env"):
        from env.serial_vector_env import SerialVectorEnv
        env = SerialVectorEnv(
            [gym.make(cfg.env.name, *cfg.env.args, **cfg.env.kwargs) for _ in range(n_envs)])
    else:
        env = SubprocVectorEnv(
            [lambda: gym.make(cfg.env.name, *cfg.env.args, **cfg.env.kwargs)
             for _ in range(n_envs)])

    from datasets.img_transforms import default_transform
    tf = default_transform(cfg.img_size)

    def embed(visual_np):  # dict visual (b, H, W, C) -> (b, 1, P, E)
        # ascontiguousarray handles deformable's BGR-reversed (negative-stride) frames
        v = torch.as_tensor(np.ascontiguousarray(visual_np)).float()
        if v.max() > 1.5:                # auto-detect [0,255] vs [0,1] across envs
            v = v / 255.0
        v = rearrange(v, "b h w c -> b c h w")
        v = tf(v).to(device)
        return encoder(v).reshape(v.shape[0], 1, encoder.n_patches, encoder.emb_dim)

    def step_env(a):  # denormalize policy action -> env step
        obs, rew, done, info = env.step(a * act_std + act_mean)
        return obs, np.asarray(rew), np.asarray(done), info

    successes, state_dists, final_cov, max_cov, total_reward = [], [], [], [], 0.0
    n_batches = max(1, cfg.n_env_evals // n_envs)
    for b in range(n_batches):
        seeds = [cfg.seed + b * n_envs + i for i in range(n_envs)]
        idxs = [(b * n_envs + i) % len(dset) for i in range(n_envs)]
        init_states = np.stack([dset.get_frames(j, [0])[2][0].numpy() for j in idxs])
        last = [dset.get_seq_length(j) - 1 for j in idxs]
        # goal = trajectory's final state (task-completion target for eval_state)
        goal_states = np.stack([dset.get_frames(j, [last[k]])[2][0].numpy() for k, j in enumerate(idxs)])
        # goal-conditioned policy: embed the goal-image (same final frame) to condition on
        goal_emb = None
        if int(cfg.env.goal_dim) > 0:
            goal_imgs = torch.stack(
                [dset.get_frames(j, [last[k]])[0]["visual"][0] for k, j in enumerate(idxs)]
            ).to(device)  # (b, C, H, W) in [0,1]
            goal_emb = encoder(goal_imgs).reshape(len(idxs), 1, encoder.n_patches, encoder.emb_dim)

        obs, _ = env.prepare(seeds, init_states)
        obs_stack = deque([embed(obs["visual"])], maxlen=cfg.window_size)
        action_list, last_info, steps, done = [], None, 0, np.array([False])
        # eval_horizon counts TOTAL env steps (dino_wm envs never set done=True),
        # so diffusion and vqbet run equal-length episodes.
        while steps < cfg.eval_horizon and not np.all(done):
            obs_seq = torch.cat(list(obs_stack), dim=1)  # (b, T, P, E)
            # expand goal to the full window: diffusion pads obs to obs_horizon but
            # NOT the goal, so a shorter goal would mismatch in cat([goal, obs]).
            gseq = None if goal_emb is None else goal_emb.expand(-1, cfg.window_size, -1, -1)
            action, _, _ = policy(obs_seq, gseq, None)
            if use_diffusion:
                # diffusion returns (b, chunk, A): execute each action in the chunk
                for t in range(action.shape[1]):
                    obs, rew, done, info = step_env(action[:, t].cpu().numpy())
                    obs_stack.append(embed(obs["visual"]))
                    total_reward += float(rew.sum()); last_info = info; steps += 1
                    if np.all(done) or steps >= cfg.eval_horizon:
                        break
            else:
                # vqbet returns (b, T, chunk, A): temporal-ensemble the chunk
                if cfg.action_window_size > 1:
                    action_list.append(action[:, -1].cpu().numpy())  # (b, chunk, A)
                    if len(action_list) > cfg.action_window_size:
                        action_list = action_list[1:]
                    curr_action = np.array(action_list).mean(axis=0)[:, 0]  # (b, A)
                    action_list = [np.concatenate(
                        (c[:, 1:], np.zeros((c.shape[0], 1, c.shape[-1]))), axis=1) for c in action_list]
                else:
                    curr_action = action[:, -1, 0, :].cpu().numpy()
                obs, rew, done, info = step_env(curr_action)
                obs_stack.append(embed(obs["visual"]))
                total_reward += float(rew.sum()); last_info = info; steps += 1

        if last_info is not None and "final_coverage" in last_info[0]:
            final_cov += [last_info[i]["final_coverage"] for i in range(n_envs)]
            max_cov += [last_info[i]["max_coverage"] for i in range(n_envs)]
        if last_info is not None and "state" in last_info[0]:
            cur_states = np.stack([last_info[i]["state"] for i in range(n_envs)])
            res = env.eval_state(goal_states, cur_states)
            successes += np.asarray(res["success"]).astype(float).tolist()
            # distance key varies by env: state_dist (pusht/maze/wall),
            # chamfer_distance (deformable); collect whichever is present.
            dist_key = "state_dist" if "state_dist" in res else (
                "chamfer_distance" if "chamfer_distance" in res else None)
            if dist_key is not None:
                state_dists += np.asarray(res[dist_key]).astype(float).tolist()
    env.close()

    metrics = {"avg_reward": total_reward / cfg.n_env_evals}
    if successes:  # eval_state success (all envs)
        metrics["success_rate"] = float(np.mean(successes))
    if state_dists:  # distance-to-goal (key varies by env; absent if neither)
        metrics["mean_goal_dist"] = float(np.mean(state_dists))
    if final_cov:  # coverage (pusht)
        metrics["final_coverage_mean"] = float(np.mean(final_cov))
        metrics["max_coverage_mean"] = float(np.mean(max_cov))
        print("final coverage mean:", metrics["final_coverage_mean"])
    return metrics


@hydra.main(config_path="conf", config_name="train_policy", version_base="1.2")
def _main(cfg: OmegaConf):
    main(cfg)


if __name__ == "__main__":
    _main()
