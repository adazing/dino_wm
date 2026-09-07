"""Standalone timing/diagnostic for one planning round (the original plan.py flow)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
import time
import gym
import hydra
import torch
import warnings
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
from plan import load_model, PlanWorkspace
from utils import cfg_to_dict, seed as set_seed

warnings.filterwarnings("ignore")


@hydra.main(config_path="../conf", config_name="plan")
def main(cfg):
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict["wandb_logging"] = False
    cfg_dict["saved_folder"] = os.getcwd()

    # 1.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"[device] CUDA OK -> {torch.cuda.get_device_name(0)} (torch CUDA {torch.version.cuda})")
    else:
        print("[device] CUDA not available, running on CPU. This alone explains a "
              "10-50x slowdown; fix the env/GPU allocation before anything else. ***")

    # 2.
    model_path = os.path.abspath(os.path.join(cfg_dict["ckpt_base_path"], "outputs", cfg_dict["model_name"]))
    with open(os.path.join(model_path, "hydra.yaml")) as f:
        model_cfg = OmegaConf.load(f)
    override_path = cfg_dict.get("override_dataset_path")
    if override_path:
        with open_dict(model_cfg):
            model_cfg.env.dataset.data_path = override_path
    set_seed(cfg_dict["seed"])

    # 3.
    t = time.perf_counter()
    _, dset = hydra.utils.call(model_cfg.env.dataset, num_hist=model_cfg.num_hist,
                               num_pred=model_cfg.num_pred, frameskip=model_cfg.frameskip)
    dset = dset["valid"]
    print(f"[time] dataset load: {time.perf_counter() - t:.1f}s  (len={len(dset)} trajectories)")

    # 4.
    t = time.perf_counter()
    model = load_model(Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth",
                       model_cfg, model_cfg.num_action_repeat, device=device)
    print(f"[time] model load: {time.perf_counter() - t:.1f}s  (params on {next(model.parameters()).device})")

    env = SubprocVectorEnv([
        lambda: gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)
        for _ in range(cfg_dict["n_evals"])
    ])

    # 5. PlanWorkspace setup, includes sample_traj_segment_from_dset (the dataset decode).
    t = time.perf_counter()
    ws = PlanWorkspace(cfg_dict=cfg_dict, wm=model, dset=dset, env=env,
                       env_name=model_cfg.env.name, frameskip=model_cfg.frameskip, wandb_run=None)
    print(f"[time] PlanWorkspace setup (dataset sampling): {time.perf_counter() - t:.1f}s")

    p = cfg_dict["planner"]
    ns = int(p.get("num_samples", 300))
    opt_steps = int(p.get("opt_steps", 30))
    print(f"[cfg] n_evals={cfg_dict['n_evals']} planner={p.get('name')} opt_steps={opt_steps} "
          f"num_samples={ns} horizon={ws.planner.horizon} eval_every={p.get('eval_every')}")

    # everything below is inference-only, no_grad so the batch-300 rollouts don't retain
    # activations for backprop, that's what OOM'd, the CEM itself runs under no_grad too.
    with torch.no_grad():
        # micro-benchmark one wm.rollout, the exact unit the CEM repeats opt_steps*n_evals times
        from utils import move_to_device
        from einops import repeat as _repeat
        trans = move_to_device(ws.data_preprocessor.transform_obs(ws.obs_0), device)
        cur = {k: _repeat(v[0:1], "1 ... -> n ...", n=ns) for k, v in trans.items()}
        act = torch.randn(ns, ws.planner.horizon, ws.action_dim, device=device)
        ws.wm.rollout(cur, act)   # warmup (load kernels)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        R = 5
        for _ in range(R):
            ws.wm.rollout(cur, act)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        per = (time.perf_counter() - t) / R
        n_roll = opt_steps * int(cfg_dict["n_evals"])
        print(f"[bench] one wm.rollout(batch={ns}, horizon={ws.planner.horizon}): {per * 1000:.0f} ms")
        print(f"[bench] CEM runs ~{n_roll} of these -> predicted pure-CEM ~{per * n_roll / 60:.1f} min")

        # full pure-CEM timing (per-step evaluator off)
        ws.planner.evaluator = None
        t = time.perf_counter()
        ws.planner.plan(obs_0=ws.obs_0, obs_g=ws.obs_g, actions=None)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t
        print(f"[time] planner.plan() [pure CEM, evaluator OFF]: {dt:.1f}s  ({dt / 60:.1f} min)")
    env.close()


if __name__ == "__main__":
    main()
