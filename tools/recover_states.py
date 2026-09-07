"""Replay every trajectory through the env and save the resulting states as states_true.pth."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from omegaconf import OmegaConf

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "/nas/ada/dev/dino/data/wall_single")
N_ENVS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
IMG = 224
VERIFY = 40   # trajectories to check against the stored images

states = np.asarray(torch.load(DATA / "states.pth"))[..., :2].astype(np.float64)
actions = np.asarray(torch.load(DATA / "actions.pth")).astype(np.float64)
n_traj, T, _ = states.shape
print(f"{n_traj} trajectories of {T} frames from {DATA}")

import train_policy as tp
from datasets.wall_dset import WallDataset
from datasets.img_transforms import default_transform

# per-trajectory maze layout, needed before each rollout
dset = WallDataset(n_rollout=None, transform=default_transform(IMG),
                   data_path=str(DATA), normalize_action=False)

root = OmegaConf.create({
    "env": {"name": "wall", "act_dim": 2, "goal_dim": 384, "views": 1,
            "args": [], "kwargs": {}},
    "n_envs": N_ENVS, "img_size": IMG,
    "serial_env": True, "env_start_method": None,
})
env = tp.make_eval_env(root, N_ENVS)


def gray(img):
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = a.mean(axis=0)
    elif a.ndim == 3:
        a = a.mean(axis=-1)
    if a.max() > 1.5:
        a = a / 255.0
    return a


true_states = np.zeros_like(states)
verify_err, verify_n = [], 0
rng = np.random.RandomState(0)
to_verify = set(rng.choice(n_traj, size=min(VERIFY, n_traj), replace=False).tolist())

for b in range(0, n_traj, N_ENVS):
    idx = list(range(b, min(b + N_ENVS, n_traj)))
    pad = N_ENVS - len(idx)
    blk = idx + [idx[-1]] * pad
    infos = [dset.get_frames(j, [0])[3] for j in blk]
    env.update_env(infos)
    obses, rs = env.rollout(list(range(N_ENVS)),
                            states[blk, 0],
                            actions[blk, :T - 1])
    rs = np.asarray(rs)[..., :2]
    for k, j in enumerate(idx):
        true_states[j] = rs[k, :T]
    # spot-check the replayed frames against what is stored on disk
    vis = np.asarray(obses["visual"])
    for k, j in enumerate(idx):
        if j not in to_verify:
            continue
        stored = dset.get_frames(j, list(range(T)))[0]["visual"]
        for t in range(0, T, 7):
            verify_err.append(float(np.abs(gray(stored[t]) - gray(vis[k, t])).sum()))
        verify_n += 1
    if b % (N_ENVS * 20) == 0:
        print(f"  {min(b + N_ENVS, n_traj)}/{n_traj}", flush=True)

env.close()

verify_err = np.array(verify_err)
drift = np.linalg.norm(true_states - states, axis=-1)
print()
print(f"  verified {verify_n} trajectories against their stored images")
print(f"    median |replay - stored| {np.median(verify_err):8.3f}   max {verify_err.max():8.3f}")
print(f"  recovered vs recorded states")
print(f"    frame 0   median {np.median(drift[:, 0]):6.2f}")
print(f"    frame {T // 2:<3} median {np.median(drift[:, T // 2]):6.2f}")
print(f"    frame {T - 1:<3} median {np.median(drift[:, -1]):6.2f}")
print(f"    overall   median {np.median(drift):6.2f}   max {drift.max():6.2f}")

out = DATA / "states_true.pth"
torch.save(torch.from_numpy(true_states.astype(np.float32)), out)
print(f"\nwrote {out}")
