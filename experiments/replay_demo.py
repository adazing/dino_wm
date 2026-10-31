"""Replay a demo's actions through the env and compare the frames to the stored images."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import numpy as np
import torch
from omegaconf import OmegaConf

DATA = sys.argv[1] if len(sys.argv) > 1 else "/nas/ada/dev/dino/data/wall_single"
NTRAJ = int(sys.argv[2]) if len(sys.argv) > 2 else 8

IMG = 224

import train_policy as tp
from datasets.wall_dset import load_wall_slice_train_val
from datasets.img_transforms import default_transform

# pinned to the shipped array on purpose.
_, traj = load_wall_slice_train_val(
    states_file="states.pth",
    transform=default_transform(IMG), data_path=DATA,
    n_rollout=None, normalize_action=True, split_mode="random")
d = traj["valid"]
base = getattr(d, "dataset", d)
act_mean = base.action_mean.numpy()
act_std = base.action_std.numpy()

root = OmegaConf.create({
    "env": {"name": "wall", "act_dim": 2, "goal_dim": 384, "views": 1,
            "args": [], "kwargs": {}},
    "n_envs": 1, "img_size": IMG,
    "serial_env": True, "env_start_method": None,
})
env = tp.make_eval_env(root, 1)


def gray(img):
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = a.mean(axis=0)
    elif a.ndim == 3:
        a = a.mean(axis=-1)
    if a.max() > 1.5:
        a = a / 255.0
    return a


rng = np.random.RandomState(0)
results = {"raw (env makes it 2x)": [], "halved (env makes it 1x)": []}
state_err = {"raw (env makes it 2x)": [], "halved (env makes it 1x)": []}
baseline = []

for _ in range(NTRAJ):
    j = int(rng.randint(len(d)))
    T = d.get_seq_length(j)
    img, act, st, info = d.get_frames(j, list(range(T)))
    states = np.asarray(st)[:, :2]
    raw = np.asarray(act) * act_std + act_mean   # the dataset's own displacements
    stored = [gray(img["visual"][t]) for t in range(T)]

    # scale of a genuinely different frame, for reference
    baseline += [float(np.abs(stored[t] - stored[t + 5]).sum()) for t in range(0, T - 5, 7)]

    for label, scale in (("raw (env makes it 2x)", 1.0), ("halved (env makes it 1x)", 0.5)):
        env.update_env([info])
        # SerialVectorEnv.rollout(seed, init_state (n, D), actions (n, T, D)) seed is indexed per
        # env, so it has to be a list not a scalar
        obses, rstates = env.rollout([0], states[0][None], (raw[:T - 1] * scale)[None])
        vis = np.asarray(obses["visual"])[0]   # , T, H, W, C
        rs = np.asarray(rstates)[0]   # (T, D)
        n = min(len(stored), len(vis))
        results[label] += [float(np.abs(stored[t] - gray(vis[t])).sum()) for t in range(n)]
        state_err[label] += [float(np.linalg.norm(rs[t][:2] - states[t])) for t in range(n)]

print(f"\n{NTRAJ} trajectories replayed through the env\n")
print(f"  {'action scaling':<26}{'|replay - stored image|':>26}{'|replay - states|':>20}")
for label in results:
    print(f"  {label:<26}{np.median(results[label]):>26.1f}"
          f"{np.median(state_err[label]):>20.2f}")
print(f"\n  reference, two frames 5 apart in the same trajectory: "
      f"{np.median(baseline):.1f}")
print()
print(f"  closest match {min(results, key=lambda k: np.median(results[k]))}")
env.close()
