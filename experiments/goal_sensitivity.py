"""Measure whether the policy's action changes when the goal image changes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

CKPT = sys.argv[1] if len(sys.argv) > 1 else (
    "/nas/ada/dev/dino_wm/policy_outputs/"
    "20260810005046_wall_dino_avg_pool_vqbet_original_gcTrue/checkpoints/model_300.pt")
DATA = sys.argv[2] if len(sys.argv) > 2 else "/nas/ada/dev/dino/data/wall_single"

dev = "cuda"

# Compose only what conf/policy/vqbet.yaml interpolates.
root = OmegaConf.create({
    "encoder": OmegaConf.load("conf/encoder/dino_avg_pool.yaml"),
    "env": {"act_dim": 2, "goal_dim": 384, "views": 1},   # goal_dim is encoder.output_dim
    "window_size": 5,
    "action_window_size": 5,
    "policy": OmegaConf.load("conf/policy/vqbet.yaml"),
})

enc = hydra.utils.instantiate(root.encoder).to(dev).eval()
pol = hydra.utils.instantiate(root.policy).to(dev)

import train_policy as tp
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
pol.load_state_dict(tp._fix_vqvae_keys(ck["model"]))
pol.eval()
print(f"loaded {CKPT} (epoch {ck.get('epoch')})")

from datasets.wall_dset import load_wall_slice_train_val
from datasets.img_transforms import default_transform
_, traj = load_wall_slice_train_val(
    transform=default_transform(224), data_path=DATA,
    n_rollout=None, normalize_action=True, split_mode="random")
d = traj["valid"]


def emb(frames):
    """dataset visual (T, C, H, W) in [0, 1] -> (1, window, P, E), matching eval_on_env."""
    with torch.no_grad():
        e = enc(frames.to(dev))
    e = e.reshape(1, 1, enc.n_patches, enc.emb_dim)
    return e.expand(-1, root.window_size, -1, -1)


@torch.no_grad()
def act(obs_seq, goal_seq):
    a, _, _ = pol(obs_seq, goal_seq, None)
    return a


def state_xy(j, t):
    return np.asarray(d.get_frames(j, [t])[2][0])[:2]


_T = np.array([d.get_seq_length(j) for j in range(len(d))])
print()
print(f"trajectory length over {len(d)} valid demos: "
      f"min={_T.min()} median={int(np.median(_T))} max={_T.max()}")
print("eval_on_env clamps the goal to frame T-1, and when T <= goal_off+1 it also forces")
print("init_off=0, so a goal_offset past the demo length becomes 'frame 0 to final frame'.")

OBS_T = 5
OFFSETS = [10, 25, 45, "end"]
N = 50

print()
print(f"{'goal offset':>12}  {'|action|':>9}  {'same-goal':>10}  {'swapped':>9}  "
      f"{'swap/noise':>11}  {'goal dist':>9}  {'n':>4}")
print("-" * 78)

rng = np.random.RandomState(0)
for off in OFFSETS:
    diff_goal, diff_same, mags, dists = [], [], [], []
    tries = 0
    while len(mags) < N and tries < N * 20:
        tries += 1
        j1, j2 = rng.randint(len(d)), rng.randint(len(d))
        T1, T2 = d.get_seq_length(j1), d.get_seq_length(j2)
        if off == "end":
            g1t, g2t = T1 - 1, T2 - 1
        else:
            g1t, g2t = OBS_T + off, OBS_T + off
            if g1t >= T1 or g2t >= T2:
                continue
        if OBS_T >= T1:
            continue
        o1 = d.get_frames(j1, [OBS_T])[0]["visual"]
        g1 = d.get_frames(j1, [g1t])[0]["visual"]
        g2 = d.get_frames(j2, [g2t])[0]["visual"]
        obs = emb(o1)
        a1, a2 = act(obs, emb(g1)), act(obs, emb(g2))
        a1b = act(obs, emb(g1))   # control, same goal twice
        diff_goal.append(float((a1 - a2).abs().mean()))
        diff_same.append(float((a1 - a1b).abs().mean()))
        mags.append(float(a1.abs().mean()))
        dists.append(float(np.linalg.norm(state_xy(j1, g1t) - state_xy(j1, OBS_T))))

    if not mags:
        print(f"{str(off):>12}  {'no demo is long enough for this offset':^58}")
        continue
    m, sw, sm = np.mean(mags), np.mean(diff_goal), np.mean(diff_same)
    print(f"{str(off):>12}  {m:9.4f}  {sm:10.4f}  {sw:9.4f}  "
          f"{sw / max(1e-9, sm):11.2f}  {np.mean(dists):9.2f}  {len(mags):4d}")
