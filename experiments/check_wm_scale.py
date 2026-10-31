"""Check whether the world model predicts 1x or 2x the recorded displacement."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

MODEL = sys.argv[1] if len(sys.argv) > 1 else "2026-08-14/13-07-36"
BASE = sys.argv[2] if len(sys.argv) > 2 else "/nas/ada/dev/dino_wm"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 24
DATA = "/nas/ada/dev/dino/data/wall_single"

dev = "cuda"

wm_dir = Path(BASE) / "outputs" / MODEL
train_cfg = OmegaConf.load(wm_dir / "hydra.yaml")
frameskip = int(train_cfg.frameskip)
num_hist = int(train_cfg.num_hist)
print(f"world model {wm_dir}")
print(f"  frameskip={frameskip}  num_hist={num_hist}")

import plan as P
ckpts = sorted((wm_dir / "checkpoints").glob("model_*.pth"))
ckpt = max(ckpts, key=lambda p: (p.stem != "model_latest", p.stem))
model = P.load_model(ckpt, train_cfg, num_action_repeat=1, device=dev)
model.to(dev)
model.eval()
print(f"  checkpoint {ckpt.name}")

import train_policy as tp
from datasets.wall_dset import load_wall_slice_train_val
from datasets.img_transforms import default_transform

_, traj = load_wall_slice_train_val(
    transform=default_transform(224), data_path=DATA,
    n_rollout=None, normalize_action=True, split_mode="random")
d = traj["valid"]
base = getattr(d, "dataset", d)
act_mean = base.action_mean.numpy()
act_std = base.action_std.numpy()
print(f"  action_mean={act_mean}  action_std={act_std}")

root = OmegaConf.create({
    "env": {"name": "wall", "act_dim": 2, "goal_dim": 384, "views": 1,
            "args": [], "kwargs": {}},
    "n_envs": 1, "img_size": 224,
    "serial_env": True, "env_start_method": None,
})
env = tp.make_eval_env(root, 1)

# same preprocessor plan.py builds, from the demo dataset's own statistics
from utils.preprocessor import Preprocessor
prep = Preprocessor(
    action_mean=base.action_mean, action_std=base.action_std,
    state_mean=base.state_mean, state_std=base.state_std,
    proprio_mean=base.proprio_mean, proprio_std=base.proprio_std,
    transform=base.transform,
)


def obs_at(state, info):
    """Raw env observation at a position, shaped (b, t, H, W, C) the way plan.py builds obs_0."""
    env.update_env([info])
    o, _ = env.prepare([0], np.asarray(state)[None])
    v = np.asarray(o["visual"])   # (1, H, W, C) in [0, 255]
    return {"visual": np.expand_dims(v, axis=1),   # (1, 1, H, W, C)
            "proprio": np.asarray(state, dtype=np.float32)[None, None]}


CHANNELS = ("visual", "proprio")

rng = np.random.RandomState(0)
wins = {ch: {"1x": 0, "2x": 0, "stay": 0} for ch in CHANNELS}
rows = []

for _ in range(N):
    j = int(rng.randint(len(d)))
    T = d.get_seq_length(j)
    t = int(rng.randint(0, T - frameskip - 1))
    _, act, st, info = d.get_frames(j, list(range(t, t + frameskip + 1)))
    state = np.asarray(st[0])[:2]
    # the raw per-step actions this chunk applies, undoing the dataset normalisation
    raw = np.asarray(act[:frameskip]) * act_std + act_mean
    disp = raw.sum(axis=0)   # dataset dynamics pos + sum(action)
    if not (10 < state[0] < 26 or 38 < state[0] < 54) or not (10 < state[1] < 54):
        continue
    cand = {"stay": state, "1x": state + disp, "2x": state + 2 * disp}
    if any(not (2 < c[0] < 62 and 2 < c[1] < 62) for c in cand.values()):
        continue

    o0 = obs_at(state, info)
    # the model consumes normalised, frameskip-chunked actions (b, T, frameskip*act_dim)
    a_chunk = torch.as_tensor(np.asarray(act[:frameskip]).reshape(1, 1, -1)).float().to(dev)
    a_in = torch.cat([torch.zeros_like(a_chunk), a_chunk], dim=1)
    with torch.no_grad():
        o_in = prep.transform_obs(o0)
        o_in = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in o_in.items()}
        z_obs, _ = model.rollout(o_in, a_in)
        pred = {ch: z_obs[ch][:, -1].flatten() for ch in CHANNELS}

        enc = {}
        for k, c in cand.items():
            oc = obs_at(c, info)
            oc_in = prep.transform_obs(oc)
            oc_in = {kk: (v.to(dev) if hasattr(v, "to") else v) for kk, v in oc_in.items()}
            zc, _ = model.rollout(oc_in, torch.zeros_like(a_in))
            enc[k] = {ch: zc[ch][:, 0].flatten() for ch in CHANNELS}

    dists = {ch: {k: float((pred[ch] - enc[k][ch]).norm()) for k in cand} for ch in CHANNELS}
    # the same units the planner optimises in, mean squared error per element
    loss = {ch: {k: float(((pred[ch] - enc[k][ch]) ** 2).mean()) for k in cand} for ch in CHANNELS}
    for ch in CHANNELS:
        wins[ch][min(dists[ch], key=dists[ch].get)] += 1
    rows.append((float(np.linalg.norm(disp)), dists, loss))

print(f"\nn = {len(rows)} usable samples, one world-model step each "
      f"({frameskip} env actions per step)\n")
print(f"  median displacement of one chunk: {np.median([r[0] for r in rows]):.2f} world units")

for ch in CHANNELS:
    print(f"  {ch} channel")
    for k in ("stay", "1x", "2x"):
        m = np.median([r[1][ch][k] for r in rows])
        print(f"    distance to {k:<5}  median {m:9.3f}   won {wins[ch][k]:3d} / {len(rows)}")
    print(f"    winner {max(wins[ch], key=wins[ch].get)}")
    print()

v_pull = np.median([r[2]["visual"]["1x"] - r[2]["visual"]["2x"] for r in rows])
p_pull = np.median([r[2]["proprio"]["2x"] - r[2]["proprio"]["1x"] for r in rows])
print("  objective units, mean squared error per element, as planning/objectives.py computes it")
print(f"    visual  term prefers 2x by {v_pull:.6f}")
print(f"    proprio term prefers 1x by {p_pull:.6f}")
print(f"    at alpha=1 the proprio term fights the visual term with "
      f"{100 * p_pull / max(v_pull, 1e-12):.0f}% of its strength")
env.close()
