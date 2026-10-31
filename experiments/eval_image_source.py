"""Same demo goals, goal image read from the dataset versus rendered by the env."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collections import deque

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

CKPT = sys.argv[1] if len(sys.argv) > 1 else (
    "/nas/ada/dev/dino_wm/dagger/wall/rounds/round_09/train/checkpoints/model_final.pt")
DATA = sys.argv[2] if len(sys.argv) > 2 else "/nas/ada/dev/dino/data/wall_single"
TARGET = float(sys.argv[3]) if len(sys.argv) > 3 else 18.4

GOAL_OFF = 25
HORIZON = 300
N_ENVS = 10
N_PAIRS = 200
TOL = 2.0
WALL_LO, WALL_HI = 29.0, 35.0

dev = "cuda"

root = OmegaConf.create({
    "encoder": OmegaConf.load("conf/encoder/dino_avg_pool.yaml"),
    "env": {"name": "wall", "act_dim": 2, "goal_dim": 384, "views": 1,
            "args": [], "kwargs": {}},
    "window_size": 5,
    "action_window_size": 5,
    "img_size": 224,
    "n_envs": N_ENVS,
    "serial_env": True,
    "env_start_method": None,
    "policy": OmegaConf.load("conf/policy/vqbet.yaml"),
})

enc = hydra.utils.instantiate(root.encoder).to(dev).eval()
for p in enc.parameters():
    p.requires_grad = False
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
base = getattr(d, "dataset", d)
act_mean, act_std = base.action_mean.numpy(), base.action_std.numpy()

print(f"reading {len(d)} valid trajectories ...")
traj_state, traj_img, traj_info = [], [], []
for j in range(len(d)):
    img, _, st, info = d.get_frames(j, list(range(d.get_seq_length(j))))
    traj_state.append(np.asarray(st))
    traj_img.append(img["visual"])
    traj_info.append(info)
LAYOUT = traj_info[0]

pairs = []
for j, st in enumerate(traj_state):
    for off in range(len(st) - GOAL_OFF):
        i_xy, g_xy = st[off][:2], st[off + GOAL_OFF][:2]
        dist = float(np.linalg.norm(g_xy - i_xy))
        if abs(dist - TARGET) > TOL:
            continue
        pairs.append((i_xy, g_xy, traj_img[j][off + GOAL_OFF], dist))
rng = np.random.RandomState(0)
rng.shuffle(pairs)
pairs = pairs[:N_PAIRS]
print(f"{len(pairs)} demo pairs at distance {TARGET} +/- {TOL}\n")

env = tp.make_eval_env(root, N_ENVS)


@torch.no_grad()
def embed(visual_np):
    v = torch.as_tensor(np.ascontiguousarray(visual_np)).float()
    if v.max() > 1.5:
        v = v / 255.0
    if v.shape[-1] in (1, 3):
        v = v.permute(0, 3, 1, 2)
    return enc(default_transform(224)(v).to(dev)).reshape(
        v.shape[0], 1, enc.n_patches, enc.emb_dim)


@torch.no_grad()
def run(use_env_render, label):
    ever, final, closest = [], [], []
    for b in range(0, len(pairs), N_ENVS):
        blk = pairs[b:b + N_ENVS]
        real = len(blk)
        if real < N_ENVS:
            blk = blk + [blk[-1]] * (N_ENVS - real)
        init_states = np.stack([c[0] for c in blk])
        goal_states = np.stack([c[1] for c in blk])
        env.update_env([LAYOUT] * N_ENVS)
        seeds = list(range(b, b + N_ENVS))

        # the only DIFFERENCE between the two arms
        if use_env_render:
            goal_obs, _ = env.prepare(seeds, goal_states)
            g = embed(np.asarray(goal_obs["visual"]))
        else:
            g = enc(torch.stack([c[2] for c in blk]).to(dev)).reshape(
                N_ENVS, 1, enc.n_patches, enc.emb_dim)

        obs, _ = env.prepare(seeds, init_states)
        stack = deque([embed(obs["visual"])], maxlen=root.window_size)
        e = np.zeros(N_ENVS, dtype=bool)
        best = np.full(N_ENVS, np.inf)
        last, steps, action_list = None, 0, []
        aw = int(root.action_window_size)
        while steps < HORIZON:
            seq = torch.cat(list(stack), dim=1)
            a, _, _ = pol(seq, g.expand(-1, root.window_size, -1, -1), None)
            if aw > 1:
                action_list.append(a[:, -1].cpu().numpy())
                if len(action_list) > aw:
                    action_list = action_list[1:]
                curr = np.array(action_list).mean(axis=0)[:, 0]
                action_list = [np.concatenate(
                    (c[:, 1:], np.zeros((c.shape[0], 1, c.shape[-1]))), axis=1)
                    for c in action_list]
            else:
                curr = a[:, -1, 0, :].cpu().numpy()
            obs, _, _, info = env.step(curr * act_std + act_mean)
            cur = np.stack([info[i]["state"] for i in range(N_ENVS)])
            r = env.eval_state(goal_states, cur)
            sc = np.asarray(r["success"]).astype(bool)
            e |= sc
            best = np.minimum(best, np.asarray(r["state_dist"]))
            last = sc
            steps += 1
            stack.append(embed(obs["visual"]))
        ever += e[:real].tolist()
        final += last[:real].tolist()
        closest += best[:real].tolist()
    ever, final = np.array(ever, float), np.array(final, float)
    print(f"  {label:<34} n={len(ever):3d}  max={ever.mean():.2f}  "
          f"final={final.mean():.2f}  closest={np.median(closest):4.1f}", flush=True)
    return ever.mean(), float(np.median(closest))


a_max, a_close = run(False, "goal image from the DATASET file")
b_max, b_close = run(True, "goal image RENDERED by the env")

print(f"\n  difference in max_success: {b_max - a_max:+.2f}")
print(f"  difference in closest:     {b_close - a_close:+.2f} world units")
env.close()
