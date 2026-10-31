"""Compare goal25 pairs that cross the wall against distance matched pairs that do not."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import random
from collections import deque

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

CKPT = sys.argv[1] if len(sys.argv) > 1 else (
    "/nas/ada/dev/dino_wm/dagger/wall/rounds/round_09/train/checkpoints/model_final.pt")
DATA = sys.argv[2] if len(sys.argv) > 2 else "/nas/ada/dev/dino/data/wall_single"

GOAL_OFF = 25
HORIZON = 300
N_ENVS = 10
WALL_LO, WALL_HI = 29.0, 35.0   # wall spans x in [29, 35]

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
act_mean = base.action_mean.numpy()
act_std = base.action_std.numpy()

# read every valid-split trajectory once and keep its states, images and layout
print(f"reading {len(d)} valid trajectories ...")
traj_state, traj_img, traj_info = [], [], []
for j in range(len(d)):
    T = d.get_seq_length(j)
    img, _, st, info = d.get_frames(j, list(range(T)))
    traj_state.append(np.asarray(st))
    traj_img.append(img["visual"])
    traj_info.append(info)

side = lambda x: -1 if x < WALL_LO else (1 if x > WALL_HI else 0)

cross_pairs, same_pairs = [], []
for j, st in enumerate(traj_state):
    for off in range(len(st) - GOAL_OFF):
        i_xy, g_xy = st[off][:2], st[off + GOAL_OFF][:2]
        dist = float(np.linalg.norm(g_xy - i_xy))
        si, sg = side(i_xy[0]), side(g_xy[0])
        rec = (j, off, dist)
        if si and sg and si != sg:
            cross_pairs.append(rec)
        elif si and sg and si == sg:
            same_pairs.append(rec)

print(f"cross-wall pairs in valid split: {len(cross_pairs)}")
print(f"same-side pairs in valid split:  {len(same_pairs)}")
if not cross_pairs:
    sys.exit("no cross-wall pairs in this split")

# distance-matched control for each cross-wall pair take the nearest-distance same-side pair
rng = random.Random(0)
same_sorted = sorted(same_pairs, key=lambda r: r[2])
same_d = np.array([r[2] for r in same_sorted])
used, control = set(), []
for _, _, dist in cross_pairs:
    k = int(np.searchsorted(same_d, dist))
    for cand in sorted(range(len(same_sorted)), key=lambda c: abs(c - k)):
        if cand not in used:
            used.add(cand)
            control.append(same_sorted[cand])
            break

print(f"control group: {len(control)} same-side pairs, "
      f"median dist {np.median([r[2] for r in control]):.1f} vs "
      f"cross {np.median([r[2] for r in cross_pairs]):.1f}")

env = tp.make_eval_env(root, N_ENVS)


@torch.no_grad()
def embed(visual_np):
    v = torch.as_tensor(np.ascontiguousarray(visual_np)).float()
    if v.max() > 1.5:
        v = v / 255.0
    if v.shape[-1] in (1, 3):
        v = v.permute(0, 3, 1, 2)
    v = default_transform(224)(v).to(dev)
    return enc(v).reshape(v.shape[0], 1, enc.n_patches, enc.emb_dim)


@torch.no_grad()
def run(pairs, label):
    ever, final, closest, dists = [], [], [], []
    for b in range(0, len(pairs), N_ENVS):
        blk = pairs[b:b + N_ENVS]
        if len(blk) < N_ENVS:   # pad, trimmed below
            blk = blk + [blk[-1]] * (N_ENVS - len(blk))
        real = min(N_ENVS, len(pairs) - b)
        init_states = np.stack([traj_state[j][off][:2] for j, off, _ in blk])
        goal_states = np.stack([traj_state[j][off + GOAL_OFF][:2] for j, off, _ in blk])
        goal_imgs = torch.stack([traj_img[j][off + GOAL_OFF] for j, off, _ in blk])
        with torch.no_grad():
            g = enc(goal_imgs.to(dev)).reshape(len(blk), 1, enc.n_patches, enc.emb_dim)
        env.update_env([traj_info[j] for j, _, _ in blk])
        seeds = list(range(b, b + N_ENVS))
        obs, _ = env.prepare(seeds, init_states)

        stack = deque([embed(obs["visual"])], maxlen=root.window_size)
        e = np.zeros(N_ENVS, dtype=bool)
        best = np.full(N_ENVS, np.inf)
        last = None
        steps = 0
        action_list = []
        aw = int(root.action_window_size)
        while steps < HORIZON:
            seq = torch.cat(list(stack), dim=1)
            # vqbet returns, b, T, chunk, A.
            a, _, _ = pol(seq, g.expand(-1, root.window_size, -1, -1), None)
            if aw > 1:
                action_list.append(a[:, -1].cpu().numpy())   # (b, chunk, A)
                if len(action_list) > aw:
                    action_list = action_list[1:]
                curr = np.array(action_list).mean(axis=0)[:, 0]   # (b, A)
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
        dists += [x[2] for x in blk[:real]]
        print(f"  [{label}] {b + real}/{len(pairs)} done", flush=True)

    ever, final = np.array(ever, dtype=float), np.array(final, dtype=float)
    print(f"\n{label}: n={len(ever)}  max_succ={ever.mean():.2f}  "
          f"final_succ={final.mean():.2f}  median closest={np.median(closest):.1f}  "
          f"median goal_dist={np.median(dists):.1f}")
    return ever.mean()


print()
c = run(cross_pairs, "CROSS-WALL")
s = run(control, "same-side (distance-matched)")
print(f"\ndifference: cross-wall - same-side = {c - s:+.2f}")
env.close()
