"""Success rate by goal source and wall crossing, at one fixed goal distance."""

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
N_PER_CELL = 150
TOL = 2.0   # accept pairs within TOL of TARGET

# geometry, from DEFAULT_CFG img_size 65, wall_x 32, wall_width 6, dot_std 1.7, border_wall_loc 5
# -> padding 3.4, border_padding 7.4
WALL_LO, WALL_HI = 29.0, 35.0
# generate_random_state's own bands every goal is >=7.4 from a border and >=3.4 from the wall
LEFT = (7.4, 25.6)
RIGHT = (38.4, 56.6)
YRANGE = (7.4, 56.6)
# the same bands with the padding removed, so env goals can sit against the wall or a border
# exactly as demo states can
LEFT_RAW = (1.0, 28.0)
RIGHT_RAW = (36.0, 63.0)
YRANGE_RAW = (1.0, 63.0)


def clearance(gx, gy):
    """Distance from a point to the nearest solid thing (border or wall, door excepted)."""
    d_border = min(gx, gy, 64 - gx, 64 - gy)
    in_door = 26 <= gy <= 34
    d_wall = 99.0 if in_door else (0.0 if WALL_LO <= gx <= WALL_HI
                                   else min(abs(gx - WALL_LO), abs(gx - WALL_HI)))
    return min(d_border, d_wall)

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

side = lambda x: -1 if x < WALL_LO else (1 if x > WALL_HI else 0)


def demo_cells():
    """(init_xy, goal_xy, goal_image) triples from demo pairs, split by wall crossing."""
    cross, same = [], []
    for j, st in enumerate(traj_state):
        for off in range(len(st) - GOAL_OFF):
            i_xy, g_xy = st[off][:2], st[off + GOAL_OFF][:2]
            dist = float(np.linalg.norm(g_xy - i_xy))
            if abs(dist - TARGET) > TOL:
                continue
            si, sg = side(i_xy[0]), side(g_xy[0])
            if not (si and sg):
                continue
            rec = (i_xy, g_xy, traj_img[j][off + GOAL_OFF], dist)
            (cross if si != sg else same).append(rec)
    return cross, same


def env_cells(rng, padded=True):
    """(init_xy, goal_xy, None) pairs drawn from the env's bands at TARGET distance."""
    L, R, Y = ((LEFT, RIGHT, YRANGE) if padded else (LEFT_RAW, RIGHT_RAW, YRANGE_RAW))
    cross, same = [], []
    tries = 0
    while (len(cross) < N_PER_CELL or len(same) < N_PER_CELL) and tries < 4_000_000:
        tries += 1
        band_i = L if rng.random() < 0.5 else R
        want_cross = rng.random() < 0.5
        band_g = (R if band_i is L else L) if want_cross else band_i
        i_xy = np.array([rng.uniform(*band_i), rng.uniform(*Y)])
        g_xy = np.array([rng.uniform(*band_g), rng.uniform(*Y)])
        dist = float(np.linalg.norm(g_xy - i_xy))
        if abs(dist - TARGET) > TOL:
            continue
        rec = (i_xy, g_xy, None, dist)
        if want_cross and len(cross) < N_PER_CELL:
            cross.append(rec)
        elif not want_cross and len(same) < N_PER_CELL:
            same.append(rec)
    return cross, same


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
def run(cells, label):
    cells = cells[:N_PER_CELL]
    ever, final, closest, dists = [], [], [], []
    for b in range(0, len(cells), N_ENVS):
        blk = cells[b:b + N_ENVS]
        real = len(blk)
        if real < N_ENVS:
            blk = blk + [blk[-1]] * (N_ENVS - real)
        init_states = np.stack([c[0] for c in blk])
        goal_states = np.stack([c[1] for c in blk])
        env.update_env([LAYOUT] * N_ENVS)
        seeds = list(range(b, b + N_ENVS))

        if blk[0][2] is None:
            # env-sampled goal render it, same order reach_rs uses (goal, then init)
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
        dists += [c[3] for c in blk[:real]]
    ever, final = np.array(ever, float), np.array(final, float)
    cl = np.median([clearance(c[1][0], c[1][1]) for c in cells])
    print(f"  {label:<30} n={len(ever):3d}  max={ever.mean():.2f}  "
          f"final={final.mean():.2f}  closest={np.median(closest):4.1f}  "
          f"dist={np.median(dists):4.1f}  goal_clearance={cl:4.1f}", flush=True)
    return ever.mean()


rng = np.random.RandomState(0)
d_cross, d_same = demo_cells()
e_cross, e_same = env_cells(rng, padded=True)
r_cross, r_same = env_cells(rng, padded=False)
print(f"\ntarget distance {TARGET} +/- {TOL}")
print(f"demo pairs available:      cross={len(d_cross)} same={len(d_same)}")
print(f"env padded pairs sampled:  cross={len(e_cross)} same={len(e_same)}")
print(f"env unpadded pairs sampled: cross={len(r_cross)} same={len(r_same)}\n")

res = {}
res["demo/same"] = run(d_same, "demo goal, same-side")
res["demo/cross"] = run(d_cross, "demo goal, cross-wall")
res["env/same"] = run(e_same, "env goal (padded), same-side")
res["env/cross"] = run(e_cross, "env goal (padded), cross-wall")
res["raw/same"] = run(r_same, "env goal (no clearance), same")
res["raw/cross"] = run(r_cross, "env goal (no clearance), cross")

avg = lambda a, b: (res[a] + res[b]) / 2
print("\n              demo goal   env padded   env unpadded")
print(f"same-side       {res['demo/same']:.2f}        {res['env/same']:.2f}"
      f"         {res['raw/same']:.2f}")
print(f"cross-wall      {res['demo/cross']:.2f}        {res['env/cross']:.2f}"
      f"         {res['raw/cross']:.2f}")
print(f"\nclearance effect (padded - unpadded): "
      f"{avg('env/same', 'env/cross') - avg('raw/same', 'raw/cross'):+.2f}")
print(f"source effect    (env unpadded - demo): "
      f"{avg('raw/same', 'raw/cross') - avg('demo/same', 'demo/cross'):+.2f}")
_wall_eff = (res['demo/cross'] + res['env/cross'] + res['raw/cross']
             - res['demo/same'] - res['env/same'] - res['raw/same']) / 3
print(f"wall effect      (cross - same): {_wall_eff:+.2f}")
env.close()
