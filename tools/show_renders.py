"""Save dataset and env renders of the same state side by side, full frame and zoomed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

OUT = sys.argv[1] if len(sys.argv) > 1 else "./render_check"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
DATA = "/nas/ada/dev/dino/data/wall_single"

IMG = 224
GRID = 65.0
PIX = (IMG - 1) / (GRID - 1)
CROP = 28
ZOOM = 8
SEP = 8

import train_policy as tp
from datasets.wall_dset import load_wall_slice_train_val
from datasets.img_transforms import default_transform

_, traj = load_wall_slice_train_val(
    transform=default_transform(IMG), data_path=DATA,
    n_rollout=None, normalize_action=True, split_mode="random")
d = traj["valid"]

root = OmegaConf.create({
    "env": {"name": "wall", "act_dim": 2, "goal_dim": 384, "views": 1,
            "args": [], "kwargs": {}},
    "n_envs": 1, "img_size": IMG,
    "serial_env": True, "env_start_method": None,
})
env = tp.make_eval_env(root, 1)
os.makedirs(OUT, exist_ok=True)


def rgb01(img):
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = a.transpose(1, 2, 0)
    if a.max() > 1.5:
        a = a / 255.0
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    return np.clip(a, 0, 1)


def zoom_on(rgb, state):
    cx = int(round(state[0] * PIX))
    cy = int(round(state[1] * PIX))
    x0 = min(max(0, cx - CROP), IMG - 2 * CROP)
    y0 = min(max(0, cy - CROP), IMG - 2 * CROP)
    return np.kron(rgb[y0:y0 + 2 * CROP, x0:x0 + 2 * CROP], np.ones((ZOOM, ZOOM, 1)))


def row(panels):
    """Equal-width panels separated by black bars."""
    h = panels[0].shape[0]
    bar = np.zeros((h, SEP, 3))
    out = [panels[0]]
    for p in panels[1:]:
        out += [bar, p]
    return np.concatenate(out, axis=1)


def save(path, panels):
    Image.fromarray((row(panels) * 255).astype(np.uint8)).save(path)


rng = np.random.RandomState(0)
saved = 0
while saved < N:
    j = int(rng.randint(len(d)))
    t = int(rng.randint(d.get_seq_length(j)))
    img, _, st, info = d.get_frames(j, [t])
    state = np.asarray(st[0])[:2]
    if not (12 < state[0] < 26 or 38 < state[0] < 52) or not (12 < state[1] < 52):
        continue

    env.update_env([info])
    obs, _ = env.prepare([0], state[None])

    ds = rgb01(img["visual"][0])
    ev = rgb01(np.asarray(obs["visual"])[0])
    df = np.abs(ds - ev)
    peak = df.max()
    df = df / max(1e-9, peak)

    name = f"render_{saved:02d}_x{state[0]:.0f}_y{state[1]:.0f}"
    save(os.path.join(OUT, name + "_full.png"), [ds, ev, df])
    save(os.path.join(OUT, name + "_zoom.png"),
         [zoom_on(ds, state), zoom_on(ev, state), zoom_on(df, state)])
    print(f"state ({state[0]:6.2f}, {state[1]:6.2f})  peak diff {peak:.3f}  ->  {name}_*.png")
    saved += 1

print(f"\nwrote {2 * saved} images to {OUT}")
print("each file: dataset | env | difference, left to right, black bars between")
print("look at the _zoom files, the dot is too small to judge at full scale")
env.close()
