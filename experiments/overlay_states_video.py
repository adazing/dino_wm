"""Play the stored demo frames with each states array drawn on top, side by side."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "/nas/ada/dev/dino/data/wall_single")
NTRAJ = int(sys.argv[2] if len(sys.argv) > 2 else 4)
OUT = Path(sys.argv[3] if len(sys.argv) > 3 else "./state_overlay")
GRID = 65.0
FPS = 6

OUT.mkdir(parents=True, exist_ok=True)
A = np.asarray(torch.load(DATA / "states.pth")).squeeze()[..., :2]
B = np.asarray(torch.load(DATA / "states_true.pth")).squeeze()[..., :2]
print(f"states.pth       {A.shape}")
print(f"states_true.pth  {B.shape}")


def frames_of(idx):
    img = torch.load(DATA / "obses" / f"episode_{idx:03d}.pth")
    a = np.asarray(img)
    if a.ndim == 4 and a.shape[1] in (1, 3):   # , T, C, H, W
        a = a.transpose(0, 2, 3, 1)
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8) if a.max() <= 1.5 else a.astype(np.uint8)
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    return a


def mark(frame, xy, colour):
    """Crosshair at a world coordinate, plus a hollow box so it stays visible over the dot."""
    h, w = frame.shape[:2]
    px = (w - 1) / (GRID - 1)
    cx, cy = int(round(xy[0] * px)), int(round(xy[1] * px))
    r = max(4, w // 40)
    for d in range(-r, r + 1):
        if 0 <= cy < h and 0 <= cx + d < w:
            frame[cy, cx + d] = colour
        if 0 <= cy + d < h and 0 <= cx < w:
            frame[cy + d, cx] = colour
    for d in range(-r, r + 1):
        for yy, xx in ((cy - r, cx + d), (cy + r, cx + d), (cy + d, cx - r), (cy + d, cx + r)):
            if 0 <= yy < h and 0 <= xx < w:
                frame[yy, xx] = colour
    return frame


def border(frame, colour, t=3):
    frame[:t] = colour
    frame[-t:] = colour
    frame[:, :t] = colour
    frame[:, -t:] = colour
    return frame


import imageio

RED, GREEN = np.array([220, 40, 40], np.uint8), np.array([40, 200, 80], np.uint8)
rng = np.random.RandomState(0)
picks = rng.choice(len(A), size=min(NTRAJ, len(A)), replace=False)

for j in picks:
    vis = frames_of(int(j))
    T = min(len(vis), len(A[j]), len(B[j]))
    gap = np.full((vis.shape[1], 8, 3), 255, np.uint8)
    out = []
    for t in range(T):
        left = border(mark(vis[t].copy(), A[j][t], RED), RED)
        right = border(mark(vis[t].copy(), B[j][t], GREEN), GREEN)
        out.append(np.concatenate([left, gap, right], axis=1))
    path = OUT / f"traj{int(j):03d}.mp4"
    imageio.mimsave(str(path), out, fps=FPS)
    # a still at the start, middle and end for a quick look without playing anything
    for t in (0, T // 2, T - 1):
        imageio.imwrite(str(OUT / f"traj{int(j):03d}_f{t:02d}.png"), out[t])
    d = np.linalg.norm(A[j][:T] - B[j][:T], axis=-1)
    print(f"  traj {int(j):3d}  T={T:3d}  gap at frame 0 {d[0]:5.2f}  mid {d[T//2]:5.2f}  end {d[-1]:5.2f}"
          f"   -> {path}")

print(f"\nLEFT  red    = states.pth")
print(f"RIGHT green  = states_true.pth")
print("Whichever marker stays on the dot is the array these images depict.")
