"""
Validate the PushT dataset->env action/state mapping by replay."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
import argparse
import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # headless pygame render

import torch
from datasets.pusht_dset import PushTDataset
from env.pusht.pusht_wrapper import PushTWrapper


def angle_err(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 2 * np.pi - d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="dir with states.pth, *_actions.pth, obses/")
    ap.add_argument("--n", type=int, default=5, help="episodes to replay")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with_velocity", action="store_true", default=True)
    ap.add_argument("--no_velocity", dest="with_velocity", action="store_false")
    ap.add_argument("--pos_tol", type=float, default=10.0, help="mean pos error warn threshold (px)")
    args = ap.parse_args()

    dset = PushTDataset(data_path=args.data_path, transform=None,
                        with_velocity=args.with_velocity)
    env = PushTWrapper(with_velocity=args.with_velocity, with_target=True)

    n = min(args.n, len(dset))
    all_pos, all_ang = [], []
    for i in range(n):
        T = dset.get_seq_length(i)
        init_state = dset.states[i, 0].numpy()
        recorded = dset.states[i, :T].numpy()
        # dataset actions are normalized, the env expects the raw (pre-normalize) action
        a_norm = dset.actions[i, :T]
        a_raw = (a_norm * dset.action_std + dset.action_mean).numpy()

        _, states_replay = env.rollout(args.seed, init_state, a_raw)
        m = min(T, states_replay.shape[0])
        pos_err = np.linalg.norm(recorded[:m, :4] - states_replay[:m, :4], axis=-1)
        ang_err = angle_err(recorded[:m, 4], states_replay[:m, 4])
        all_pos.append(pos_err); all_ang.append(ang_err)
        print(f"ep {i:3d}  T={T:3d}  pos_err mean={pos_err.mean():7.3f} max={pos_err.max():7.3f}  "
              f"angle_err mean={ang_err.mean():.4f}")

    pos = np.concatenate(all_pos); ang = np.concatenate(all_ang)
    print("-" * 60)
    print(f"OVERALL  pos_err mean={pos.mean():.3f} max={pos.max():.3f}  "
          f"angle_err mean={ang.mean():.4f} max={ang.max():.4f}")
    status = "PASS" if pos.mean() < args.pos_tol else "CHECK"
    print(f"[{status}] mean pos error {'<' if status == 'PASS' else '>='} {args.pos_tol}px "
          "(low => action/state mapping is correct)")


if __name__ == "__main__":
    main()
