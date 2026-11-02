#!/usr/bin/env python
"""Build a FIXED wall validation set using a closed-form waypoint policy.

WHY THIS EXISTS
---------------
train.py computes `wm/val_loss` on the validation split of whatever dataset groups that
round's fine-tune was handed (train.py:178, `self.dataloaders["valid"]`). The pool grows
and changes composition every DAgger round, so the validation set itself changes every
round. `wm/val_loss` is therefore NOT comparable across rounds, and not comparable across
arms that mix data differently. The round-over-round "world model is degrading" readings
are partly an artefact of a moving yardstick.

`gen_measure/success_rate` does not solve this either: it measures PLANNING success on
fresh tasks, which folds the planner, the policy and the world model into one number.

This script produces a frozen, externally generated set of wall trajectories that is
never trained on. World-model prediction loss measured on it is directly comparable
across rounds and across arms, because the data does not move.

THE POLICY
----------
No training, no world model, no search. Pure geometry, read off env/wall/envs/wall.py:

  * the wall is a vertical bar centred at `wall_x` of width `wall_width` (6), so its
    faces sit at `wall_x` minus/plus `wall_width // 2` -- 29 and 35 for the default
    wall_x=32, which is what experiments/eval_cross_wall.py hardcodes as WALL_LO/WALL_HI
  * the door is a gap in that bar spanning `hole_y` minus/plus `door_space` (4), so y in
    26..34 for the default hole_y=30
  * the transition is `pos_next = pos + action * 2` (DotWall._generate_transition)

So the policy picks a waypoint and steers at it:

  1. inside the wall's x-band  -> drive to the exit face on the goal's side, holding the
     door's y, so it leaves through the gap rather than into the bar
  2. already on the goal's side -> drive straight at the goal
  3. lined up with the door     -> drive at the exit face on the goal's side
  4. otherwise                  -> drive to the approach point (own face, door's y)

Case 3 is what stops case 4 oscillating at the approach point. Step magnitude is the
remaining distance halved (the env doubles the action) clipped into the demonstration
action range [action_lower_bd, action_upper_bd] = [0.2, 1.8].

Tasks come from `env.sample_random_init_goal_states`, which samples the start on one side
of the wall and the goal on the other, so every trajectory is a genuine crossing.

OUTPUT
------
A payload whose keys are exactly `datasets.planned_dset.TRAJ_KEYS`, so
`PlannedTrajDataset` reads it with no new loader code: point any dataset group at it.
Proprioception is stored RAW, matching generate_planned_trajectories.py -- the demo
loader's statistics are applied by PlannedTrajDataset at read time, which is the fix for
the normalisation bug in BUG_HISTORY-adaBot.md section 1. Do not normalise it here.

USAGE
-----
  python tools/gen_wall_waypoint_dset.py --out /path/wall_waypoint_val.pth --n 400
  python tools/gen_wall_waypoint_dset.py --out ... --target-windows 3000
  python tools/gen_wall_waypoint_dset.py --out ... --wall-location 40 --door-location 48
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from env.wall.wall_env_wrapper import WallEnvWrapper, DEFAULT_CFG
from datasets.planned_dset import TRAJ_KEYS


def waypoint_action(pos, goal, wall_x, hole_y, half_width, door_space,
                    clear, step_lo, step_hi, align_frac):
    """One action from the current position. Pure geometry, no model.

    Returns a length-2 float array. The env applies `pos + action * 2`.
    """
    x, y = float(pos[0]), float(pos[1])
    gx, gy = float(goal[0]), float(goal[1])

    face = half_width + clear              # x offset from wall centre to a clear face
    side = -1.0 if x < wall_x else 1.0     # which side of the wall the agent is on
    goal_side = -1.0 if gx < wall_x else 1.0
    exit_pt = (wall_x + goal_side * face, hole_y)   # clear point on the goal's side
    # The band is the SOLID bar only, never the bar plus the clearance. If the clearance
    # were included, the exit point would sit exactly on the band boundary, the policy
    # would still read as in-band on arrival, keep targeting the point it is standing on,
    # and emit a zero action forever. `clear` must stay above this 0.5 margin.
    in_band = abs(x - wall_x) <= half_width + 0.5

    if in_band:
        # Only the door is passable here, so leave along the door's y, never the goal's y.
        # Without this the agent turns toward the goal while still between the faces and
        # drives into the bar from inside the gap.
        target = exit_pt
    elif side == goal_side:
        target = (gx, gy)
    elif abs(y - hole_y) <= door_space * align_frac:
        # Lined up with the gap: commit and drive through.
        target = exit_pt
    else:
        # Not lined up: go to the approach point on this side, at the door's height.
        target = (wall_x + side * face, hole_y)

    d = np.array([target[0] - x, target[1] - y], dtype=float)
    n = float(np.linalg.norm(d))
    if n < 1e-8:
        return np.zeros(2, dtype=float)
    # the env doubles the action, so halve the distance, then clip to the demo range
    return d / n * float(np.clip(n / 2.0, step_lo, step_hi))


def rollout_one(env, seed, wall_x, hole_y, args):
    """One episode. Returns a dict of per-trajectory tensors, or None if too short."""
    init_state, goal_state = env.sample_random_init_goal_states(seed)
    init_state = np.asarray(init_state, dtype=float)
    goal_state = np.asarray(goal_state, dtype=float)

    obs, state = env.prepare(seed, init_state)
    vis = [np.asarray(obs["visual"])]
    props = [np.asarray(obs["proprio"], dtype=np.float32)]
    states = [np.asarray(state, dtype=np.float32)]
    acts = []
    reached = False

    half_width = env.wall_config.wall_width // 2
    for _ in range(args.max_steps):
        pos = states[-1]
        a = waypoint_action(pos, goal_state, wall_x, hole_y, half_width,
                            env.wall_config.door_space, args.clear,
                            args.step_lo, args.step_hi, args.align_frac)
        o, _r, _d, info = env.step(torch.as_tensor(a).float())
        # DotWall.step does NOT resize; the wrapper's step_multiple does. Mirror it so the
        # stored frames match what every other loader in the repo sees.
        v = env.transform(o["visual"]).permute(1, 2, 0)
        vis.append(np.asarray(v))
        props.append(np.asarray(o["proprio"], dtype=np.float32))
        states.append(np.asarray(info["state"], dtype=np.float32))
        acts.append(a)
        if float(np.linalg.norm(states[-1][:2] - goal_state[:2])) < args.success_radius:
            reached = True
            break

    T = len(vis)
    if T < args.min_steps:
        return None
    act_arr = np.asarray(acts, dtype=np.float32) if acts else np.zeros((0, 2), np.float32)
    # T frames cover T-1 executed actions; pad by repeating the last so actions align with
    # frames, exactly as _pad_actions_to does in generate_planned_trajectories.py
    if act_arr.shape[0] < T:
        pad = (act_arr[-1:] if act_arr.shape[0] else np.zeros((1, 2), np.float32))
        act_arr = np.concatenate([act_arr, np.repeat(pad, T - act_arr.shape[0], 0)], 0)

    v = torch.as_tensor(np.stack(vis))
    if v.dtype != torch.uint8:
        v = v.round().clamp(0, 255).to(torch.uint8)
    return {
        "visual": v,                                              # (T, H, W, C) uint8
        "actions": torch.as_tensor(act_arr).float(),              # (T, 2)
        "states": torch.as_tensor(np.stack(states)).float(),      # (T, 2)
        "proprios": torch.as_tensor(np.stack(props)).float(),     # (T, 2) RAW
        "seq_lengths": int(T),
        "init_states": torch.as_tensor(init_state).float(),
        "goal_states": torch.as_tensor(goal_state).float(),
        "successes": bool(reached),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output .pth path")
    p.add_argument("--n", type=int, default=400, help="trajectories to attempt")
    p.add_argument("--target-windows", type=int, default=0,
                   help="stop once this many world-model windows have accumulated (0 = off)")
    p.add_argument("--seed", type=int, default=100000,
                   help="base seed; keep it DISJOINT from any training seed")
    p.add_argument("--wall-location", type=int, default=32)
    p.add_argument("--door-location", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--min-steps", type=int, default=17,
                   help="drop shorter trajectories; must exceed the window span")
    p.add_argument("--clear", type=float, default=2.0,
                   help="clearance kept from a wall face when aiming at it")
    p.add_argument("--step-lo", type=float, default=float(DEFAULT_CFG.action_lower_bd))
    # action_step_mean (1.0), NOT action_upper_bd (1.8). Two reasons. The demonstrations the
    # world model was trained on average a step of action_step_mean, so a policy pinned at
    # the upper bound would feed it an action distribution it never saw. And at 1.8 the env
    # moves 3.6 px per step, so a crossing finishes in a mean of 12.8 frames -- below the
    # 16-frame window span, yielding 0.5 windows per trajectory. At 1.0 the mean is 22.2
    # frames and 7.6 windows, which is demo parity (wall demos give 7).
    p.add_argument("--step-hi", type=float, default=float(DEFAULT_CFG.action_step_mean))
    p.add_argument("--align-frac", type=float, default=0.5,
                   help="fraction of door_space counted as 'lined up with the gap'")
    p.add_argument("--success-radius", type=float, default=4.5,
                   help="matches WallEnvWrapper.eval_state")
    p.add_argument("--shard-every", type=int, default=0,
                   help="write a shard directory every N trajectories instead of one file "
                        "(0 = single file). At demo parity a 3000-window set is ~411 "
                        "trajectories of 224x224x3 uint8 frames = ~1.4 GB, which torch.load "
                        "pulls into RAM in one piece; 100 keeps each shard near 330 MB. "
                        "PlannedTrajDataset._load_payload reads a shard directory directly.")
    p.add_argument("--num-hist", type=int, default=3)
    p.add_argument("--num-pred", type=int, default=1)
    p.add_argument("--frameskip", type=int, default=5)
    args = p.parse_args()

    # A clearance at or below the in_band margin puts the exit point inside the band and the
    # policy deadlocks on arrival, emitting zero actions until max_steps. Measured: clear=0.5
    # drops the success rate to 0.19, clear>=1.0 holds at 1.00.
    if args.clear < 1.0:
        raise SystemExit(f"--clear must be at least 1.0 (got {args.clear}); below that the "
                         f"exit point falls inside the wall band and the policy deadlocks.")

    # window span in env frames, for the window accounting only
    span = (args.num_hist + args.num_pred - 1) * args.frameskip + 1

    env = WallEnvWrapper(device="cpu")
    env.update_env({"fix_wall_location": torch.tensor(args.wall_location),
                    "fix_door_location": torch.tensor(args.door_location)})
    wall_x = float(env.wall_x)
    hole_y = float(env.hole_y)
    half_width = env.wall_config.wall_width // 2
    print(f"[waypoint] wall_x={wall_x} hole_y={hole_y} "
          f"faces at {wall_x - half_width} / {wall_x + half_width}, "
          f"door y in {hole_y - env.wall_config.door_space}.."
          f"{hole_y + env.wall_config.door_space}")

    out = os.path.abspath(args.out)
    sharded = args.shard_every > 0
    shard_dir = out[:-4] if out.endswith(".pth") else out
    if sharded:
        os.makedirs(shard_dir, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # These three exist in TRAJ_KEYS but carry no meaning for hand-generated data, so they are
    # None in EVERY shard. _load_payload seeds each key from the first shard and only extends
    # when both sides are non-None, so a key that is None everywhere stays None after the merge
    # -- and `select=wm_fail` then fails loudly rather than thresholding on fabricated error.
    none_keys = ("wm_pred_final_states", "wm_latent_err", "dreamed_latents")

    def payload_of(d):
        d = {k: (None if k in none_keys else v) for k, v in d.items()}
        return {**d, "dreamed": False, "env_name": "wall", "frameskip": args.frameskip}

    acc = {k: [] for k in TRAJ_KEYS}
    lens_all, n_succ, n_short, n_windows, shard_idx = [], 0, 0, 0, 0

    def flush():
        """Write the buffered trajectories as one shard and free the RAM they hold."""
        nonlocal shard_idx
        if not acc["seq_lengths"]:
            return
        path = os.path.join(shard_dir, f"shard_{shard_idx:04d}.pth")
        torch.save(payload_of(acc), path)
        print(f"[waypoint] shard {shard_idx}: {len(acc['seq_lengths'])} trajectories -> {path}")
        shard_idx += 1
        for k in TRAJ_KEYS:
            acc[k].clear()

    for i in range(args.n):
        rec = rollout_one(env, args.seed + i, wall_x, hole_y, args)
        if rec is None:
            n_short += 1
            continue
        for k, v in rec.items():
            acc[k].append(v)
        lens_all.append(int(rec["seq_lengths"]))
        n_succ += int(rec["successes"])
        n_windows += max(0, rec["seq_lengths"] - span + 1)
        if (i + 1) % 50 == 0:
            print(f"[waypoint] {len(lens_all)} kept / {i + 1} attempted, "
                  f"{n_succ} reached goal, ~{n_windows} windows")
        if sharded and len(acc["seq_lengths"]) >= args.shard_every:
            flush()
        if args.target_windows and n_windows >= args.target_windows:
            print(f"[waypoint] reached --target-windows={args.target_windows}, stopping")
            break

    kept = len(lens_all)
    if kept == 0:
        raise SystemExit("no trajectories survived --min-steps; lower it or raise --max-steps")

    if sharded:
        flush()
        target, meta_path = shard_dir, os.path.join(shard_dir, "manifest.json")
    else:
        torch.save(payload_of(acc), out)
        target, meta_path = out, shard_dir + "_meta.json"

    meta = {
        "trajectories": kept, "attempted": args.n, "dropped_too_short": n_short,
        "reached_goal": n_succ, "success_rate": round(n_succ / kept, 4),
        "windows": n_windows, "window_span_frames": span,
        "len_min": min(lens_all), "len_mean": round(float(np.mean(lens_all)), 2),
        "len_max": max(lens_all), "shards": shard_idx if sharded else 0,
        "wall_location": args.wall_location, "door_location": args.door_location,
        "step_lo": args.step_lo, "step_hi": args.step_hi, "clear": args.clear,
        "base_seed": args.seed, "policy": "closed-form waypoint, no training",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[waypoint] saved {kept} trajectories -> {target}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
