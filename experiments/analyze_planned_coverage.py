"""How well does a generated (planned) dataset cover the (init, goal) pairs it was sampled from?"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
import csv
import json
import argparse
import numpy as np


def _load_planned(path, only_successes=False, max_trajectories=None):
    """Return (init_states (n,ds), goal_states (n,ds), successes (n,) bool or None, visual list) from
    a planned dataset file/dir. only_successes/max_trajectories mirror BC's filtering (success filter
    first, then first-N cap) so coverage can reflect exactly what BC trained on. visual[k] is the
    trajectory's (T,H,W,C) uint8 frames (aligned with init[k]), for saving fail-only videos."""
    from datasets.planned_dset import PlannedTrajDataset
    import torch
    pd = PlannedTrajDataset(path, transform=None, only_successes=only_successes,
                            max_trajectories=max_trajectories)
    if pd.init_states is None or pd.goal_states is None:
        raise ValueError(
            f"{path} has no init_states/goal_states metadata; regenerate with the current "
            "generate_planned_trajectories.py (it stores them per trajectory).")
    init = torch.stack([torch.as_tensor(np.asarray(s)).float().flatten() for s in pd.init_states]).numpy()
    goal = torch.stack([torch.as_tensor(np.asarray(s)).float().flatten() for s in pd.goal_states]).numpy()
    succ = None if pd.successes is None else np.asarray([bool(x) for x in pd.successes])
    return init, goal, succ, pd.visual


def _eval_pairs_distinct(pool, eval_seed, n_eval_episodes, eval_goal_off):
    """The DISTINCT (gtraj, init_off) pairs the BC eval samples (train_policy.py:1034), in first-seen
    order. Used as the coverage universe when --eval_only is set."""
    import random as _random
    valid_ids = [gt for gt, (sp, li, s) in enumerate(pool) if s.shape[0] >= eval_goal_off + 1]
    if not valid_ids:
        raise ValueError(f"no trajectory is long enough (>= {eval_goal_off + 1}) for eval pairs.")
    seen, pairs = set(), []
    for e in range(max(1, int(n_eval_episodes))):
        rng = _random.Random(int(eval_seed) + e)
        j = rng.choice(valid_ids)
        io = rng.randint(0, max(0, pool[j][2].shape[0] - eval_goal_off - 1))
        if (j, io) not in seen:
            seen.add((j, io)); pairs.append((j, io))
    return pairs


def _save_fail_videos(fail_k, visual, traj_pair, traj_label, pool, traj_dset, offset,
                      out_dir, fps, cap):
    """Write an mp4 per planned trajectory in fail_k, each frame laid out as
    [ init demo frame | goal demo frame | rollout frame_t ] so the difficulty (how far the goal is
    from the start) is visible. init/goal images come from the ORIGINAL demo (the goal is the frame
    `offset` ahead of the init), resized to the rollout resolution."""
    import imageio
    import torch
    os.makedirs(out_dir, exist_ok=True)
    if cap is not None and len(fail_k) > int(cap):
        print(f"[coverage] {len(fail_k)} fail-only trajectories > --max_fail_videos={cap}; "
              f"saving the first {cap}.")
        fail_k = fail_k[: int(cap)]

    def _demo_img(gt, frame_idx):   # -> (H,W,C) uint8 from the original demo
        split, local_idx, s = pool[gt]
        dset = traj_dset["train"] if split == "train" else traj_dset["valid"]
        fi = int(min(frame_idx, s.shape[0] - 1))
        v = dset.get_frames(local_idx, [fi])[0]["visual"][0]   # (C,H,W) float [0,1]
        return (v.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)

    def _resize(a, H, W):   # (h,w,C) uint8 -> (H,W,C) uint8, bilinear
        t = torch.as_tensor(a).permute(2, 0, 1).unsqueeze(0).float()
        t = torch.nn.functional.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
        return t.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().numpy()

    written = 0
    for k in fail_k:
        if visual is None or k >= len(visual) or visual[k] is None or traj_pair[k] is None:
            continue
        roll = np.asarray(visual[k])
        if roll.dtype != np.uint8:
            roll = np.clip(roll, 0, 255).astype(np.uint8)
        H, W = roll.shape[1], roll.shape[2]
        gt, io = traj_pair[k]
        init_p = _resize(_demo_img(gt, io), H, W)   # static init panel
        goal_p = _resize(_demo_img(gt, io + offset), H, W)   # static goal panel (offset ahead)
        lbl = str(traj_label.get(gt, gt)).replace(":", "-")
        path = os.path.join(out_dir, f"failonly_{lbl}_off{io}_k{k}.mp4")
        w = imageio.get_writer(path, fps=int(fps))
        for t in range(roll.shape[0]):
            w.append_data(np.concatenate([init_p, goal_p, roll[t]], axis=1))   # [init | goal | rollout]
        w.close()
        written += 1
    print(f"[coverage] wrote {written} fail-only videos ([init | goal | rollout]) to {out_dir}")


def _load_original_pool(args):
    """Load the original dataset and return the trajectory pool generation sampled from
    (source_split), as a list of per-trajectory state arrays plus a label for each.
    'both' concatenates train THEN valid, matching _ConcatTrajDataset's indexing in generation."""
    from datasets.pusht_dset import load_pusht_slice_train_val
    _, traj_dset = load_pusht_slice_train_val(
        transform=None, n_rollout=None, data_path=args.data_path,
        normalize_action=True, split_ratio=args.split_ratio,
        num_hist=1, num_pred=1, frameskip=args.frameskip,
        with_velocity=True, relative=args.relative, action_scale=args.action_scale)

    def _states(dset, split_name):
        out = []
        for i in range(len(dset)):
            T = int(dset.get_seq_length(i))
            s = dset.states[i, :T].float().numpy()   # (T, ds)
            out.append((split_name, i, s))
        return out

    if args.source_split == "train":
        pool = _states(traj_dset["train"], "train")
    elif args.source_split in ("val", "valid"):
        pool = _states(traj_dset["valid"], "valid")
    elif args.source_split == "both":
        pool = _states(traj_dset["train"], "train") + _states(traj_dset["valid"], "valid")
    else:
        raise ValueError(f"--source_split must be train|val|both, got {args.source_split}")
    return pool, traj_dset


def _match_dims(ds, override):
    if override is not None:
        return int(override)
    # pusht 7-dim state = [agent_xy, block_xy, angle, agent_vxy], match on position+angle only,
    # velocity is finite-diff/noisy and does not identify a frame.
    return 5 if ds == 7 else ds


def _hist(counts, buckets=(0, 1, 2, 3)):
    h = {}
    for b in buckets[:-1]:
        h[str(b)] = int((counts == b).sum())
    h[f">={buckets[-1]}"] = int((counts >= buckets[-1]).sum())
    return h


def _eval_pair_coverage(pool, sampleable, eval_seed, n_eval_episodes, eval_goal_off):
    """Reproduce the exact (init, goal) episodes the eval samples (train_policy.py:1034:
    rng=Random(eval_seed+episode_idx), traj=rng.choice(valid_ids), init_off=rng.randint(0, hi)),
    and classify each by its planned training support: lands on a pair the planner SUCCEEDED on,
    a pair that was only attempted-and-failed, or a never-sampled pair. Assumes the eval draws from
    the same split as `pool` (source_split) and eval_goal_off == the planned offset, so pair identity
    (gtraj, init_off) matches the sampleable keys exactly."""
    import random as _random
    valid_ids = [gt for gt, (sp, li, s) in enumerate(pool) if s.shape[0] >= eval_goal_off + 1]
    if not valid_ids:
        return None
    total_eps = max(1, int(n_eval_episodes))
    episodes = {"success": 0, "fail_only": 0, "never": 0}
    uniq = {}
    for e in range(total_eps):
        rng = _random.Random(int(eval_seed) + e)   # per-episode, model-independent (as in eval)
        j = rng.choice(valid_ids)
        hi = max(0, pool[j][2].shape[0] - eval_goal_off - 1)
        io = rng.randint(0, hi)
        ns, nsu, nf = sampleable.get((j, io), [0, 0, 0])
        st = "success" if nsu >= 1 else ("fail_only" if ns >= 1 else "never")
        episodes[st] += 1
        uniq[(j, io)] = st
    unique = {"success": 0, "fail_only": 0, "never": 0}
    for st in uniq.values():
        unique[st] += 1
    return {
        "eval_seed": int(eval_seed), "n_eval_episodes": int(n_eval_episodes),
        "eval_goal_offset": int(eval_goal_off), "total_episodes": int(total_eps),
        "unique_pairs": len(uniq),
        "episodes": episodes, "unique": unique,
        "episode_frac_with_success_support": round(episodes["success"] / max(1, total_eps), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planned", required=True, help="planned dataset file or shard dir")
    ap.add_argument("--data_path", required=True, help="original demo dataset dir")
    ap.add_argument("--source_split", default="train", choices=["train", "val", "valid", "both"],
                    help="split generation sampled (init, goal) from (source_split in generation)")
    ap.add_argument("--split_ratio", type=float, default=0.9)
    ap.add_argument("--frameskip", type=int, default=5, help="WM frameskip used in planning")
    ap.add_argument("--goal_H", type=int, default=5, help="goal horizon; offset = frameskip*goal_H")
    ap.add_argument("--relative", action="store_true", default=True)
    ap.add_argument("--action_scale", type=float, default=100.0)
    ap.add_argument("--match_dims", type=int, default=None,
                    help="state dims to match on (default: 5 for pusht-7, else all)")
    ap.add_argument("--match_tol", type=float, default=1.0,
                    help="max nearest-neighbor state distance to accept a match (raw units)")
    ap.add_argument("--src_cap", type=int, default=None,
                    help="source_max_trajectories/original_max_trajectories cap, if generation used one")
    # cap the planned data to what BC actually trained on, BC applies only_successes THEN first-N.
    ap.add_argument("--max_planned", type=int, default=None,
                    help="use only the first N planned trajectories (matches BC planned_max_trajectories), "
                         "so coverage reflects what BC actually trained on. Applied after the optional "
                         "success filter, in file order, exactly like BC.")
    ap.add_argument("--planned_only_successes", action="store_true",
                    help="drop failed planned trajectories before counting/capping (matches BC "
                         "planned_only_successes=true). Removes the fail breakdown.")
    # restrict coverage to the exact (init, goal) episodes the eval samples
    # (train_policy.py:1034), rng=Random(eval_seed + episode_index), traj=rng.choice(valid_ids),
    # init_off=rng.randint(...).
    ap.add_argument("--eval_seed", type=int, default=None,
                    help="if set, ALSO report coverage over only the seeded eval episodes (BC cfg.seed). "
                         "Assumes the eval samples from the same split as --source_split.")
    ap.add_argument("--n_eval_episodes", type=int, default=100,
                    help="how many seeded eval episodes to check (= BC n_env_evals). The seeds are "
                         "eval_seed+0..N-1, exactly the episodes the eval runs.")
    ap.add_argument("--eval_goal_offset", type=int, default=None,
                    help="eval goal_offset in frames (default: planned offset frameskip*goal_H). Must "
                         "equal the planned offset for per-pair training support to be exact.")
    ap.add_argument("--eval_only", action="store_true",
                    help="restrict the ENTIRE analysis (universe, histograms, partition, per-traj) to "
                         "the distinct eval pairs, instead of the full validation space. Needs --eval_seed. "
                         "Use for eval_pairs generation so every %% is relative to the eval tasks.")
    # save videos of trajectories at FAIL-only pairs, pairs the planner never solved.
    ap.add_argument("--fail_video_dir", default=None,
                    help="if set, write an mp4 of every planned trajectory at a fail-only pair (a pair "
                         "with 0 successes) to this folder, so you can watch the hard tasks fail.")
    ap.add_argument("--fail_video_fps", type=int, default=12)
    ap.add_argument("--max_fail_videos", type=int, default=100,
                    help="cap how many fail-only videos to write (null-safe first-N).")
    ap.add_argument("--out", default="coverage.json", help="summary json path (csv written alongside)")
    args = ap.parse_args()

    offset = args.frameskip * args.goal_H
    print(f"[coverage] goal offset = frameskip*goal_H = {args.frameskip}*{args.goal_H} = {offset} frames")

    init, goal, succ, visual = _load_planned(args.planned, only_successes=args.planned_only_successes,
                                             max_trajectories=args.max_planned)
    n_gen = len(init)
    n_succ_total = None if succ is None else int(succ.sum())
    n_fail_total = None if succ is None else int((~succ).sum())
    print(f"[coverage] planned trajectories: {n_gen}"
          + ("" if succ is None else f"  ({n_succ_total} successes, {n_fail_total} failures)"))
    if succ is None:
        print("[coverage] NOTE: no 'successes' metadata -> reporting attempts only (no success/fail split).")

    pool, traj_dset = _load_original_pool(args)   # list of (split, local_idx, states(T,ds))
    ds = pool[0][2].shape[1]
    mdims = _match_dims(ds, args.match_dims)
    print(f"[coverage] source_split={args.source_split}: {len(pool)} trajectories, state_dim={ds}, "
          f"matching on first {mdims} dims")

    # space of the source split (every (gtraj, init_off) with a goal frame in range).
    eval_goal_off = args.eval_goal_offset if args.eval_goal_offset is not None else offset
    restrict = bool(args.eval_only)
    if restrict and args.eval_seed is None:
        raise ValueError("--eval_only requires --eval_seed (which eval episodes define the pair set).")
    sampleable = {}   # (gtraj, init_off) -> [n_sampled, n_succ, n_fail]
    per_traj_sampleable = {}   # gtraj -> count of sampleable offsets
    traj_label = {}
    if restrict:
        if eval_goal_off != offset:
            print(f"[coverage] WARNING: eval_goal_offset={eval_goal_off} != planned offset={offset}; "
                  "the eval-pair identities may not line up with the planned goals.")
        for gt, io in _eval_pairs_distinct(pool, args.eval_seed, args.n_eval_episodes, eval_goal_off):
            traj_label[gt] = f"{pool[gt][0]}:{pool[gt][1]}"
            sampleable[(gt, io)] = [0, 0, 0]
            per_traj_sampleable[gt] = per_traj_sampleable.get(gt, 0) + 1
        print(f"[coverage] --eval_only: universe = {len(sampleable)} distinct eval pairs "
              f"(eval_seed={args.eval_seed}, {args.n_eval_episodes} episodes)")
    else:
        for gt, (split, li, s) in enumerate(pool):
            if args.src_cap is not None and gt >= int(args.src_cap):
                continue
            traj_label[gt] = f"{split}:{li}"
            T = s.shape[0]
            n_off = T - offset
            if n_off <= 0:
                per_traj_sampleable[gt] = 0
                continue
            per_traj_sampleable[gt] = n_off
            for io in range(n_off):
                sampleable[(gt, io)] = [0, 0, 0]
    n_sampleable = len(sampleable)
    _universe = "eval pairs" if restrict else "source pool"
    print(f"[coverage] sampleable (init, goal) pairs in {_universe}: {n_sampleable} "
          f"across {sum(1 for v in per_traj_sampleable.values() if v > 0)} usable trajectories")

    # nearest-neighbor lookup, every (gtraj, frame) -> its state, so a planned init/goal STATE
    # maps back to the original (trajectory, frame) it was copied from.
    frame_owner, frame_vecs = [], []
    for gt, (split, li, s) in enumerate(pool):
        if args.src_cap is not None and gt >= int(args.src_cap):
            continue
        for fr in range(s.shape[0]):
            frame_owner.append((gt, fr))
            frame_vecs.append(s[fr, :mdims])
    frame_vecs = np.asarray(frame_vecs, dtype=np.float64)   # (M, mdims)

    def nearest(vec):
        d = ((frame_vecs - vec[:mdims]) ** 2).sum(1)
        j = int(np.argmin(d))
        return frame_owner[j], float(np.sqrt(d[j]))

    # attribute each planned trajectory to a sampleable (gtraj, init_off) via its init state,
    # split the tally into success vs fail.
    matched = 0
    unmatched = []   # (idx, init_dist)
    off_mismatch = 0
    max_init_dist = 0.0
    traj_pair = [None] * n_gen   # planned traj k -> its (gtraj, init_off) pair
    skipped_non_universe = 0
    for k in range(n_gen):
        (gt_i, fr_i), di = nearest(init[k])
        (gt_g, fr_g), dg = nearest(goal[k])
        max_init_dist = max(max_init_dist, di)
        if di > args.match_tol:
            unmatched.append((k, di))
            continue
        if gt_g != gt_i or (fr_g - fr_i) != offset:
            off_mismatch += 1   # goal not exactly offset ahead in same traj
        key = (gt_i, fr_i)
        traj_pair[k] = key
        if key not in sampleable:
            if restrict:   # planned traj not at an eval pair, ignore it
                skipped_non_universe += 1
                continue
            sampleable[key] = [0, 0, 0]   # init frame outside the enumerated range
        sampleable[key][0] += 1   # sampled (attempted)
        if succ is None or bool(succ[k]):
            sampleable[key][1] += 1   # success, or unknown -> counted as sampled only
        if succ is not None and not bool(succ[k]):
            sampleable[key][2] += 1   # fail
        matched += 1
    if restrict and skipped_non_universe:
        print(f"[coverage] --eval_only: ignored {skipped_non_universe} planned trajectories not at an "
              "eval pair (generated for other pairs).")

    print(f"[coverage] matched {matched}/{n_gen} planned inits to source frames "
          f"(max init match dist {max_init_dist:.3g}); unmatched (dist>{args.match_tol}): "
          f"{len(unmatched)}; goal-offset mismatches: {off_mismatch}")
    if len(unmatched) > 0.1 * max(1, n_gen):
        print("[coverage] WARNING: many unmatched inits -- source_split/split_ratio/frameskip/goal_H "
              "likely do not match the generation run, or --match_dims/--match_tol need tuning.")

    # per-pair count arrays over the FULL sampleable space (0s included)
    samp = np.array([v[0] for v in sampleable.values()])
    suc = np.array([v[1] for v in sampleable.values()])
    fai = np.array([v[2] for v in sampleable.values()])
    has_succ = succ is not None

    # pair categories and their connections, a partition of all sampleable pairs
    #   never_sampled  n_sampled == 0
    #   sampled  n_sampled >= 1  = succ_only + fail_only + mixed
    #   succ_only  >=1 success, 0 fail
    #   fail_only  >=1 sampled, 0 success  <- lost by only_successes (H1 bias)
    #   mixed  >=1 success and >=1 fail
    n_never = int((samp == 0).sum())
    n_sampled = int((samp >= 1).sum())
    n_succ_any = int((suc >= 1).sum()) if has_succ else n_sampled
    n_fail_any = int((fai >= 1).sum()) if has_succ else 0
    n_fail_only = int(((samp >= 1) & (suc == 0)).sum()) if has_succ else 0
    n_succ_only = int(((suc >= 1) & (fai == 0)).sum()) if has_succ else n_sampled
    n_mixed = int(((suc >= 1) & (fai >= 1)).sum()) if has_succ else 0

    # per-trajectory rollup
    per_traj = {}
    for (gt, io), (ns, nsu, nf) in sampleable.items():
        a = per_traj.setdefault(gt, [0, 0, 0])
        a[0] += ns; a[1] += nsu; a[2] += nf
    n_usable_traj = sum(1 for v in per_traj_sampleable.values() if v > 0)
    trajs_sampled = sum(1 for v in per_traj.values() if v[0] > 0)
    trajs_succ = sum(1 for v in per_traj.values() if v[1] > 0) if has_succ else trajs_sampled

    # save videos of every planned trajectory at a FAIL-only pair (a pair with >=1 attempt, 0
    # success)
    if args.fail_video_dir:
        if not has_succ:
            print("[coverage] --fail_video_dir set but the planned file has no success metadata; "
                  "cannot identify fail-only trajectories.")
        else:
            fail_only_pairs = {key for key, v in sampleable.items() if v[0] >= 1 and v[1] == 0}
            fail_k = [k for k in range(n_gen) if traj_pair[k] in fail_only_pairs]
            print(f"[coverage] fail-only pairs: {len(fail_only_pairs)}; planned trajectories at them: "
                  f"{len(fail_k)} -> saving videos to {args.fail_video_dir}")
            _save_fail_videos(fail_k, visual, traj_pair, traj_label, pool, traj_dset, offset,
                              args.fail_video_dir, args.fail_video_fps, args.max_fail_videos)

    # optional, coverage over only the exact seeded episodes the eval will run
    eval_cov = None
    if args.eval_seed is not None:
        eval_goal_off = args.eval_goal_offset if args.eval_goal_offset is not None else offset
        if eval_goal_off != offset:
            print(f"[coverage] WARNING: eval_goal_offset={eval_goal_off} != planned offset={offset}; "
                  "the eval goal is a different frame than the planned goal, so per-pair training "
                  "support below is not exact.")
        eval_cov = _eval_pair_coverage(pool, sampleable, args.eval_seed, args.n_eval_episodes,
                                       eval_goal_off)

    def frac(x):
        return round(x / max(1, n_sampleable), 4)

    summary = {
        "planned": os.path.abspath(args.planned),
        "data_path": os.path.abspath(args.data_path),
        "source_split": args.source_split,
        "eval_only": restrict,   # true = universe restricted to the distinct eval pairs
        "goal_offset_frames": offset,
        "has_success_metadata": has_succ,
        "n_generated": int(n_gen),
        "n_generated_success": n_succ_total,
        "n_generated_fail": n_fail_total,
        "n_matched": int(matched),
        "n_unmatched": int(len(unmatched)),
        "goal_offset_mismatches": int(off_mismatch),
        "max_init_match_dist": float(max_init_dist),
        "n_sampleable_pairs": int(n_sampleable),
        "n_usable_source_trajectories": int(n_usable_traj),
        # pair categories, counts + fraction of all sampleable pairs
        "pairs": {
            "never_sampled": {"n": n_never, "frac": frac(n_never)},
            "sampled": {"n": n_sampled, "frac": frac(n_sampled)},
            "sampled_and_succeeded": {"n": n_succ_any, "frac": frac(n_succ_any)},
            "sampled_and_failed": {"n": n_fail_any, "frac": frac(n_fail_any)},
            "success_only": {"n": n_succ_only, "frac": frac(n_succ_only)},
            "fail_only": {"n": n_fail_only, "frac": frac(n_fail_only)},
            "mixed_success_and_fail": {"n": n_mixed, "frac": frac(n_mixed)},
        },
        "hist_sampled_per_pair": _hist(samp),
        "hist_success_per_pair": _hist(suc) if has_succ else None,
        "hist_fail_per_pair": _hist(fai) if has_succ else None,
        "max_sampled_for_one_pair": int(samp.max()) if len(samp) else 0,
        "trajectories_sampled": int(trajs_sampled),
        "trajectories_with_a_success": int(trajs_succ),
        "eval_pair_coverage": eval_cov,
    }

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    csv_path = os.path.splitext(args.out)[0] + "_pairs.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_traj", "init_offset", "goal_offset", "n_sampled", "n_success", "n_fail"])
        for (gt, io), (ns, nsu, nf) in sorted(sampleable.items()):
            w.writerow([traj_label.get(gt, gt), io, io + offset, ns, nsu, nf])

    def line(label, n):
        return f"  {label:<28} {n:6d}  ({100*n/max(1,n_sampleable):5.1f}%)"

    _uni_label = "eval pairs only" if restrict else f"source_split={args.source_split}"
    _pairs_label = "eval (init, goal) pairs" if restrict else "sampleable (init, goal) pairs"
    print(f"\n=== COVERAGE SUMMARY ({_uni_label}, offset={offset}) ===")
    print(f"planned trajectories:          {n_gen}"
          + ("" if not has_succ else f"  ({n_succ_total} success, {n_fail_total} fail)"))
    print(f"{_pairs_label}: {n_sampleable}  across {n_usable_traj} usable trajectories\n")
    print(f"of all {'eval' if restrict else 'sampleable'} pairs:")
    print(line("never sampled:", n_never))
    print(line("sampled (>=1 attempt):", n_sampled))
    if has_succ:
        print(line("  -> succeeded (>=1 succ):", n_succ_any) + "   <- only_successes BC trains on")
        print(line("  -> only failed (0 succ):", n_fail_only) + "   <- lost by only_successes (H1 bias)")
        print("\nconnections within the sampled pairs (partition):")
        print(line("success-only (no fails):", n_succ_only))
        print(line("fail-only (no succ):", n_fail_only))
        print(line("mixed (both succ & fail):", n_mixed))
    print(f"\nsampled-per-pair histogram:    {summary['hist_sampled_per_pair']}")
    if has_succ:
        print(f"success-per-pair histogram:    {summary['hist_success_per_pair']}")
        print(f"fail-per-pair histogram:       {summary['hist_fail_per_pair']}")
    print(f"source trajectories sampled:   {trajs_sampled}/{n_usable_traj}"
          + ("" if not has_succ else f"  ({trajs_succ} with >=1 success)"))

    if eval_cov is not None:
        ec = eval_cov
        t = ec["total_episodes"]
        ep = ec["episodes"]
        print(f"\n=== EVAL-PAIR COVERAGE (eval_seed={ec['eval_seed']}, {t} seeded episodes, "
              f"offset={ec['eval_goal_offset']}) ===")
        print("the exact (init, goal) episodes the eval samples, by planned training support:")
        print(f"  on a SUCCEEDED planned pair:   {ep['success']:6d}  ({100*ep['success']/max(1,t):5.1f}%)"
              "   <- has training support")
        if has_succ:
            print(f"  on a fail-only planned pair:   {ep['fail_only']:6d}  "
                  f"({100*ep['fail_only']/max(1,t):5.1f}%)")
        print(f"  on a never-sampled pair:       {ep['never']:6d}  ({100*ep['never']/max(1,t):5.1f}%)"
              "   <- no training support")
        print(f"  ({ec['unique_pairs']} distinct pairs among the {t} eval episodes)")

    print(f"\n[coverage] wrote {args.out} and {csv_path}")


if __name__ == "__main__":
    main()
