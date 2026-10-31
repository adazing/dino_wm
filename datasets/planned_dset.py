"""
Dataset for WM-planned trajectories produced by generate_planned_trajectories.py.

visual:  list of uint8 tensors (T, H, W, C)  in [0, 255]
actions:  list of float tensors (T, A)  (normalized, same space as the
states:  list of float tensors (T, Ds)
proprios:  list of float tensors (T, Dp)
"""
import numpy as np
import torch
import random
from pathlib import Path
from einops import rearrange
from typing import Callable, Optional

from .traj_dset import TrajDataset


# Every per-trajectory list a planned file holds. generate_planned_trajectories.py builds its
# payload and clears its shard buffers from this, and _load_payload merges exactly these, so a
# new field only has to be added here.
class NoMatchingTrajectories(ValueError):
    """The pool legitimately holds none of the requested outcome.

    Distinct from a bad `select` name or missing metadata, which are programming/config errors.
    train.py skips a group that raises THIS and lets anything else propagate, so a typo cannot
    masquerade as "the planner has not failed at anything yet".
    """


TRAJ_KEYS = ("visual", "actions", "states", "proprios", "seq_lengths",
             "init_states", "goal_states", "successes",
             "wm_pred_final_states", "wm_latent_err", "dreamed_latents")


class PlannedTrajDataset(TrajDataset):
    def __init__(self, data_path: str, transform: Optional[Callable] = None,
                 only_successes: bool = False, max_trajectories: Optional[int] = None,
                 selection: str = "first", selection_seed: int = 0,
                 select: str = "all", wm_err_pct: float = 50.0,
                 proprio_mean=None, proprio_std=None):
        """only_successes: keep only trajectories that reached the goal (needs the
        'successes' metadata). max_trajectories: cap how many to keep, applied after the
        success filter (None = all).

        'first'  -> the first N in file order (the original, deterministic behavior)
        'recent' -> the LAST N: a chronological queue over a growing pool, since shards
        'random' -> N drawn uniformly without replacement from the whole pool, seeded by
        'random_with_replacement' -> N drawn WITH replacement, so a trajectory can appear

        select picks which trajectories are worth training a world model on. wm_fail means the
        endpoint the world model predicted was further from the real endpoint than wm_err_pct
        percent of this file's trajectories, so the threshold moves as the model improves.
          all                 every trajectory
          policy_fail         the plan did not reach its goal
          wm_fail             the world model mispredicted the endpoint
          wm_ok               the complement of wm_fail: error BELOW the percentile
          wm_and_policy_fail  both
          wm_fail_policy_ok   mispredicted but the plan still reached the goal
        """
        self.data_path = Path(data_path)
        self.transform = transform
        # Generation stores proprio RAW; the demo loader normalises it, and that is what
        # the world model was trained on. Raw puts it ~30 std off-centre into a FROZEN
        # proprio encoder: wall fine-tune training loss 12 vs 0.011 on demonstrations.
        self._proprio_mean = (None if proprio_mean is None
                              else torch.as_tensor(proprio_mean).float())
        self._proprio_std = (None if proprio_std is None
                             else torch.as_tensor(proprio_std).float().clamp_min(1e-6))
        payload = self._load_payload(self.data_path)
        visual = payload["visual"]
        actions = payload["actions"]
        states = payload["states"]
        proprios = payload["proprios"]
        seq_lengths = list(payload["seq_lengths"])
        # optional per-trajectory metadata, older files may not have these
        init_states = payload.get("init_states")   # planned initial state
        goal_states = payload.get("goal_states")   # planning goal state
        successes = payload.get("successes")   # reached-goal tag

        wm_pred = payload.get("wm_pred_final_states")   # needs probe_ckpt at generation time

        # select trajectories, success filter first, then cap the count
        keep = list(range(len(seq_lengths)))
        self.n_pool = len(seq_lengths)   # trajectories in the pool before any filtering
        keep = self._apply_select(keep, str(select), float(wm_err_pct),
                                  successes, wm_pred, states, seq_lengths,
                                  payload.get("wm_latent_err"))
        self.n_selected = len(keep)   # what select left, before only_successes/max_trajectories
        if only_successes:
            if successes is None:
                raise ValueError(
                    "only_successes=True but this file has no 'successes' metadata; "
                    "regenerate with the current generate_planned_trajectories.py."
                )
            keep = [i for i in keep if bool(successes[i])]
        # Check before the cap, random_with_replacement on an empty pool would otherwise die in
        # rng.choice([]) with a bare IndexError instead of this message.
        if len(keep) == 0:
            raise ValueError(
                f"No planned trajectories left after only_successes={only_successes} "
                f"({len(seq_lengths)} in the pool). Generate more, or set "
                f"planned_only_successes=false.")
        if max_trajectories is not None:
            n = int(max_trajectories)
            sel = str(selection)
            # only 'random_with_replacement' can exceed the pool size, every other mode has to
            # have the trajectories it is asked for.
            if n > len(keep) and sel != "random_with_replacement":
                raise ValueError(
                    f"planned_max_trajectories={n} > {len(keep)} available planned "
                    f"trajectories (after only_successes={only_successes}). Lower it to <= "
                    f"{len(keep)}, generate more, or use selection=random_with_replacement.")
            if sel == "first":
                keep = keep[:n]
            elif sel == "recent":
                keep = keep[-n:]   # chronological queue, newest N
            elif sel in ("random", "random_with_replacement"):
                rng = random.Random(int(selection_seed))
                keep = (rng.sample(keep, n) if sel == "random"
                        else [rng.choice(keep) for _ in range(n)])
                if sel == "random":
                    keep.sort()   # keep file order for reproducible indexing
            else:
                raise ValueError(
                    f"selection must be 'first' | 'recent' | 'random' | "
                    f"'random_with_replacement', got {selection!r}")
        print(f"[planned] {self.data_path}: {len(keep)}/{len(seq_lengths)} trajectories kept "
              f"(only_successes={only_successes}, max_trajectories={max_trajectories}, "
              f"selection={selection})")
        self.n_kept = len(keep)   # what actually gets trained on
        if len(keep) == 0:
            raise ValueError(
                f"No planned trajectories left after filtering "
                f"(only_successes={only_successes}, max_trajectories={max_trajectories})."
            )

        def _sel(lst):
            return [lst[i] for i in keep] if lst is not None else None

        self.visual = _sel(visual)
        self.actions = _sel(actions)
        self.states = _sel(states)
        self.proprios = _sel(proprios)
        self.seq_lengths = [seq_lengths[i] for i in keep]
        self.init_states = _sel(init_states)
        self.goal_states = _sel(goal_states)
        self.successes = _sel(successes)
        print(f"PlannedTrajDataset: using {len(self.seq_lengths)}/{len(seq_lengths)} trajectories "
              f"(only_successes={only_successes}, max_trajectories={max_trajectories}, "
              f"selection={selection}"
              + (f", selection_seed={selection_seed}" if str(selection).startswith("random") else "")
              + ")")

        self.action_dim = self.actions[0].shape[-1]
        self.state_dim = self.states[0].shape[-1]
        self.proprio_dim = self.proprios[0].shape[-1]

    @staticmethod
    def _apply_select(keep, select, wm_err_pct, successes, wm_pred, states, seq_lengths,
                      wm_latent_err=None):
        """Keep the trajectories `select` asks for. Returns keep unchanged for 'all'."""
        if select == "all":
            return keep
        modes = ("all", "policy_fail", "wm_fail", "wm_ok", "wm_and_policy_fail",
                 "wm_fail_policy_ok")
        if select not in modes:
            raise ValueError(f"select must be one of {modes}, got {select!r}")
        if successes is None:
            raise ValueError(f"select={select!r} needs 'successes' metadata in the planned file.")
        needs_wm = select in ("wm_fail", "wm_ok", "wm_and_policy_fail", "wm_fail_policy_ok")
        if needs_wm and not wm_latent_err and not wm_pred:
            raise ValueError(
                f"select={select!r} needs a world-model endpoint error. Generation writes "
                f"'wm_latent_err' on any run and 'wm_pred_final_states' when probe_ckpt is set, "
                f"and this file has neither. Regenerate, or use select=policy_fail.")

        err = {}
        if needs_wm:
            if wm_latent_err:
                # latent units, only the ordering matters since the threshold is a percentile
                for i in keep:
                    err[i] = float(wm_latent_err[i])
                units = "latent"
            else:
                # probe path, the error is in world units
                for i in keep:
                    real = torch.as_tensor(states[i][int(seq_lengths[i]) - 1]).float().flatten()
                    pred = torch.as_tensor(wm_pred[i]).float().flatten()
                    n = min(len(real), len(pred))
                    err[i] = float(torch.linalg.norm(pred[:n] - real[:n]))
                units = "state"
            thr = float(np.percentile(list(err.values()), wm_err_pct)) if err else 0.0

        out = []
        for i in keep:
            ok = bool(successes[i])
            wm_bad = needs_wm and err[i] >= thr
            if select == "policy_fail" and not ok:
                out.append(i)
            elif select == "wm_ok" and not wm_bad:
                out.append(i)
            elif select == "wm_fail" and wm_bad:
                out.append(i)
            elif select == "wm_and_policy_fail" and wm_bad and not ok:
                out.append(i)
            elif select == "wm_fail_policy_ok" and wm_bad and ok:
                out.append(i)
        if not out:
            n_fail = sum(1 for i in keep if not bool(successes[i]))
            n_wm = sum(1 for i in keep if needs_wm and err[i] >= thr)
            raise NoMatchingTrajectories(
                f"select={select!r} matched no trajectories out of {len(keep)}. "
                f"{n_fail} failed, {n_wm} were mispredicted"
                + (f" (threshold {thr:.3f} {units})" if needs_wm else "")
                + ". The two sets do not overlap, so lower wm_finetune_err_pct to widen the "
                  "mispredicted set, or use select=wm_fail, which a percentile always fills.")
        print(f"[planned] select={select} kept {len(out)}/{len(keep)} trajectories"
              + (f", wm endpoint error threshold {thr:.3f} ({units})" if needs_wm else ""))
        return out


    @staticmethod
    def _load_payload(path: Path):
        """Load a single .pth file, or merge a directory of shard_*.pth files
        (produced by generate_planned_trajectories.py with shard_every > 0)."""
        if not path.is_dir():
            return torch.load(path, map_location="cpu", weights_only=False)
        shard_files = sorted(path.glob("shard_*.pth"))
        if not shard_files:
            raise FileNotFoundError(f"No shard_*.pth files found in {path}")
        list_keys = list(TRAJ_KEYS)
        merged = None
        for sf in shard_files:
            p = torch.load(sf, map_location="cpu", weights_only=False)
            if merged is None:
                merged = {k: (list(p[k]) if p.get(k) is not None else None) for k in list_keys}
                merged["env_name"] = p.get("env_name")
                merged["frameskip"] = p.get("frameskip")
            else:
                for k in list_keys:
                    if merged[k] is not None and p.get(k) is not None:
                        merged[k].extend(p[k])
        print(f"PlannedTrajDataset: merged {len(shard_files)} shards from {path}")
        return merged

    def get_seq_length(self, idx):
        return int(self.seq_lengths[idx])

    def get_all_actions(self):
        return torch.cat(
            [self.actions[i][: self.get_seq_length(i)].float() for i in range(len(self))],
            dim=0,
        )

    def get_frames(self, idx, frames):
        frames = list(frames)
        image = self.visual[idx][frames].float() / 255.0   # , T, H, W, C
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        proprio = self.proprios[idx][frames].float()
        if self._proprio_mean is not None:
            proprio = (proprio - self._proprio_mean) / self._proprio_std
        obs = {"visual": image, "proprio": proprio}
        act = self.actions[idx][frames].float()
        state = self.states[idx][frames].float()
        return obs, act, state, {}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)
