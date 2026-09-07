"""
Dataset for WM-planned trajectories produced by generate_planned_trajectories.py.

visual:  list of uint8 tensors (T, H, W, C)  in [0, 255]
actions:  list of float tensors (T, A)  (normalized, same space as the
states:  list of float tensors (T, Ds)
proprios:  list of float tensors (T, Dp)
"""
import torch
import random
from pathlib import Path
from einops import rearrange
from typing import Callable, Optional

from .traj_dset import TrajDataset


class PlannedTrajDataset(TrajDataset):
    def __init__(self, data_path: str, transform: Optional[Callable] = None,
                 only_successes: bool = False, max_trajectories: Optional[int] = None,
                 selection: str = "first", selection_seed: int = 0):
        """only_successes: keep only trajectories that reached the goal (needs the
        'successes' metadata). max_trajectories: cap how many to keep, applied after the
        success filter (None = all).

        'first'  -> the first N in file order (the original, deterministic behavior)
        'recent' -> the LAST N: a chronological queue over a growing pool, since shards
        'random' -> N drawn uniformly without replacement from the whole pool, seeded by
        'random_with_replacement' -> N drawn WITH replacement, so a trajectory can appear
        """
        self.data_path = Path(data_path)
        self.transform = transform
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

        # select trajectories, success filter first, then cap the count
        keep = list(range(len(seq_lengths)))
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
    def _load_payload(path: Path):
        """Load a single .pth file, or merge a directory of shard_*.pth files
        (produced by generate_planned_trajectories.py with shard_every > 0)."""
        if not path.is_dir():
            return torch.load(path, map_location="cpu", weights_only=False)
        shard_files = sorted(path.glob("shard_*.pth"))
        if not shard_files:
            raise FileNotFoundError(f"No shard_*.pth files found in {path}")
        list_keys = ["visual", "actions", "states", "proprios", "seq_lengths",
                     "init_states", "goal_states", "successes"]
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
        obs = {"visual": image, "proprio": proprio}
        act = self.actions[idx][frames].float()
        state = self.states[idx][frames].float()
        return obs, act, state, {}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)
