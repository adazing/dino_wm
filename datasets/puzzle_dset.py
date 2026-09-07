"""Dataset loader for OGBench Puzzle demos produced by tools/gen_puzzle_dset.py.

obses/episode_00000.npy  (T, H, W, 3) uint8, memory-mapped on read
states.pth  (N, T, state_dim) float32, qpos | qvel | button_states
proprios.pth  (N, T, 19)  float32
actions.pth  (N, T, 5)  float32, raw env actions in [-1, 1]
seq_lengths.pkl  list[int]
meta.pkl  dict (env_type, board shape, nq, nv, img_size, ...)
"""

import pickle
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from einops import rearrange

from .traj_dset import TrajDataset, TrajSlicerDataset

# Below this, a dimension is treated as constant and left unscaled, qvel entries that are always
# zero at episode starts would otherwise blow up.
_MIN_STD = 1e-3

#---------------------------------------------------------------------------------------------
# shared action NORMALISATION, the pusht pattern, see datasets/pusht_dset.py ACTION_MEAN/STD.
#   planned trajectories store actions normalised by the WM's dataset  (planned_dset.py)
#   train_policy then DENORMALISES them with the main dataset's stats  (train_policy.py:860)
#   actions  5-D, dx, dy, dz, dyaw, dgripper
#   proprio 19-D  PROPRIO_DIM in env/puzzle/puzzle_wrapper.py, joint_pos(6) + joint_vel(6)
ACTION_MEAN = None
ACTION_STD = None
PROPRIO_MEAN = None
PROPRIO_STD = None


class PuzzleDataset(TrajDataset):
    def __init__(
        self,
        data_path: str,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = True,
        split: Optional[str] = None,
        split_ratio: float = 0.9,
    ):
        self.data_path = Path(data_path)
        self.transform = transform

        self.states = torch.load(self.data_path / "states.pth").float()
        self.proprios = torch.load(self.data_path / "proprios.pth").float()
        self.actions = torch.load(self.data_path / "actions.pth").float()

        with open(self.data_path / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)

        meta_path = self.data_path / "meta.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                self.meta = pickle.load(f)
        else:
            self.meta = {}

        obs_dir = self.data_path / "obses"
        if not obs_dir.exists():
            raise FileNotFoundError(f"No obses/ directory under {self.data_path}")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        # Stats are computed over every episode in the directory, before the split.
        if normalize_action:
            full_lengths = list(self.seq_lengths)
            # actions, shared constants if set, see the ACTION_MEAN note at the top of this file,
            # so expert/noisy/regenerated folders all land in one action space and the
            # normalise-here / denormalise-there round trip cancels exactly.
            if ACTION_MEAN is not None and ACTION_STD is not None:
                if ACTION_MEAN.shape[-1] != self.action_dim:
                    raise ValueError(
                        f"ACTION_MEAN has {ACTION_MEAN.shape[-1]} dims but this dataset's "
                        f"actions have {self.action_dim}. Puzzle actions are 5-D for every "
                        f"board size, so this means the constants came from a different env.")
                self.action_mean = ACTION_MEAN.float()
                self.action_std = ACTION_STD.float().clamp(min=_MIN_STD)
            else:
                self.action_mean, self.action_std = self._stats(self.actions, full_lengths)
            # proprio, shared constants if set, same reasoning as actions (19-D on every board).
            if PROPRIO_MEAN is not None and PROPRIO_STD is not None:
                if PROPRIO_MEAN.shape[-1] != self.proprio_dim:
                    raise ValueError(
                        f"PROPRIO_MEAN has {PROPRIO_MEAN.shape[-1]} dims but this dataset's "
                        f"proprio has {self.proprio_dim}. Puzzle proprio is 19-D for every "
                        f"board size, so this means the constants came from a different env.")
                self.proprio_mean = PROPRIO_MEAN.float()
                self.proprio_std = PROPRIO_STD.float().clamp(min=_MIN_STD)
            else:
                self.proprio_mean, self.proprio_std = self._stats(self.proprios, full_lengths)
            # state stays data-derived, its width does change with board size.
            self.state_mean, self.state_std = self._stats(self.states, full_lengths)
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)

        # Episode-level split, applied after the stats above.
        total = len(self.seq_lengths)
        if split in ("train", "val"):
            n_train = int(total * split_ratio)
            ids = list(range(n_train)) if split == "train" else list(range(n_train, total))
        else:
            ids = list(range(total))
        if n_rollout:
            ids = ids[:n_rollout]
        self.episode_ids = ids

        self.states = self.states[ids]
        self.proprios = self.proprios[ids]
        self.actions = self.actions[ids]
        self.seq_lengths = [self.seq_lengths[i] for i in ids]

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std
        # states stay raw on purpose, env.prepare() consumes them directly.

        print(
            f"Loaded {len(ids)} puzzle rollouts from {self.data_path} "
            f"(split={split}, board={self.meta.get('env_type', '?')}, "
            f"state_dim={self.state_dim}, proprio_dim={self.proprio_dim})"
        )

    @staticmethod
    def _stats(tensor, lengths):
        """Mean/std over valid (non-padded) timesteps only."""
        flat = torch.cat([tensor[i, : lengths[i]] for i in range(len(lengths))], dim=0)
        mean = flat.mean(dim=0)
        std = flat.std(dim=0)
        # A constant dimension, the gripper channel is pinned closed for the whole dataset and
        # qvel is zero at every episode start would otherwise divide by ~0.
        std = torch.where(std < _MIN_STD, torch.ones_like(std), std)
        return mean, std

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        return torch.cat(
            [self.actions[i, : self.seq_lengths[i]] for i in range(len(self.seq_lengths))],
            dim=0,
        )

    def _read_obs(self, idx, frames):
        """Read only `frames` from episode `idx`."""
        ep_id = self.episode_ids[idx]
        path = self.data_path / "obses" / f"episode_{ep_id:05d}.npy"
        if path.exists():
            episode = np.load(path, mmap_mode="r")
            image = np.asarray(episode[frames])   # fancy index -> writable copy of the window
            image = torch.from_numpy(image)
        else:   # legacy torch-serialised dumps
            episode = torch.load(str(path.with_suffix(".pth")))
            if isinstance(episode, np.ndarray):
                episode = torch.from_numpy(episode)
            image = episode[frames]
        image = image.float() / 255.0   # THWC uint8 -> float
        return rearrange(image, "T H W C -> T C H W")

    def get_frames(self, idx, frames):
        frames = list(frames)
        image = self._read_obs(idx, frames)
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": self.proprios[idx, frames]}
        # 'shape' is a placeholder so the tuple matches what plan.py expects to hand to
        # update_env(), the puzzle board is fixed at construction, so it carries nothing.
        return obs, self.actions[idx, frames], self.states[idx, frames], {"shape": 0}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs)
        return rearrange(imgs, "b h w c -> b c h w") / 255.0


def load_puzzle_slice_train_val(
    transform,
    n_rollout=None,
    data_path="data/puzzle_3x3_expert",
    normalize_action=True,
    split_ratio=0.9,
    num_hist=0,
    num_pred=0,
    frameskip=0,
):
    base = Path(data_path)
    if (base / "train").exists() and (base / "val").exists():
        train_dset = PuzzleDataset(
            data_path=str(base / "train"), n_rollout=n_rollout, transform=transform,
            normalize_action=normalize_action,
        )
        val_dset = PuzzleDataset(
            data_path=str(base / "val"), n_rollout=n_rollout, transform=transform,
            normalize_action=normalize_action,
        )
    else:
        train_dset = PuzzleDataset(
            data_path=str(base), n_rollout=n_rollout, transform=transform,
            normalize_action=normalize_action, split="train", split_ratio=split_ratio,
        )
        val_dset = PuzzleDataset(
            data_path=str(base), n_rollout=n_rollout, transform=transform,
            normalize_action=normalize_action, split="val", split_ratio=split_ratio,
        )

    num_frames = num_hist + num_pred
    datasets = {
        "train": TrajSlicerDataset(train_dset, num_frames, frameskip),
        "valid": TrajSlicerDataset(val_dset, num_frames, frameskip),
    }
    traj_dset = {"train": train_dset, "valid": val_dset}
    return datasets, traj_dset
