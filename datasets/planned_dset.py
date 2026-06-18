"""
Dataset for WM-planned trajectories produced by generate_planned_trajectories.py.

Stored as a single torch file: a dict with per-trajectory lists
    visual:      list of uint8 tensors (T, H, W, C)   in [0, 255]
    actions:     list of float tensors (T, A)         (normalized, same space as the
                                                        original dino_wm dataset)
    states:      list of float tensors (T, Ds)
    proprios:    list of float tensors (T, Dp)
    seq_lengths: list[int]

Implements the TrajDataset interface (get_seq_length / get_frames /
get_all_actions) so it plugs into datasets.policy_dataset.DinoWMTrajForPolicy
exactly like the original demonstration datasets. Construct with transform=None
to keep visuals in [0, 1] for the patch encoder.
"""
import torch
from pathlib import Path
from einops import rearrange
from typing import Callable, Optional

from .traj_dset import TrajDataset


class PlannedTrajDataset(TrajDataset):
    def __init__(self, data_path: str, transform: Optional[Callable] = None):
        self.data_path = Path(data_path)
        self.transform = transform
        payload = torch.load(self.data_path, map_location="cpu", weights_only=False)
        self.visual = payload["visual"]
        self.actions = payload["actions"]
        self.states = payload["states"]
        self.proprios = payload["proprios"]
        self.seq_lengths = list(payload["seq_lengths"])

        self.action_dim = self.actions[0].shape[-1]
        self.state_dim = self.states[0].shape[-1]
        self.proprio_dim = self.proprios[0].shape[-1]

    def get_seq_length(self, idx):
        return int(self.seq_lengths[idx])

    def get_all_actions(self):
        return torch.cat(
            [self.actions[i][: self.get_seq_length(i)].float() for i in range(len(self))],
            dim=0,
        )

    def get_frames(self, idx, frames):
        frames = list(frames)
        image = self.visual[idx][frames].float() / 255.0  # (T, H, W, C)
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
