import torch
import decord
import pickle
import numpy as np
from pathlib import Path
from einops import rearrange
from decord import VideoReader
from typing import Callable, Optional
from .traj_dset import TrajDataset, TrajSlicerDataset
from typing import Optional, Callable, Any
decord.bridge.set_bridge("torch")

# precomputed dataset stats
ACTION_MEAN = torch.tensor([-0.0087, 0.0068])
ACTION_STD = torch.tensor([0.2019, 0.2002])
STATE_MEAN = torch.tensor([236.6155, 264.5674, 255.1307, 266.3721, 1.9584, -2.93032027,  2.54307914])
STATE_STD = torch.tensor([101.1202, 87.0112, 52.7054, 57.4971, 1.7556, 74.84556075, 74.14009094])
PROPRIO_MEAN = torch.tensor([236.6155, 264.5674, -2.93032027,  2.54307914])
PROPRIO_STD = torch.tensor([101.1202, 87.0112, 74.84556075, 74.14009094])

class PushTDataset(TrajDataset):
    def __init__(
        self,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        data_path: str = "data/pusht_dataset",
        normalize_action: bool = True,
        relative=True,
        action_scale=100.0,
        with_velocity: bool = True,   # agent's velocity
        split: Optional[str] = None,   # None | "train" | "val", split a single dir by episode
        split_ratio: float = 0.9,   # fraction of episodes for train when split is set
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.relative = relative
        self.action_scale = action_scale
        self.normalize_action = normalize_action
        self.states = torch.load(self.data_path / "states.pth")
        self.states = self.states.float()

        # Actions. dino_wm's env is relative (target = agent_pos + action*scale), so `relative`
        # actions are the native space.
        rel_path = self.data_path / "rel_actions.pth"
        abs_path = self.data_path / "abs_actions.pth"
        if relative:
            if rel_path.exists():
                self.actions = torch.load(rel_path).float()
            else:
                abs_actions = torch.load(abs_path).float()
                self.actions = abs_actions - self.states[..., :2]   # (..., 2) agent pos
        else:
            self.actions = torch.load(abs_path).float()
        self.actions = self.actions / action_scale   # scaled back up in env

        # obses are either per-episode .mp4 (dino_wm) or .pth THWC tensors (patch_policy)
        obs_dir = self.data_path / "obses"
        if (obs_dir / "episode_000.pth").exists():
            self.obs_format = "pth"
        elif (obs_dir / "episode_000.mp4").exists():
            self.obs_format = "mp4"
        else:
            raise FileNotFoundError(f"No episode_000.pth or .mp4 obs files under {obs_dir}")

        with open(self.data_path / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)

        # load shapes, assume all shapes are 'T' if file not found
        shapes_file = self.data_path / "shapes.pkl"
        if shapes_file.exists():
            with open(shapes_file, 'rb') as f:
                shapes = pickle.load(f)
                self.shapes = shapes
        else:
            self.shapes = ['T'] * len(self.states)

        # pick which disk episodes belong to this split.
        total = len(self.states)
        if split in ("train", "val"):
            n_train = int(total * split_ratio)
            ids = list(range(0, n_train)) if split == "train" else list(range(n_train, total))
        else:
            ids = list(range(total))
        self.n_rollout = n_rollout
        if n_rollout:
            ids = ids[:n_rollout]
        self.episode_ids = ids
        n = len(ids)

        self.states = self.states[ids]
        self.actions = self.actions[ids]
        self.seq_lengths = [self.seq_lengths[i] for i in ids]
        self.shapes = [self.shapes[i] for i in ids]
        self.proprios = self.states[..., :2].clone()   # For pusht, first 2 dim of states is proprio
        # load velocities and update states and proprios
        self.with_velocity = with_velocity
        if with_velocity:
            vel_path = self.data_path / "velocities.pth"
            if vel_path.exists():
                self.velocities = torch.load(vel_path)[ids].float()
            else:
                # patch_policy dumps have no velocities, approximate agent velocity by
                # finite-differencing agent position at control_hz=10 (env's fps).
                self.velocities = self._finite_diff_velocity(self.states[..., :2])
            self.states = torch.cat([self.states, self.velocities], dim=-1)
            self.proprios = torch.cat([self.proprios, self.velocities], dim=-1)
        print(f"Loaded {n} rollouts (obs={self.obs_format}, split={split})")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            self.action_mean = ACTION_MEAN
            self.action_std = ACTION_STD
            self.state_mean = STATE_MEAN[:self.state_dim]
            self.state_std = STATE_STD[:self.state_dim]
            self.proprio_mean = PROPRIO_MEAN[:self.proprio_dim]
            self.proprio_std = PROPRIO_STD[:self.proprio_dim]
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    @staticmethod
    def _finite_diff_velocity(pos, control_hz=10.0):
        # pos, (n, T, 2) -> velocity (n, T, 2), v[t] = (pos[t]-pos[t-1])*control_hz, v[0]=0
        vel = torch.zeros_like(pos)
        vel[:, 1:] = (pos[:, 1:] - pos[:, :-1]) * control_hz
        return vel

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def _read_obs(self, idx, frames):
        obs_dir = self.data_path / "obses"
        ep_id = self.episode_ids[idx]   # local index -> on-disk episode id
        if self.obs_format == "mp4":
            reader = VideoReader(str(obs_dir / f"episode_{ep_id:03d}.mp4"), num_threads=1)
            image = reader.get_batch(frames)   # THWC
        else:   # patch_policy, per-episode THWC uint8 tensor
            episode = torch.load(str(obs_dir / f"episode_{ep_id:03d}.pth"))
            image = episode[frames]   # THWC
        image = image.float() / 255.0
        return rearrange(image, "T H W C -> T C H W")

    def get_frames(self, idx, frames):
        frames = list(frames)
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]
        shape = self.shapes[idx]

        image = self._read_obs(idx, frames)
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, {'shape': shape}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0


def load_pusht_slice_train_val(
    transform,
    n_rollout=50,
    data_path="data/pusht_dataset",
    normalize_action=True,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    with_velocity=True,
    relative=True,
    action_scale=100.0,
):
    # relative / action_scale forwarded to PushTDataset so a config can be explicit about the
    # action representation.
    base = Path(data_path)
    if (base / "train").exists() and (base / "val").exists():
        # dino_wm layout, separate train/ and val/ directories
        train_dset = PushTDataset(
            n_rollout=n_rollout, transform=transform, data_path=str(base / "train"),
            normalize_action=normalize_action, with_velocity=with_velocity,
            relative=relative, action_scale=action_scale,
        )
        val_dset = PushTDataset(
            n_rollout=n_rollout, transform=transform, data_path=str(base / "val"),
            normalize_action=normalize_action, with_velocity=with_velocity,
            relative=relative, action_scale=action_scale,
        )
    else:
        # single directory (e.g.
        train_dset = PushTDataset(
            n_rollout=n_rollout, transform=transform, data_path=str(base),
            normalize_action=normalize_action, with_velocity=with_velocity,
            relative=relative, action_scale=action_scale,
            split="train", split_ratio=split_ratio,
        )
        val_dset = PushTDataset(
            n_rollout=n_rollout, transform=transform, data_path=str(base),
            normalize_action=normalize_action, with_velocity=with_velocity,
            relative=relative, action_scale=action_scale,
            split="val", split_ratio=split_ratio,
        )

    num_frames = num_hist + num_pred
    train_slices = TrajSlicerDataset(train_dset, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val_dset, num_frames, frameskip)

    datasets = {}
    datasets["train"] = train_slices
    datasets["valid"] = val_slices
    traj_dset = {}
    traj_dset["train"] = train_dset
    traj_dset["valid"] = val_dset
    return datasets, traj_dset
