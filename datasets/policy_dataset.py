"""
Data bridge for training BC policies (VQ-BeT / Diffusion) on dino_wm data.

Pipeline:
    dino_wm raw TrajDataset  (obs dict {visual, proprio}, act, state, info)
      -> DinoWMTrajForPolicy  (exposes get_frames -> (visual (T,V,C,H,W), act, mask))
      -> [optionally ConcatTrajForPolicy for the "both" data mode]
      -> split_traj_datasets   (train / val)
      -> TrajectoryEmbeddingDataset  (precompute frozen encoder embeddings)
      -> VqbetTrajectorySlicerDataset (windowed obs + action chunk)
      -> DataLoader

Visuals are in [0, 1] (default_transform), which is what the encoders expect.
"""
import abc
import torch
import numpy as np
from torch import default_generator, randperm
from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Callable, Sequence, List


# tensor helpers
def repeat_start_to_length(x: torch.Tensor, length: int, dim: int = 0):
    """Pad x to `length` along `dim`, repeating the first value at the start."""
    pad_size = length - x.shape[dim]
    if pad_size <= 0:
        return x
    first = x.index_select(dim, torch.tensor(0, device=x.device))
    repeat_shape = [1] * len(x.shape)
    repeat_shape[dim] = pad_size
    return torch.cat([first.repeat(*repeat_shape), x], dim=dim)


def repeat_end_to_length(x: torch.Tensor, length: int, dim: int = 0):
    """Pad x to `length` along `dim`, repeating the last value at the end."""
    pad_size = length - x.shape[dim]
    if pad_size <= 0:
        return x
    last = x.index_select(dim, torch.tensor(x.shape[dim] - 1, device=x.device))
    repeat_shape = [1] * len(x.shape)
    repeat_shape[dim] = pad_size
    return torch.cat([x, last.repeat(*repeat_shape)], dim=dim)


# base classes
class TrajectoryDataset(Dataset, abc.ABC):
    """TrajectoryDataset[i] -> (observations[T, ...], actions[T, ...], mask[T])."""

    @abc.abstractmethod
    def get_seq_length(self, idx):
        raise NotImplementedError

    @abc.abstractmethod
    def get_frames(self, idx, frames):
        raise NotImplementedError


class TrajectorySubset(TrajectoryDataset, Subset):
    def __init__(self, dataset: TrajectoryDataset, indices: Sequence[int]):
        Subset.__init__(self, dataset, indices)

    def get_seq_length(self, idx):
        return self.dataset.get_seq_length(self.indices[idx])

    def get_all_actions(self):
        return self.dataset.get_all_actions()

    def get_frames(self, idx, frames):
        return self.dataset.get_frames(self.indices[idx], frames)


def _accumulate(iterable, fn=lambda x, y: x + y):
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total


def random_split_traj(dataset, lengths, generator=default_generator):
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths != length of input dataset!")
    indices = randperm(sum(lengths), generator=generator).tolist()
    return [
        TrajectorySubset(dataset, indices[offset - length: offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def split_traj_datasets(dataset, train_fraction=0.95, random_seed=42):
    n = len(dataset)
    lengths = [int(train_fraction * n), n - int(train_fraction * n)]
    return random_split_traj(
        dataset, lengths, generator=torch.Generator().manual_seed(random_seed)
    )


# shim: adapt a dino_wm TrajDataset to the (obs, act, mask) interface
class DinoWMTrajForPolicy(TrajectoryDataset):
    """Expose a dino_wm trajectory dataset in the (obs_tensor, act, mask) form.

    The wrapped dataset must implement ``get_seq_length(idx)`` and either
    ``get_frames(idx, frames) -> (obs_dict, act, state, info)`` or be indexable
    to the same. ``obs_dict['visual']`` must be (T, C, H, W) in [0, 1].

    Image-only: proprio/state are dropped. A view dimension V=1 is inserted,
    giving visual (T, 1, C, H, W).
    """

    def __init__(self, dataset, visual_key: str = "visual"):
        self.dataset = dataset
        self.visual_key = visual_key

    def __len__(self):
        return len(self.dataset)

    def get_seq_length(self, idx):
        return int(self.dataset.get_seq_length(idx))

    def _extract(self, sample):
        obs, act = sample[0], sample[1]
        visual = obs[self.visual_key] if isinstance(obs, dict) else obs
        if not torch.is_tensor(visual):
            visual = torch.as_tensor(visual)
        visual = visual.float()
        if visual.ndim == 4:  # (T, C, H, W) -> add view dim
            visual = visual.unsqueeze(1)  # (T, 1, C, H, W)
        if not torch.is_tensor(act):
            act = torch.as_tensor(act)
        act = act.float()
        mask = torch.ones(visual.shape[0], dtype=torch.bool)
        return visual, act, mask

    def get_frames(self, idx, frames):
        if hasattr(self.dataset, "get_frames"):
            sample = self.dataset.get_frames(idx, frames)
        else:
            sample = self.dataset[idx]
            sample = tuple(x[frames] if torch.is_tensor(x) else x for x in sample)
        return self._extract(sample)

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def get_all_actions(self):
        if hasattr(self.dataset, "get_all_actions"):
            return self.dataset.get_all_actions().float()
        actions = []
        for i in range(len(self.dataset)):
            _, act, _ = self.get_frames(i, range(self.get_seq_length(i)))
            actions.append(act)
        return torch.cat(actions, dim=0)


class ConcatTrajForPolicy(TrajectoryDataset):
    """Concatenate several TrajectoryDatasets (for the 'both' data mode)."""

    def __init__(self, datasets: List[TrajectoryDataset]):
        assert len(datasets) > 0
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cum = np.cumsum([0] + self.lengths)

    def __len__(self):
        return int(self.cum[-1])

    def _locate(self, idx):
        d = int(np.searchsorted(self.cum, idx, side="right") - 1)
        return d, idx - int(self.cum[d])

    def get_seq_length(self, idx):
        d, j = self._locate(idx)
        return self.datasets[d].get_seq_length(j)

    def get_frames(self, idx, frames):
        d, j = self._locate(idx)
        return self.datasets[d].get_frames(j, frames)

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def get_all_actions(self):
        return torch.cat([d.get_all_actions() for d in self.datasets], dim=0)


# embedding + slicing
@torch.no_grad()
def embed_trajectory_dataset(model, dataset, device="cpu"):
    """Precompute frozen-encoder embeddings for every trajectory.

    Returns a list of (obs_enc[T, V, P, E], act[T, A], mask[T]) per trajectory.
    """
    model_device = next(model.parameters()).device
    result_device = torch.device(device)
    was_training = model.training
    model.eval()
    result = []
    try:
        for i in range(len(dataset)):
            obs, *rest = dataset[i]  # obs: (T, V, C, H, W) in [0, 1]
            obs = obs.to(model_device)
            obs_enc = model(obs).detach().to(result_device)  # (T, V, P, E)
            rest = [x.to(result_device) for x in rest]
            result.append((obs_enc, *rest))
    finally:
        model.train(was_training)
    return result


class TrajectoryEmbeddingDataset(TrajectoryDataset):
    def __init__(self, model, dataset, device="cpu"):
        self.data = embed_trajectory_dataset(model, dataset, device=device)
        assert len(self.data) == len(dataset)
        self.seq_lengths = [len(x[0]) for x in self.data]
        n_tensors = len(self.data[0])
        self.on_device_data = [
            pad_sequence([x[i] for x in self.data], batch_first=True).to(device)
            for i in range(n_tensors)
        ]
        self.data = self.on_device_data

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        # actions are element 1; only take valid (unpadded) frames
        return torch.cat(
            [self.data[1][i, : self.seq_lengths[i]] for i in range(len(self.seq_lengths))],
            dim=0,
        )

    def get_frames(self, idx, frames):
        return [x[idx, frames] for x in self.data]

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)


class VqbetTrajectorySlicerDataset(TrajectoryDataset):
    """Slice trajectories into (obs_window, action_chunk, mask) windows.

    With ``vqbet_get_future_action_chunk=False`` the action chunk has length
    ``window + action_window - 1`` (obs and actions overlap by one step).
    """

    def __init__(
        self,
        dataset: TrajectoryDataset,
        window: int,
        action_window: int,
        vqbet_get_future_action_chunk: bool = False,
        transform: Optional[Callable] = None,
        pad_seq_length: bool = True,
        goal_conditional: bool = False,
    ):
        self.dataset = dataset
        self.window = window
        self.action_window = action_window
        self.vqbet_get_future_action_chunk = vqbet_get_future_action_chunk
        self.transform = transform
        self.pad_seq_length = pad_seq_length
        # If True, __getitem__ returns (obs, act, goal) where goal is the
        # trajectory's final-frame embedding (hindsight goal). For planned data
        # the endpoint IS the planner's goal; for demos it's task completion.
        self.goal_conditional = goal_conditional
        self.slices = []

        min_window_required = window + action_window - 1
        for i in range(len(self.dataset)):
            T = self.dataset.get_seq_length(i)
            self.slices += [(i, 0, end + 1) for end in range(window - 1)]
            if self.pad_seq_length:
                if T - self.window >= 0:
                    self.slices += [
                        (i, start, start + self.window)
                        for start in range(T - self.window + 1)
                    ]
            else:
                if T - min_window_required < 0:
                    print(f"Ignored short sequence #{i}: len={T}, window={min_window_required}")
                else:
                    self.slices += [
                        (i, start, start + self.window)
                        for start in range(T - min_window_required + 1)
                    ]

    def get_seq_length(self, idx):
        return self.window

    def get_all_actions(self):
        return self.dataset.get_all_actions()

    def get_frames(self, idx, frames):
        raise NotImplementedError("Slicer is indexed via __getitem__ only.")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        obs, act, *others = self.dataset[i]

        if end - start < self.window:
            obs_win = repeat_start_to_length(obs[start:end], self.window, dim=0)
            act = repeat_start_to_length(
                act[start: end - 1 + self.action_window],
                self.window + self.action_window - 1,
                dim=0,
            )
            repeated_others = [
                repeat_start_to_length(other[start:end], self.window, dim=0)
                for other in others
            ]
        else:
            obs_win = obs[start:end]
            act = act[start: end - 1 + self.action_window]
            repeated_others = [other[start:end] for other in others]

        if self.vqbet_get_future_action_chunk:
            expected_len = self.action_window
            act = act[self.window - 1:]
        else:
            expected_len = self.window + self.action_window - 1

        if act.shape[0] < expected_len:
            act = repeat_end_to_length(act, expected_len, dim=0)

        if self.goal_conditional:
            # hindsight goal: the trajectory's final-frame embedding, repeated
            # across the obs window. obs is the full traj (T, V, P, E).
            goal_win = obs[-1:].repeat(self.window, *([1] * (obs.ndim - 1)))
            values = [obs_win, act, goal_win]
        else:
            values = [obs_win, act, *repeated_others]
        if self.transform is not None:
            values = self.transform(values)
        return tuple(values)


# builders for the 3 data-source modes
def make_raw_policy_traj(original_traj_datasets=None, planned_traj_datasets=None,
                         data_source="original"):
    """Wrap raw TrajDatasets as a single TrajectoryDataset for the chosen data
    source (original | planned | both)."""
    original_traj_datasets = original_traj_datasets or []
    planned_traj_datasets = planned_traj_datasets or []
    if data_source == "original":
        raw = [DinoWMTrajForPolicy(d) for d in original_traj_datasets]
    elif data_source == "planned":
        raw = [DinoWMTrajForPolicy(d) for d in planned_traj_datasets]
    elif data_source == "both":
        raw = [DinoWMTrajForPolicy(d) for d in (list(original_traj_datasets) + list(planned_traj_datasets))]
    else:
        raise ValueError(f"Unknown data_source: {data_source!r}")
    if len(raw) == 0:
        raise ValueError(f"No trajectory datasets provided for data_source={data_source!r}")
    return raw[0] if len(raw) == 1 else ConcatTrajForPolicy(raw)


def build_policy_datasets(
    encoder,
    original_traj_datasets: Optional[List] = None,
    planned_traj_datasets: Optional[List] = None,
    data_source: str = "original",
    window_size: int = 5,
    action_window_size: int = 5,
    vqbet_get_future_action_chunk: bool = False,
    train_fraction: float = 0.95,
    random_seed: int = 42,
    embed_device: str = "cpu",
    goal_conditional: bool = False,
):
    """Assemble train/val sliced policy datasets for one of the 3 data modes.

    data_source:
        "original"        -> original demonstrations only
        "planned"         -> WM-planned trajectories only
        "both"            -> original + planned (ConcatTrajForPolicy)

    *_traj_datasets are lists of raw dino_wm TrajDatasets constructed WITHOUT the
    [-1, 1] transform (so visuals are in [0, 1] for the patch encoder).
    Returns (train_slices, val_slices).
    """
    shim = make_raw_policy_traj(original_traj_datasets, planned_traj_datasets, data_source)
    train_raw, val_raw = split_traj_datasets(shim, train_fraction, random_seed)

    train_emb = TrajectoryEmbeddingDataset(encoder, train_raw, device=embed_device)
    val_emb = TrajectoryEmbeddingDataset(encoder, val_raw, device=embed_device)

    slicer_kwargs = dict(
        window=window_size,
        action_window=action_window_size,
        vqbet_get_future_action_chunk=vqbet_get_future_action_chunk,
        goal_conditional=goal_conditional,
    )
    train_slices = VqbetTrajectorySlicerDataset(train_emb, **slicer_kwargs)
    val_slices = VqbetTrajectorySlicerDataset(val_emb, **slicer_kwargs)
    return train_slices, val_slices
