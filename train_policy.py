"""
Train a behavior-cloning policy (VQ-BeT or Diffusion) on a frozen patch encoder,
over one of three data sources:

data_source=original  -> original demonstrations only
data_source=planned  -> WM-planned trajectories only (generate_planned_trajectories.py)
data_source=both  -> original + planned
accelerate launch train_policy.py policy=vqbet  data_source=original
"""
import os
# MuJoCo defaults to the GLFW backend, which needs an X display and fails on a headless box.
os.environ.setdefault("MUJOCO_GL", "egl")
# Set before any CUDA init so torch.use_deterministic_algorithms() can make cuBLAS matmuls
# deterministic when deterministic=true, harmless otherwise, respects a shell-set value.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import gym
import tqdm
import hydra
import torch
import random
import traceback
import numpy as np
from pathlib import Path
from datetime import timedelta
from collections import deque
from einops import rearrange
from omegaconf import OmegaConf
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs

from utils import seed as set_seed
from env.venv import SubprocVectorEnv

from datasets.policy_dataset import (
    make_raw_policy_traj,
    split_traj_datasets,
    TrajectoryEmbeddingDataset,
    VqbetTrajectorySlicerDataset,
)

# allow ${eval:'...'} in configs, used by the diffusion policy's pred_horizon
OmegaConf.register_new_resolver("eval", eval, replace=True)

# Run wandb's sync in a thread, not a separate service process (patch_policy does this).
os.environ.setdefault("WANDB_START_METHOD", "thread")

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False


# parameter-count helpers
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_params(n):
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{n}"
        n /= 1000.0
    return f"{n:.1f}T"


def _frame_to_uint8(frame):
    """A single (H, W, C) frame (tensor or array, [0,1] or [0,255]) -> uint8 [0,255]."""
    f = np.asarray(frame, dtype=np.float32)
    if f.max() <= 1.5:
        f = f * 255.0
    return np.clip(f, 0, 255).astype(np.uint8)


# build raw trajectory datasets
def _seed_worker(worker_id):
    """Make each DataLoader worker's numpy/random RNG deterministic (seeded from the worker's
    torch seed = base_seed + worker_id), so a stochastic __getitem__ (e.g. future-goal sampling)
    is reproducible under deterministic mode. Module-level so it's picklable for the workers."""
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def build_raw_dino_traj_datasets(dataset_cfg, num_hist, num_pred, frameskip):
    """Instantiate an env's trajectory datasets and return the train split only."""
    _, traj_dset = hydra.utils.call(
        dataset_cfg, num_hist=num_hist, num_pred=num_pred, frameskip=frameskip
    )
    key = "train" if traj_dset.get("train") is not None else next(iter(traj_dset))
    return [traj_dset[key]]


def _cap_traj_dataset(dset, cap):
    """Keep only the first `cap` trajectories (deterministic, None = all). Mirrors
    planned_max_trajectories, lets you train/eval on a fixed small subset (e.g. a single
    trajectory) so BC-on-original and generate-from-the-same-source line up by index.
    Raises if `cap` exceeds the available trajectories (asking for more than exist is a mistake,
    not a silent "use all")."""
    if cap is None:
        return dset
    cap = int(cap)
    if cap > len(dset):
        raise ValueError(
            f"original_max_trajectories={cap} > {len(dset)} available original trajectories. "
            f"Lower it to <= {len(dset)}, or use more data.")
    if cap == len(dset):
        return dset
    from datasets.policy_dataset import TrajectorySubset
    return TrajectorySubset(dset, list(range(cap)))


def _select_original(dset, cfg):
    """Restrict the ORIGINAL trajectory dataset for BC/eval. Two modes:
      - segment mode (original_segment_length or original_subset_seed set): a seeded-random set
        of `original_max_trajectories` windows of `original_segment_length` steps, via the shared
        select_original_segments, so BC-on-original trains on exactly the windows generation
        planned from (same original_subset_seed + same source pool). segment_length=null uses
        each picked trajectory's full length.
      - cap mode (both null): the plain first-N whole-trajectory cap (backward compatible)."""
    seg_len = cfg.get("original_segment_length", None)
    seed = cfg.get("original_subset_seed", None)
    n = cfg.get("original_max_trajectories", None)
    if seg_len is None and seed is None:
        return _cap_traj_dataset(dset, n)
    from datasets.policy_dataset import select_original_segments, TrajectorySegmentView
    segments = select_original_segments(
        dset, max_trajectories=n, segment_length=seg_len, subset_seed=seed)
    return TrajectorySegmentView(dset, segments)


class _RawTrajConcat:
    """Concatenate raw trajectory datasets, routing get_frames/get_seq_length by global
    index. Lets the eval reproduce the BC train split across original+planned with one
    index space (matching make_raw_policy_traj's original-then-planned concat order)."""

    def __init__(self, dsets):
        self.dsets = list(dsets)
        self._cum = [0]
        for d in self.dsets:
            self._cum.append(self._cum[-1] + len(d))

    def __len__(self):
        return self._cum[-1]

    def _route(self, idx):
        for k in range(len(self.dsets)):
            if idx < self._cum[k + 1]:
                return self.dsets[k], idx - self._cum[k]
        raise IndexError(idx)

    def get_seq_length(self, idx):
        d, j = self._route(idx)
        return d.get_seq_length(j)

    def get_frames(self, idx, frames):
        d, j = self._route(idx)
        return d.get_frames(j, frames)


def _bc_train_perm(n_total, train_fraction, seed):
    """The exact trajectory indices BC used as its train split: shuffle [0, n_total)
    with the fixed-seed generator and take the first int(train_fraction*n_total).
    Mirrors datasets.policy_dataset.split_traj_datasets."""
    perm = torch.randperm(
        n_total, generator=torch.Generator().manual_seed(int(seed))).tolist()
    return perm[:int(train_fraction * n_total)]


def _fix_vqvae_keys(state):
    """Repair malformed VQ-BeT VQVAE state_dict keys before load. The VQVAE's custom
    state_dict writes submodule prefixes without a dot (e.g. 'encoderencoder.0.weight',
    'vq_embeddinglayers...'), which don't match the default load keys ('encoder.encoder',
    'vq_layer.layers'). Fix those prefixes so a saved checkpoint loads. No-op on
    already-correct keys and on non-VQVAE keys (GPT/prior)."""
    repl = [
        ("_vqvae_model.encoderencoder",     "_vqvae_model.encoder.encoder"),
        ("_vqvae_model.encoderfc",          "_vqvae_model.encoder.fc"),
        ("_vqvae_model.decoderencoder",     "_vqvae_model.decoder.encoder"),
        ("_vqvae_model.decoderfc",          "_vqvae_model.decoder.fc"),
        ("_vqvae_model.vq_embeddinglayers", "_vqvae_model.vq_layer.layers"),
    ]
    fixed = {}
    for k, v in state.items():
        for a, b in repl:
            if k.startswith(a):
                k = b + k[len(a):]
                break
        fixed[k] = v
    return fixed


def _build_eval_traj_dataset(cfg, traj_dset, goal_source, pass_type,
                             coverage_task=False, external=False):
    """The trajectory dataset an eval pass samples init/goal states from. `goal_source` here is the
    normalized base split ('train' or 'valid'), the caller maps eval_train/eval_valid -> external
    train/valid.

    external=True -> traj_dset is a separate eval-only dataset (eval_task_dataset), not the one BC
    valid -> the held-out 'valid' split (full episodes).
    train -> the exact trajectories BC trained on: reproduce the 95/5 split over the
    """
    if external:
        if goal_source == "train" and traj_dset.get("train") is not None:
            return traj_dset["train"], "task-dataset/train"
        return traj_dset["valid"], "task-dataset/valid"
    if goal_source == "valid":
        return traj_dset["valid"], "valid"

    from datasets.policy_dataset import TrajectorySubset
    orig_key = "train" if traj_dset.get("train") is not None else next(iter(traj_dset))
    # same original selection BC training used, so goal_source=train evals the same windows
    orig_dset = _select_original(traj_dset[orig_key], cfg)
    n_orig = len(orig_dset) if cfg.data_source in ("original", "both") else 0
    planned_dset, n_planned = None, 0
    if cfg.data_source in ("planned", "both"):
        from datasets.planned_dset import PlannedTrajDataset
        from datasets.img_transforms import default_transform
        # same selection/seed as training below, or goal_source=train would replay a different
        # subset than BC actually trained on.
        planned_dset = PlannedTrajDataset(
            cfg.planned_data_path, transform=default_transform(cfg.img_size),
            only_successes=bool(cfg.get("planned_only_successes", False)),
            max_trajectories=cfg.get("planned_max_trajectories", None),
            selection=str(cfg.get("planned_selection", "first")),
            selection_seed=int(cfg.get("planned_selection_seed", cfg.seed)))
        n_planned = len(planned_dset)
    perm = _bc_train_perm(n_orig + n_planned, cfg.train_fraction, cfg.seed)

    if coverage_task:
        # coverage needs full original episodes, keep only original-block indices
        if n_orig == 0:
            raise ValueError("coverage task pass with goal_source=train needs original data.")
        sel = [i for i in perm if i < n_orig]
        if not sel:
            raise ValueError("goal_source=train selected 0 original trajectories.")
        return TrajectorySubset(orig_dset, sel), f"train/original ({len(sel)})"

    # goal/all pass, use the whole trained-on set (original+planned), frame-0 inits
    if cfg.data_source == "original":
        base = orig_dset
    elif cfg.data_source == "planned":
        base = planned_dset
    else:   # both
        base = _RawTrajConcat([orig_dset, planned_dset])
    if not perm:
        raise ValueError("goal_source=train selected 0 trajectories.")
    return TrajectorySubset(base, list(perm)), f"train ({len(perm)})"


def _resolve_task_dataset_spec(cfg):
    """Resolve cfg.eval_task_dataset into a callable dataset spec (or None = disabled).

    - null  -> None (task passes reuse the main env dataset)
    - a dataset-group NAME -> conf/dataset/<name>.yaml (e.g. 'pusht_noise'), so you can write
    - a full inline spec  -> used as-is (a DictConfig with _target_)
    """
    spec = cfg.get("eval_task_dataset", None)
    if spec is None:
        return None
    if isinstance(spec, str):
        # conf/ sits next to train_policy.py, so this is robust regardless of Hydra's chdir.
        path = Path(__file__).resolve().parent / "conf" / "dataset" / f"{spec}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"eval_task_dataset='{spec}' -> {path} not found. Use a name from conf/dataset/ "
                f"(pusht | pusht_noise | wall), null, or a full inline dataset spec.")
        loaded = OmegaConf.load(path)
        # give the group config a root that carries the values its ${...} interpolations reference
        holder = OmegaConf.create({
            "normalize_action": cfg.normalize_action,
            "img_size": cfg.img_size,
            "env_vars": cfg.env_vars,
            "ds": loaded,
        })
        return holder.ds
    # inline DictConfig spec, disabled unless it actually has a _target_
    return spec if spec.get("_target_") is not None else None


def _warn_env_dataset_mismatch(cfg):
    """env and dataset are chosen independently (dataset config group), so warn on an obvious
    mismatch (e.g. env=wall with a pusht dataset) instead of failing deep in the loader."""
    target = str(OmegaConf.select(cfg, "env.dataset._target_") or "")
    name = str(cfg.env.name)
    if name == "wall" and "wall" not in target:
        print(f"[WARN] env=wall but dataset loader is {target!r} -- set dataset=wall?")
    elif name == "pusht" and "pusht" not in target:
        print(f"[WARN] env=pusht but dataset loader is {target!r} -- set dataset=pusht or pusht_noise?")
    elif name == "puzzle" and "puzzle" not in target:
        print(f"[WARN] env=puzzle but dataset loader is {target!r} -- set dataset=puzzle?")
    elif name == "point_maze" and "point_maze" not in target:
        print(f"[WARN] env=point_maze but dataset loader is {target!r} -- set dataset=point_maze?")


def main(cfg):
    _warn_env_dataset_mismatch(cfg)
    # accelerate / logging
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
    dist_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # wandb is driven directly via wandb.init/log below, not accelerate's tracker API so don't
    # also register it with Accelerator, redundant and double-touches the service.
    accelerator = Accelerator(
        kwargs_handlers=[process_group_kwargs, dist_kwargs],
    )
    device = accelerator.device
    set_seed(cfg.seed)
    # Opt-in full determinism (deterministic=true), pin cuDNN + force deterministic algorithms +
    # seed the DataLoader, so two same-seed runs train to the same model and, since the env seeds
    # and the VQ-BeT action sampling both run off the now-reproducible RNG, produce the same eval
    # numbers.
    deterministic = bool(cfg.get("deterministic", False))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        if accelerator.is_main_process:
            print("[determinism] ON: cuDNN deterministic + use_deterministic_algorithms(warn_only) "
                  "+ seeded DataLoader. Slower; same-seed runs should now match closely.")

    use_diffusion = "diffusion" in cfg.policy["_target_"]
    gpu_batch_size = max(1, cfg.batch_size // accelerator.num_processes)

    # validate env-eval passes up front, fail before any training.
    #   goal_source, 'valid' (original held-out split) | 'train' (the exact trajectories
    #   type, 'goal', success/distance vs a goal_offset-ahead goal | 'task' (coverage vs
    eval_goal_source = str(cfg.get("eval_goal_source", "valid"))
    # (type, resolved goal_source) for every pass, the no-passes fallback is type 'all'.
    _pass_specs = [(str(_p.get("type", "all")),
                    (eval_goal_source if _p.get("goal_source", "cfg") == "cfg"
                     else str(_p.get("goal_source"))))
                   for _p in (cfg.get("eval_passes") or [])] or [("all", eval_goal_source)]
    _has_original = cfg.data_source in ("original", "both")
    _has_planned = cfg.data_source in ("planned", "both")
    _has_task_dataset = cfg.get("eval_task_dataset", None) is not None
    _env_has_coverage = bool(cfg.env.get("has_coverage", str(cfg.env.name) == "pusht"))
    # goal_source, train/valid = the main (trained-on) dataset, eval_train/eval_valid = the
    # separate eval_task_dataset's own splits (needs it set), random_state = env-sampled (no demo
    # data).
    _VALID_SOURCES = ("valid", "train", "eval_train", "eval_valid", "random_state")
    for _type, gs in _pass_specs:
        if gs == "random_action":
            raise ValueError(
                "goal_source=random_action is a generation-only DATA mode (random-action rollouts "
                "with no goal to reach); there is nothing for a trained policy to be scored against, "
                "so it is not a valid eval goal_source. Use " + " / ".join(_VALID_SOURCES) + ".")
        if gs not in _VALID_SOURCES:
            raise ValueError(f"goal_source must be one of {_VALID_SOURCES}, got {gs!r}")
        if _type not in ("goal", "task", "all"):
            raise ValueError(f"eval pass type must be 'goal' or 'task', got {_type!r}")
        if gs in ("eval_train", "eval_valid") and not _has_task_dataset:
            raise ValueError(f"goal_source={gs} requires eval_task_dataset to be set.")
        _coverage_task = (_type == "task" and _env_has_coverage and gs not in ("random_state",))
        if gs == "train":   # main trained-on split only
            if _has_planned and cfg.get("planned_data_path") is None:
                raise ValueError("goal_source=train with planned/both data requires planned_data_path.")
            # coverage needs full original episodes (only for coverage envs, e.g.
            if _coverage_task and not _has_original:
                raise ValueError(
                    "a coverage task pass with goal_source=train requires original data "
                    "(data_source in original/both): planned trajectories are short and "
                    "are not full task episodes. Use goal_source=valid/eval_valid for task on "
                    "planned-only runs.")

    # goal-conditioned BC, goal = a goal-image embedding (dim = encoder.output_dim) stacked with
    # obs inside the policy.
    goal_conditional = bool(cfg.get("goal_conditional", False))
    cfg.env.goal_dim = int(cfg.encoder.output_dim) if goal_conditional else 0

    if accelerator.is_main_process:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        print(f"Saving to {os.getcwd()}")

    # Optionally start from a prior train_policy.py checkpoint (conf, checkpoint, resume).
    #   resume, true  -> RESTART the same run, load model + optimizer + epoch and continue
    #   resume, false -> FINETUNE, load model WEIGHTS only, fresh optimizer, fresh wandb run
    ckpt_path = cfg.get("checkpoint", None)
    resume = bool(cfg.get("resume", False))
    ckpt_payload = torch.load(ckpt_path, map_location="cpu", weights_only=False) if ckpt_path else None
    if ckpt_payload is not None:
        print(f"loaded checkpoint: {ckpt_path} (resume={resume})")
    resume_epoch, resume_run_id = 0, None
    if resume and isinstance(ckpt_payload, dict):
        resume_epoch = int(ckpt_payload.get("epoch", -1)) + 1
        resume_run_id = ckpt_payload.get("wandb_run_id")

    # Create the closed-loop eval env once (main process), before wandb starts its background
    # threads, then reuse it for every eval.
    eval_env = make_eval_env(cfg, cfg.n_envs) if (cfg.eval_on_env and accelerator.is_main_process) else None
    # Load the eval trajectory dataset once and reuse it across all passes/epochs, instead of
    # re-reading the whole dataset from disk every eval.
    eval_traj_dset = None
    # Optional separate dataset that 'task' passes draw init/goal from (eval_task_dataset).
    eval_task_traj_dset = None
    if cfg.eval_on_env and accelerator.is_main_process:
        _, eval_traj_dset = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist,
                                             num_pred=cfg.num_pred, frameskip=cfg.frameskip)
        _task_spec = _resolve_task_dataset_spec(cfg)
        if _task_spec is not None:
            _, eval_task_traj_dset = hydra.utils.call(
                _task_spec, num_hist=cfg.num_hist,
                num_pred=cfg.num_pred, frameskip=cfg.frameskip)
            print(f"[eval] loaded a separate eval_task_dataset "
                  f"({OmegaConf.select(_task_spec, 'data_path')}) for 'task' passes "
                  f"(goal passes still use the main env dataset).")

    do_wandb = accelerator.is_main_process and _HAS_WANDB and cfg.get("wandb_logging", True)
    if do_wandb:
        # name the wandb run after the (descriptive) Hydra run folder explicit
        # name/group/job_type/tags when the config supplies them (dagger_loop does) else the Hydra
        # run folder.
        init_kwargs = dict(project=cfg.wandb.project, entity=cfg.wandb.entity,
                           name=cfg.wandb.get("name") or os.path.basename(os.getcwd()),
                           group=cfg.wandb.get("group") or None,
                           job_type=cfg.wandb.get("job_type") or None,
                           tags=list(cfg.wandb.get("tags") or []) or None,
                           config=OmegaConf.to_container(cfg, resolve=True))
        if resume and resume_run_id:
            init_kwargs.update(id=resume_run_id, resume="allow")   # continue the same run
        wandb.init(**init_kwargs)

    # frozen patch encoder, moved to device, not DDP-wrapped, it's only used for inference,
    # including per-batch encoding in the lazy path
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # raw dataset for the chosen data source
    original = (build_raw_dino_traj_datasets(cfg.env.dataset, cfg.num_hist, cfg.num_pred, cfg.frameskip)
                if cfg.data_source in ("original", "both") else [])
    # restrict original data, seeded-random segments (original_segment_length/_subset_seed) or the
    # plain first-N cap.
    if original:
        original = [_select_original(original[0], cfg)]
    planned = []
    if cfg.data_source in ("planned", "both"):
        assert cfg.planned_data_path is not None, "planned_data_path required for planned/both"
        from datasets.planned_dset import PlannedTrajDataset
        from datasets.img_transforms import default_transform
        # default_transform resizes to img_size and keeps [0,1]
        planned = [PlannedTrajDataset(
            cfg.planned_data_path,
            transform=default_transform(cfg.img_size),
            only_successes=bool(cfg.get("planned_only_successes", False)),
            max_trajectories=cfg.get("planned_max_trajectories", None),
            # which capped subset to load when the pool holds more than the cap, first | recent
            # (chronological queue) | random | random_with_replacement.
            selection=str(cfg.get("planned_selection", "first")),
            selection_seed=int(cfg.get("planned_selection_seed", cfg.seed)),
        )]
    # fail fast if the configured action dim doesn't match the actual data
    ref = (original or planned)[0]
    ref_base = getattr(ref, "dataset", ref)
    data_act_dim = getattr(ref_base, "action_dim", None)
    if data_act_dim is not None and int(cfg.env.act_dim) != int(data_act_dim):
        raise ValueError(
            f"cfg.env.act_dim={cfg.env.act_dim} but the dataset's action_dim={data_act_dim}. "
            f"Set act_dim correctly in conf/env/{cfg.env.name}.yaml."
        )

    dataset = make_raw_policy_traj(original, planned, cfg.data_source)

    # how much data this run is actually training on.
    _n_orig = len(original[0]) if original else 0
    _n_plan = len(planned[0]) if planned else 0
    data_counts = {
        "data/original_trajectories": _n_orig,
        "data/planned_trajectories": _n_plan,
        "data/total_trajectories": len(dataset),
    }
    print(f"[data] source={cfg.data_source}: {_n_orig} original + {_n_plan} planned "
          f"= {len(dataset)} trajectories")

    # policy + optimizer
    cbet_model = hydra.utils.instantiate(cfg.policy).to(device)
    if ckpt_payload is not None:   # load pretrained policy weights (resume or finetune)
        _state = (ckpt_payload["model"] if isinstance(ckpt_payload, dict) and "model" in ckpt_payload
                  else ckpt_payload.state_dict() if hasattr(ckpt_payload, "state_dict") else ckpt_payload)
        _state = _fix_vqvae_keys(_state)   # repair VQVAE's malformed state_dict prefixes
        cbet_model.load_state_dict(_state)
        print("loaded policy weights from checkpoint")
    optimizer = cbet_model.configure_optimizers(
        weight_decay=cfg.optim.weight_decay,
        learning_rate=cfg.optim.lr,
        betas=tuple(cfg.optim.betas),
    )
    if use_diffusion:
        from utils.normalizer import LinearNormalizer
        action_normalizer = LinearNormalizer()
        action_normalizer.fit(dataset.get_all_actions())
        cbet_model.set_normalizer(action_normalizer)

    # parameter counts
    if accelerator.is_main_process:
        enc_total, enc_train = count_parameters(encoder)
        mdl_total, mdl_train = count_parameters(cbet_model)
        print("\n" + "=" * 60 + "\nMODEL PARAMETER COUNTS\n" + "=" * 60)
        print(f"\nEncoder:\n  Total parameters:     {format_params(enc_total):>10} ({enc_total:,})")
        print(f"  Trainable parameters: {format_params(enc_train):>10} ({enc_train:,})")
        print(f"\nPolicy Model (cbet_model):\n  Total parameters:     {format_params(mdl_total):>10} ({mdl_total:,})")
        print(f"  Trainable parameters: {format_params(mdl_train):>10} ({mdl_train:,})")
        print(f"\nCombined (Encoder + Policy):\n  Total parameters:     {format_params(enc_total + mdl_total):>10} ({enc_total + mdl_total:,})")
        print(f"  Trainable parameters: {format_params(enc_train + mdl_train):>10} ({enc_train + mdl_train:,})")

    # split, (optionally precompute embeddings), slice, build loaders
    precompute = bool(cfg.get("precompute_embeddings", True))
    train_data, test_data = split_traj_datasets(
        dataset, train_fraction=cfg.train_fraction, random_seed=cfg.seed
    )
    # with a tiny train set, e.g. a single segment the held-out split can be empty, skip the whole
    # validation path rather than crash on an empty dataset (TrajectoryEmbeddingDataset indexes
    # self.data[0]).
    have_test = len(test_data) > 0
    if not have_test:
        print(f"[val] test split is empty (train_fraction={cfg.train_fraction}, "
              f"{len(dataset)} trajectories), skipping the validation action-loss.")
    data_counts["data/train_trajectories"] = len(train_data)
    data_counts["data/valid_trajectories"] = len(test_data)
    print(f"[data] split: {len(train_data)} train / {len(test_data)} valid "
          f"(train_fraction={cfg.train_fraction})")
    if do_wandb:
        # summary puts them in the runs table as sortable columns, which is where rounds are
        # compared.
        wandb.run.summary.update(data_counts)
        wandb.log(data_counts)
    if precompute:
        train_data = TrajectoryEmbeddingDataset(encoder, train_data, device=cfg.embed_device)
        if have_test:
            test_data = TrajectoryEmbeddingDataset(encoder, test_data, device=cfg.embed_device)
    future_conditional = bool(cfg.get("future_conditional", False))
    if future_conditional and not goal_conditional:
        raise ValueError("future_conditional=true requires goal_conditional=true")
    slicer_kwargs = dict(window=cfg.window_size, action_window=cfg.action_window_size,
                         vqbet_get_future_action_chunk=False, goal_conditional=goal_conditional,
                         future_conditional=future_conditional,
                         min_future_sep=int(cfg.get("min_future_sep", cfg.action_window_size)),
                         future_seq_len=int(cfg.get("future_seq_len", 1)),
                         only_sample_tail=bool(cfg.get("only_sample_tail", False)))
    train_data = VqbetTrajectorySlicerDataset(train_data, **slicer_kwargs)
    if have_test:
        test_data = VqbetTrajectorySlicerDataset(test_data, **slicer_kwargs)
    # catch a train set too short to yield any windows, else training silently does nothing
    if len(train_data) == 0:
        raise ValueError(
            f"0 training windows after slicing (window={cfg.window_size}, "
            f"min_future_sep={int(cfg.get('min_future_sep', cfg.action_window_size))}). The "
            f"segment/trajectory is too short -- raise original_segment_length or shrink the window.")

    # deterministic mode, fix the shuffle order (generator) and per-worker RNG (worker_init_fn) so
    # the data pipeline is reproducible too.
    loader_gen = torch.Generator().manual_seed(int(cfg.seed)) if deterministic else None
    seed_worker = _seed_worker if deterministic else None
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=gpu_batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=False, drop_last=True,
        worker_init_fn=seed_worker, generator=loader_gen)
    test_loader = (torch.utils.data.DataLoader(
        test_data, batch_size=gpu_batch_size, shuffle=False, num_workers=cfg.num_workers,
        worker_init_fn=seed_worker)
        if have_test else None)

    if have_test:
        cbet_model, optimizer, train_loader, test_loader = accelerator.prepare(
            cbet_model, optimizer, train_loader, test_loader)
    else:
        cbet_model, optimizer, train_loader = accelerator.prepare(
            cbet_model, optimizer, train_loader)
    # resume (same data), restore optimizer state so momentum/LR schedule continue.
    if resume and isinstance(ckpt_payload, dict) and ckpt_payload.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt_payload["optimizer"])
        # load_state_dict keeps the saved (CPU) state tensors, move them to the param device so
        # Adam's step doesn't mismatch GPU params against CPU moments.
        base_opt = getattr(optimizer, "optimizer", optimizer)   # unwrap AcceleratedOptimizer
        for st in base_opt.state.values():
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    st[k] = v.to(device)
        print(f"resumed optimizer state; continuing from epoch {resume_epoch}")
    elif resume and ckpt_payload is not None:
        print("WARNING: resume=true but the checkpoint has no optimizer/epoch (old format) -- "
              "loaded weights only, starting a fresh optimizer at epoch 0.")

    goal_dim = int(cfg.env.goal_dim)
    # the goal window is a single repeated frame, final-frame goal, or a length-1 future goal
    # unless future conditioning samples a multi-frame clip.
    goal_is_uniform = not (future_conditional and int(cfg.get("future_seq_len", 1)) > 1)

    def run_model(batch):
        # precompute, batch[0]/[2] are embeddings (N,T,V,P,E).
        obs, act = batch[0].to(device), batch[1].to(device)
        goal = batch[2].to(device) if goal_dim > 0 else None
        if not precompute:
            with torch.no_grad():
                obs = encoder(obs)
                if goal is not None:
                    if goal_is_uniform:
                        # all window frames are the same goal, encode once
                        gw = goal.shape[1]
                        goal = encoder(goal[:, :1]).expand(-1, gw, -1, -1, -1)
                    else:
                        # distinct future-clip frames, encode the whole window
                        goal = encoder(goal)
        obs = rearrange(obs, "N T V P E -> N T (V P) E")
        if goal is not None:
            goal = rearrange(goal, "N T V P E -> N T (V P) E")
        return cbet_model(obs, goal, act)

    out_dir = Path(os.getcwd())
    ckpt_dir = out_dir / "checkpoints"   # keep model_*.pt out of the run dir's top level

    def save_ckpt(name, epoch):
        # checkpoint = model weights + optimizer state + epoch + wandb run id, so it can be
        # resumed, conf, checkpoint + resume OR loaded weights-only for finetuning.
        ckpt_dir.mkdir(exist_ok=True)
        torch.save({
            "model": accelerator.unwrap_model(cbet_model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "wandb_run_id": (wandb.run.id if do_wandb and wandb.run is not None else None),
        }, ckpt_dir / name)

    metrics_history = []
    for epoch in tqdm.trange(resume_epoch, cfg.epochs, disable=not accelerator.is_main_process):
        # closed-loop env eval (main process only). The other ranks resync at the barrier below so
        # DDP all-reduce in the train step doesn't desync.
        if cfg.eval_on_env and (epoch == resume_epoch or epoch % cfg.eval_on_env_freq == 0):
            if accelerator.is_main_process:
                try:
                    metrics = run_eval_passes(cfg, encoder,
                                          accelerator.unwrap_model(cbet_model), device, use_diffusion,
                                          epoch=epoch, out_dir=out_dir, env=eval_env,
                                          traj_dset=eval_traj_dset, task_traj_dset=eval_task_traj_dset)
                    print(f"eval_on_env: {metrics}")
                    metrics_history.append(metrics)
                    if do_wandb:
                        wandb.log({**{f"env/{k}": v for k, v in metrics.items()}, "epoch": epoch})
                except Exception as e:
                    print(f"eval_on_env skipped: {e}")
                    traceback.print_exc()
            accelerator.wait_for_everyone()

        # validation loss + action_diff metrics, skipped when the held-out split is empty
        if test_loader is not None and epoch % cfg.eval_freq == 0:
            cbet_model.eval()
            total_loss, n = 0.0, 0
            ad = {k: 0.0 for k in ("action_diff", "action_diff_tot",
                                   "action_diff_mean_res1", "action_diff_mean_res2", "action_diff_max")}
            with torch.no_grad():
                for batch in test_loader:
                    _, loss, loss_dict = run_model(batch)
                    if loss is None:
                        continue
                    total_loss += loss.item(); n += 1
                    if do_wandb:
                        wandb.log({**{f"eval/{x}": y for x, y in loss_dict.items()}, "epoch": epoch})
                    if not use_diffusion:
                        for k in ad:
                            ad[k] += loss_dict.get(k, 0.0)
            if accelerator.is_main_process:
                print(f"Test loss: {total_loss / max(1, n)}")
                if do_wandb and not use_diffusion:
                    wandb.log({**{f"eval/epoch_wise_{k}": v for k, v in ad.items()}, "epoch": epoch})

        # save before this epoch's training, so model_{epoch}.pt is the exact model that produced
        # the epoch-{epoch} metrics logged above, env eval + val loss.
        if epoch % cfg.save_every == 0 and accelerator.is_main_process:
            save_ckpt(f"model_{epoch}.pt", epoch)

        # train
        cbet_model.train()
        train_loss = 0.0
        for batch in tqdm.tqdm(train_loader, disable=not accelerator.is_main_process):
            optimizer.zero_grad()
            _, loss, loss_dict = run_model(batch)
            train_loss += loss.item()
            accelerator.backward(loss)
            optimizer.step()
            if use_diffusion:
                if hasattr(cbet_model, "module"):
                    accelerator.unwrap_model(cbet_model).ema_step()
                else:
                    cbet_model.ema_step()
            if do_wandb:
                wandb.log({**{f"train/{x}": y for x, y in loss_dict.items()}, "epoch": epoch})

        # VQ-BeT fits its codebook at epoch boundary
        raw_model = accelerator.unwrap_model(cbet_model)
        if hasattr(raw_model, "finish_epoch"):
            raw_model.finish_epoch()
        if accelerator.is_main_process:
            print(f"Train loss: {train_loss / max(1, len(train_loader))}")

    # final closed-loop eval + best-metric report (mirrors online_eval.py's tail)
    if cfg.eval_on_env:
        if accelerator.is_main_process:
            try:
                metrics = run_eval_passes(cfg, encoder, accelerator.unwrap_model(cbet_model),
                                      device, use_diffusion, epoch=cfg.epochs, out_dir=out_dir,
                                      env=eval_env, traj_dset=eval_traj_dset,
                                      task_traj_dset=eval_task_traj_dset)
                print(f"final eval_on_env: {metrics}")
                metrics_history.append(metrics)
                if do_wandb:
                    wandb.log({**{f"env/{k}": v for k, v in metrics.items()}, "epoch": cfg.epochs})
                # best across all evals per pass, keys are "<pass>/<metric>", matched by suffix.
                seen_keys = {k for m in metrics_history for k in m}
                best_specs = [(s, max) for s in ("final_coverage_mean", "max_coverage_mean",
                                                 "final_success_rate", "max_success_rate")]
                best_specs += [(s, min) for s in ("final_goal_dist_mean", "closest_goal_dist_mean")]
                for suffix, agg in best_specs:
                    for key in sorted(k for k in seen_keys if k.endswith(suffix)):
                        vals = [m[key] for m in metrics_history if key in m]
                        if vals:
                            best = agg(vals)
                            print(f"best {key} over training: {best}")
                            if do_wandb:
                                wandb.log({f"best/{key}": best, "epoch": cfg.epochs})
            except Exception as e:
                print(f"final eval_on_env skipped: {e}")
                traceback.print_exc()
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        save_ckpt("model_final.pt", cfg.epochs - 1)
        if eval_env is not None:   # tear down the reused eval env once, at the very end
            try:
                eval_env.close()
            except Exception:
                pass


def make_eval_env(cfg, n_envs):
    """Create the closed-loop eval vector env. Create this once and reuse it across all evals, forking a fresh SubprocVectorEnv every eval (from a main process with live
    wandb/CUDA/DataLoader threads) risks a sporadic fork-deadlock in a worker, after which the
    parent hangs on recv forever (GPU -> 0%). patch_policy creates it once for this reason."""
    # serial_env=true -> every env lives IN this process, no fork.
    serial_env = bool(cfg.get("serial_env", False))
    # Plain copies, under start_method=spawn the closure is cloudpickled to the child, so do not
    # capture the live OmegaConf cfg.
    _env_name = str(cfg.env.name)
    _env_args = list(cfg.env.args or [])
    _env_kwargs = OmegaConf.to_container(cfg.env.kwargs, resolve=True) if cfg.env.kwargs else {}
    # 'fork' (default) | 'spawn' | 'forkserver'.
    start_method = cfg.get("env_start_method", None)
    if serial_env or _env_name in ("wall", "deformable_env"):
        from env.serial_vector_env import SerialVectorEnv
        if serial_env and _env_name not in ("wall", "deformable_env"):
            print(f"[env] serial_env=true -> {n_envs} eval envs IN-PROCESS (no subprocesses).")
        return SerialVectorEnv(
            [gym.make(_env_name, *_env_args, **_env_kwargs) for _ in range(n_envs)])
    if start_method:
        print(f"[env] {n_envs} eval envs via start_method={start_method}")
    return SubprocVectorEnv(
        [lambda: gym.make(_env_name, *_env_args, **_env_kwargs) for _ in range(n_envs)],
        start_method=start_method)


# Closed-loop env eval, roll the policy out per step, report eval_state success plus coverage for
# envs that expose it (pusht).
@torch.no_grad()
def eval_on_env(cfg, encoder, policy, device, use_diffusion, epoch=0, out_dir=None,
                goal_offset="cfg", horizon=None, pass_name="", goal_source="cfg",
                pass_type="all", env=None, traj_dset=None, task_traj_dset=None):
    # goal_offset / horizon override cfg, so one config can run several eval passes.
    #   'goal' -> condition on the frame goal_offset ahead, log only success/distance.
    #   'task' -> condition on the FINAL frame. On a coverage env (has_coverage, e.g. pusht) log
    #   'all'  -> no filtering (the no-eval_passes fallback). pass_name tags videos/metrics.
    #   train / valid  -> the main dataset, what BC trained on. train = the exact
    #   eval_train / eval_valid -> the separate eval_task_dataset's own train / valid splits (the
    #   policy never trained on it -> raw loader splits, no BC/original/
    #   random_state  -> init/goal SAMPLED FROM the ENV (pusht = random agent+T-pose
    goal_source = (str(cfg.get("eval_goal_source", "valid"))
                   if goal_source == "cfg" else str(goal_source))
    rs_mode = (goal_source == "random_state")
    external = goal_source in ("eval_train", "eval_valid")   # read the eval_task_dataset
    if external and task_traj_dset is None:
        raise ValueError(f"goal_source={goal_source} requires eval_task_dataset to be set.")
    base_split = {"train": "train", "valid": "valid",
                  "eval_train": "train", "eval_valid": "valid"}.get(goal_source, "valid")
    src_traj_dset = task_traj_dset if external else traj_dset
    # does this env expose a built-in task goal + coverage?
    env_has_coverage = bool(cfg.env.get("has_coverage", str(cfg.env.name) == "pusht"))
    coverage_task = (pass_type == "task" and env_has_coverage and not rs_mode)
    goal_off = cfg.get("eval_goal_offset", None) if goal_offset == "cfg" else goal_offset
    if pass_type == "task" or rs_mode:
        goal_off = None   # task/random_state condition on a fixed goal, not a frames-ahead one
    horizon = int(cfg.eval_horizon if horizon is None else horizon)
    # Where a demo-goal pass gets its goal PICTURE from.
    #   env  draw it with the same renderer that draws the observations
    #   dataset  read the stored frame, the old behaviour
    goal_render_mode = str(cfg.get("eval_goal_render", "env"))
    if goal_render_mode not in ("env", "dataset"):
        raise ValueError(f"eval_goal_render must be 'env' or 'dataset', got {goal_render_mode!r}")
    if pass_type == "goal" and not rs_mode:
        if goal_off is None:
            raise ValueError(f"goal pass '{pass_name}' needs a goal_offset (frames ahead).")
        if horizon < int(goal_off):
            raise ValueError(
                f"goal pass '{pass_name}': horizon={horizon} < goal_offset={int(goal_off)}, "
                f"too short to reach the goal. Set horizon >= goal_offset.")
    n_envs = cfg.n_envs
    assert cfg.n_env_evals % n_envs == 0, "n_env_evals must be a multiple of n_envs"
    save_video = bool(cfg.get("save_eval_video", True))
    n_videos = int(cfg.get("n_eval_videos", 3))
    video_fps = int(cfg.get("eval_video_fps", 12))
    video_dir = (Path(out_dir) if out_dir is not None else Path(os.getcwd())) / "eval_videos"

    # Action denormalization stats always come from the original dataset's valid split, the policy
    # emits actions in the original dino_wm normalization, and planned actions are stored in that
    # same space, so env stepping denormalizes with the original stats regardless of where the
    # goals come from.
    if traj_dset is None:
        _, traj_dset = hydra.utils.call(
            cfg.env.dataset, num_hist=cfg.num_hist,
            num_pred=cfg.num_pred, frameskip=cfg.frameskip)
    if not external:
        src_traj_dset = traj_dset   # , was None until the standalone load above
    valid_dset = traj_dset["valid"]   # always the main dataset -> policy's training action norm
    valid_base = getattr(valid_dset, "dataset", valid_dset)
    act_mean, act_std = valid_base.action_mean.numpy(), valid_base.action_std.numpy()

    # init/goal come from src_traj_dset, main, or the eval_task_dataset when external, at
    # base_split (train/valid).
    dset, dset_desc = _build_eval_traj_dataset(
        cfg, src_traj_dset, "valid" if rs_mode else base_split, pass_type,
        coverage_task=coverage_task, external=external)
    print(f"[{pass_name or 'eval'}] type={pass_type} "
          f"goal_source={'random_state' if rs_mode else goal_source} "
          f"dataset={'task' if external else 'main'} -> {dset_desc}"
          + ("" if rs_mode else f" goal_render={goal_render_mode}"))

    # reuse the caller's env if given, avoids per-eval forking, see make_eval_env otherwise create
    # one here and close it at the end (standalone use).
    own_env = env is None
    if own_env:
        env = make_eval_env(cfg, n_envs)

    from datasets.img_transforms import default_transform
    tf = default_transform(cfg.img_size)

    def embed(visual_np):   # dict visual, b, H, W, C ->, b, 1, P, E
        # ascontiguousarray handles deformable's BGR-reversed (negative-stride) frames
        v = torch.as_tensor(np.ascontiguousarray(visual_np)).float()
        if v.max() > 1.5:   # auto-detect [0,255] vs [0,1] across envs
            v = v / 255.0
        v = rearrange(v, "b h w c -> b c h w")
        v = tf(v).to(device)
        return encoder(v).reshape(v.shape[0], 1, encoder.n_patches, encoder.emb_dim)

    def step_env(a):   # denormalize policy action -> env step
        obs, rew, done, info = env.step(a * act_std + act_mean)
        return obs, np.asarray(rew), np.asarray(done), info

    # success, final-state vs ever-reached (max), distance, final vs closest-approach (min)
    successes, max_successes, state_dists, min_dists = [], [], [], []
    _ep_records = []   # per-episode rows, dumped to CSV at the end
    final_cov, max_cov, rewards = [], [], []
    n_batches = max(1, cfg.n_env_evals // n_envs)
    # A goal pass with random start offsets can OVERSAMPLE the valid split, re-using a trajectory
    # with a different start offset is a genuinely distinct (init, goal) pair, so cycling the
    # trajectories to reach the full n_env_evals is legitimate, and it smooths the choppy
    # small-sample success metric when the valid split has few trajectories.
    rand_start = goal_off is not None and bool(cfg.get("eval_goal_random_start", True))
    avail_batches = max(1, len(dset) // n_envs)
    # rs_mode samples init/goal from the env, so it doesn't draw from the dataset and is never
    # capped by len(dset), it always runs the full n_env_evals episodes.
    if not rand_start and not rs_mode and avail_batches < n_batches:
        print(f"[{pass_name or 'eval'}] only {len(dset)} eval trajectories < n_env_evals="
              f"{cfg.n_env_evals}; evaluating {avail_batches * n_envs} episodes instead")
        n_batches = avail_batches
    elif rand_start and avail_batches < n_batches:
        print(f"[{pass_name or 'eval'}] oversampling {len(dset)} valid trajectories to "
              f"{n_batches * n_envs} episodes via random start offsets")
    total_eps = n_batches * n_envs

    # pick which episodes to film, a random subset across all batches.
    #   eval_video_fixed=true  -> seed by cfg.seed only -> same episodes every eval
    #   eval_video_fixed=false -> seed by cfg.seed+epoch -> fresh random scenes each eval
    n_vid = min(int(n_videos), total_eps) if save_video else 0
    vid_seed = int(cfg.seed) if bool(cfg.get("eval_video_fixed", False)) else int(cfg.seed) + int(epoch)
    filmed_eps = (set(random.Random(vid_seed).sample(range(total_eps), n_vid))
                  if n_vid > 0 else set())

    # short-goal passes sample trajectories at random WITH REPLACEMENT, like
    # generate_planned_trajectories / dino-wm, restricted to trajectories long enough to hold a
    # goal_off-ahead goal (dino-wm's valid_ids filter).
    valid_ids = None
    if rand_start:
        valid_ids = [j for j in range(len(dset)) if dset.get_seq_length(j) >= int(goal_off) + 1]
        if not valid_ids:   # none long enough, fall back to all, goal_idx clamps to the end
            valid_ids = list(range(len(dset)))

    # wall, planned data doesn't store the maze layout, but wall_single uses a SINGLE fixed maze.
    wall_fixed_info = None
    if cfg.env.name == "wall" and src_traj_dset is not None:
        _ref = src_traj_dset.get("valid") or src_traj_dset.get("train")
        if _ref is not None:
            wall_fixed_info = _ref.get_frames(0, [0])[3]

    for b in range(n_batches):
        seeds = [cfg.seed + b * n_envs + i for i in range(n_envs)]
        # which (trajectory, start frame) each episode uses, task / final-frame goal (goal_off is
        # None), sequential frame-0 coverage, one pass over the distinct episodes, each run whole
        # toward task completion.
        idxs, init_off = [], []
        goal_render = None   # env-rendered goal image, b, H, W, C, rs_mode only
        if rs_mode:
            # random_state task, init/goal sampled from the env, on OPPOSITE sides of the wall.
            if wall_fixed_info is not None:
                env.update_env([wall_fixed_info] * n_envs)
            init_states, goal_states = env.sample_random_init_goal_states(seeds)
            init_states = np.asarray(init_states)
            goal_states = np.asarray(goal_states)
            env_infos = ([wall_fixed_info] * n_envs) if wall_fixed_info is not None else []
            # render the goal state once for goal-conditioning and the video, goal frame on the
            # right.
            goal_obs, _ = env.prepare(seeds, goal_states)
            goal_render = np.asarray(goal_obs["visual"])
            goal_emb = embed(goal_render) if int(cfg.env.goal_dim) > 0 else None
        else:
            for i in range(n_envs):
                if not rand_start:
                    idxs.append((b * n_envs + i) % len(dset))
                    init_off.append(0)
                else:
                    rng = random.Random(int(cfg.seed) + b * n_envs + i)   # per-episode, epoch-stable
                    j = rng.choice(valid_ids)   # random trajectory (repl.)
                    hi = max(0, dset.get_seq_length(j) - int(goal_off) - 1)   # room for the goal
                    idxs.append(j)
                    init_off.append(rng.randint(0, hi))
            # get_frames ->, obs, act, state, info, info ([3]) carries the per-trajectory env
            # layout for envs that have one (wall, door/wall location).
            _init_frames = [dset.get_frames(j, [init_off[k]]) for k, j in enumerate(idxs)]
            init_states = np.stack([f[2][0].numpy() for f in _init_frames])
            env_infos = [f[3] for f in _init_frames]
            # goal = state `goal_off` frames ahead of the (possibly offset) start, clamped to the
            # trajectory end, if None, the final frame.
            if goal_off is None:
                goal_idx = [dset.get_seq_length(j) - 1 for j in idxs]
            else:
                goal_idx = [min(init_off[k] + int(goal_off), dset.get_seq_length(j) - 1)
                            for k, j in enumerate(idxs)]
            goal_states = np.stack([dset.get_frames(j, [goal_idx[k]])[2][0].numpy() for k, j in enumerate(idxs)])
            # goal-conditioned policy, embed the goal image to condition on
            goal_emb = None
            if int(cfg.env.goal_dim) > 0 and goal_render_mode == "env":
                # draw the goal at its own coordinates, exactly as the random_state path does.
                if cfg.env.name == "wall" and env_infos and all(
                        "fix_door_location" in ei for ei in env_infos):
                    env.update_env(env_infos)
                goal_obs, _ = env.prepare(seeds, goal_states)
                goal_render = np.asarray(goal_obs["visual"])
                goal_emb = embed(goal_render)
            elif int(cfg.env.goal_dim) > 0:
                goal_imgs = torch.stack(
                    [dset.get_frames(j, [goal_idx[k]])[0]["visual"][0] for k, j in enumerate(idxs)]
                ).to(device)   # , b, C, H, W in [0,1]
                goal_emb = encoder(goal_imgs).reshape(len(idxs), 1, encoder.n_patches, encoder.emb_dim)

        # film only the globally-selected episodes that fall in this batch
        local_film = [i for i in range(n_envs) if (b * n_envs + i) in filmed_eps]
        record = len(local_film) > 0
        frames_per_env = {i: [] for i in local_film}
        goal_vis = {}
        if record:
            for i in local_film:
                if goal_render is not None:
                    goal_vis[i] = _frame_to_uint8(goal_render[i])   # already (H, W, C)
                else:
                    goal_vis[i] = _frame_to_uint8(rearrange(
                        dset.get_frames(idxs[i], [goal_idx[i]])[0]["visual"][0], "c h w -> h w c"))

        def rec(visual_np):
            if not record:
                return
            v = np.asarray(visual_np)
            for i in local_film:
                frame = _frame_to_uint8(v[i])
                # append goal on the right only if heights match, env render size can differ from
                # the dataset's resized goal frame
                if goal_vis[i].shape[0] == frame.shape[0]:
                    frame = np.concatenate([frame, goal_vis[i]], axis=1)
                frames_per_env[i].append(frame)

        # per-step goal tracking, ever-reached (max success) and closest approach (min dist)
        ever_succ = np.zeros(n_envs, dtype=bool)
        best_dist = np.full(n_envs, np.inf)
        last_succ, last_dist = None, None
        last_pos = None   # most recent (b, S) state, for the path integral
        path_len = np.zeros(n_envs)   # cumulative distance travelled
        ep_reward = np.zeros(n_envs, dtype=float)   # cumulative reward per episode

        def track(info):
            nonlocal last_succ, last_dist, last_pos, path_len
            # only COVERAGE task passes (pusht) skip the per-step eval_state round-trip, every
            # other pass, goal, random_state, wall/non-coverage task needs success/distance, so it
            # tracks.
            if coverage_task or info is None or "state" not in info[0]:
                return
            cur = np.stack([info[i]["state"] for i in range(n_envs)])
            # travelled distance, summed step-to-step displacement, so an episode that wanders far
            # and comes back is distinguishable from one that stops early.
            if last_pos is not None:
                path_len += np.linalg.norm(cur[:, :2] - last_pos[:, :2], axis=1)
            last_pos = cur.copy()
            r = env.eval_state(goal_states, cur)
            last_succ = np.asarray(r["success"]).astype(bool)
            ever_succ[:] = ever_succ | last_succ
            dk = "state_dist" if "state_dist" in r else (
                "chamfer_distance" if "chamfer_distance" in r else None)
            if dk is not None:
                last_dist = np.asarray(r[dk]).astype(float)
                best_dist[:] = np.minimum(best_dist, last_dist)

        # wall, each episode was recorded in its own maze layout (door/wall location), which must
        # be set before reset so the rollout and the goal share that configuration, otherwise
        # train) carries it in get_frames' info, PLANNED data does not store the maze, so its
        # env_info has no fix_door_location/fix_wall_location, skip update_env there (guard)
        # instead of crashing.
        if cfg.env.name == "wall":
            if env_infos and all("fix_door_location" in ei for ei in env_infos):
                env.update_env(env_infos)   # per-episode maze (valid / original)
            elif wall_fixed_info is not None:
                env.update_env([wall_fixed_info] * n_envs)   # planned data, the fixed wall_single maze
            elif b == 0:
                print(f"[{pass_name or 'eval'}] wall: no maze in the data and no wall dataset to "
                      f"pull the fixed maze from -> DEFAULT maze; success/distance may be unreliable.")
        obs, _ = env.prepare(seeds, init_states)
        rec(obs["visual"])
        obs_stack = deque([embed(obs["visual"])], maxlen=cfg.window_size)
        action_list, last_info, steps, done = [], None, 0, np.array([False])
        # eval_horizon counts TOTAL env steps (dino_wm envs never set done=True) so diffusion and
        # vqbet run equal-length episodes.
        while steps < horizon and not np.all(done):
            obs_seq = torch.cat(list(obs_stack), dim=1)   # , b, T, P, E
            # expand goal to the full window, diffusion pads obs to obs_horizon but not the goal,
            # so a shorter goal would mismatch in cat([goal, obs]).
            gseq = None if goal_emb is None else goal_emb.expand(-1, cfg.window_size, -1, -1)
            action, _, _ = policy(obs_seq, gseq, None)
            if use_diffusion:
                # diffusion returns (b, chunk, A), execute each action in the chunk
                for t in range(action.shape[1]):
                    obs, rew, done, info = step_env(action[:, t].cpu().numpy())
                    rec(obs["visual"]); track(info)
                    obs_stack.append(embed(obs["visual"]))
                    ep_reward += np.asarray(rew, dtype=float); last_info = info; steps += 1
                    if np.all(done) or steps >= horizon:
                        break
            else:
                # vqbet returns, b, T, chunk, A, temporal-ensemble the chunk
                if cfg.action_window_size > 1:
                    action_list.append(action[:, -1].cpu().numpy())   # (b, chunk, A)
                    if len(action_list) > cfg.action_window_size:
                        action_list = action_list[1:]
                    curr_action = np.array(action_list).mean(axis=0)[:, 0]   # (b, A)
                    action_list = [np.concatenate(
                        (c[:, 1:], np.zeros((c.shape[0], 1, c.shape[-1]))), axis=1) for c in action_list]
                else:
                    curr_action = action[:, -1, 0, :].cpu().numpy()
                obs, rew, done, info = step_env(curr_action)
                rec(obs["visual"]); track(info)
                obs_stack.append(embed(obs["visual"]))
                ep_reward += np.asarray(rew, dtype=float); last_info = info; steps += 1

        # coverage from the env's own info, final + max over the episode
        if last_info is not None and "final_coverage" in last_info[0]:
            final_cov += [last_info[i]["final_coverage"] for i in range(n_envs)]
            max_cov += [last_info[i]["max_coverage"] for i in range(n_envs)]
        # success, final-state vs ever-reached, distance, final vs closest approach
        if last_succ is not None:
            successes += last_succ.astype(float).tolist()
            max_successes += ever_succ.astype(float).tolist()
            # per-EPISODE record, so an aggregate can be partitioned after the fact, e.g.
            for _i in range(n_envs):
                _ep_records.append({
                    "episode": b * n_envs + _i,
                    "init_x": float(np.asarray(init_states)[_i][0]),
                    "init_y": float(np.asarray(init_states)[_i][1]),
                    "goal_x": float(np.asarray(goal_states)[_i][0]),
                    "goal_y": float(np.asarray(goal_states)[_i][1]),
                    "goal_dist": float(np.linalg.norm(
                        np.asarray(goal_states)[_i][:2] - np.asarray(init_states)[_i][:2])),
                    "success_max": float(ever_succ[_i]),
                    "success_final": float(last_succ[_i]),
                    "closest_dist": float(best_dist[_i]),
                    "final_x": (float(last_pos[_i][0]) if last_pos is not None else float("nan")),
                    "final_y": (float(last_pos[_i][1]) if last_pos is not None else float("nan")),
                    "path_len": float(path_len[_i]),
                })
        if last_dist is not None:
            state_dists += last_dist.tolist()
            min_dists += best_dist.tolist()
        rewards += ep_reward.tolist()   # cumulative reward per episode

        # write per-episode rollout videos, the globally-selected ones in this batch
        if record:
            import imageio
            os.makedirs(video_dir, exist_ok=True)
            for i in local_film:
                ge = b * n_envs + i
                # coverage tasks have no per-step success tracking -> tag by final coverage every
                # other pass, goal, random_state, non-coverage task tags by success/failure.
                if coverage_task and last_info is not None and "final_coverage" in last_info[0]:
                    tag = f"_cov{last_info[i]['final_coverage']:.2f}"
                else:
                    tag = "_success" if ever_succ[i] else "_failure"
                pfx = f"{pass_name}_" if pass_name else ""
                path = video_dir / f"eval_{pfx}epoch{epoch}_ep{ge}{tag}.mp4"
                imageio.mimsave(str(path), frames_per_env[i], fps=video_fps)
            print(f"saved {len(local_film)} eval videos (batch {b}) to {video_dir}")
    if _ep_records and out_dir is not None:
        import csv
        _csv = Path(out_dir) / f"eval_episodes_{pass_name or 'all'}.csv"
        with open(_csv, "w", newline="") as _f:
            _w = csv.DictWriter(_f, fieldnames=list(_ep_records[0].keys()))
            _w.writeheader()
            _w.writerows(_ep_records)
        print(f"[{pass_name or 'eval'}] wrote {len(_ep_records)} per-episode rows -> {_csv}")
    if own_env:
        env.close()

    metrics = {}
    if rewards:   # per-episode cumulative reward (pusht, integral of clip(coverage/thresh))
        rw = np.asarray(rewards)
        metrics["reward_mean"] = float(rw.mean())   # == patch_policy's avg_reward
        metrics["reward_max"] = float(rw.max())
        metrics["reward_min"] = float(rw.min())
    if successes:   # eval_state success, final-state vs ever-reached over the episode
        metrics["final_success_rate"] = float(np.mean(successes))   # success at the last frame
        metrics["max_success_rate"] = float(np.mean(max_successes))   # reached at any step
    if state_dists:   # distance-to-goal (continuous), mean/max/min over episodes, for the
        # final-frame distance and the closest approach during the episode
        sd, md = np.asarray(state_dists), np.asarray(min_dists)
        metrics["final_goal_dist_mean"] = float(sd.mean())
        metrics["final_goal_dist_max"] = float(sd.max())
        metrics["final_goal_dist_min"] = float(sd.min())
        metrics["closest_goal_dist_mean"] = float(md.mean())
        metrics["closest_goal_dist_max"] = float(md.max())
        metrics["closest_goal_dist_min"] = float(md.min())
    if final_cov:   # coverage (pusht), mean/max/min, matching online_eval.py
        metrics["final_coverage_mean"] = float(np.mean(final_cov))
        metrics["final_coverage_max"] = float(np.max(final_cov))
        metrics["final_coverage_min"] = float(np.min(final_cov))
        metrics["max_coverage_mean"] = float(np.mean(max_cov))
        metrics["max_coverage_max"] = float(np.max(max_cov))
        metrics["max_coverage_min"] = float(np.min(max_cov))
        print("final coverage mean:", metrics["final_coverage_mean"])
    # keep only the metrics this pass is about, coverage tasks (pusht task, non-random_state) ->
    # coverage + reward, everything else, goal, random_state, wall/non-coverage task ->
    # success/distance.
    if coverage_task:
        metrics = {k: v for k, v in metrics.items()
                   if "coverage" in k or "reward" in k}
    elif pass_type in ("goal", "task"):
        metrics = {k: v for k, v in metrics.items()
                   if "success_rate" in k or "goal_dist" in k}
    return metrics


# Run one or more env-eval passes and merge their metrics under "<name>/" prefixes.
#   type 'goal' -> short-goal success/distance vs a goal_offset-ahead goal (goal_offset optional
#   type 'task' -> full-horizon coverage vs the final frame on a coverage env (pusht). Wall has no
#   goal_source 'train' | 'valid' | 'eval_train' | 'eval_valid' | 'random_state' (overrides the
def run_eval_passes(cfg, encoder, model, device, use_diffusion, epoch, out_dir, env=None,
                    traj_dset=None, task_traj_dset=None):
    passes = cfg.get("eval_passes", None)
    if not passes:
        return eval_on_env(cfg, encoder, model, device, use_diffusion,
                           epoch=epoch, out_dir=out_dir, env=env, traj_dset=traj_dset,
                           task_traj_dset=task_traj_dset)
    merged = {}
    for p in passes:
        name = str(p.get("name", "") or "")
        m = eval_on_env(cfg, encoder, model, device, use_diffusion,
                        epoch=epoch, out_dir=out_dir,
                        goal_offset=p.get("goal_offset", "cfg"),
                        horizon=p.get("horizon", None), pass_name=name,
                        goal_source=p.get("goal_source", "cfg"),
                        pass_type=str(p.get("type", "all")), env=env, traj_dset=traj_dset,
                        task_traj_dset=task_traj_dset)
        prefix = f"{name}/" if name else ""
        merged.update({f"{prefix}{k}": v for k, v in m.items()})
    return merged


@hydra.main(config_path="conf", config_name="train_policy", version_base="1.2")
def _main(cfg: OmegaConf):
    main(cfg)


if __name__ == "__main__":
    _main()
