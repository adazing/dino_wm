import os
import json
import zlib
import time
import hydra
import torch
import wandb
import logging
import warnings
import threading
import itertools
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict
from einops import rearrange
from accelerate import Accelerator
from torchvision import utils
import torch.distributed as dist
from pathlib import Path
from collections import OrderedDict
from hydra.types import RunMode
from hydra.core.hydra_config import HydraConfig
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from metrics.image_metrics import eval_images
from utils import slice_trajdict_with_t, cfg_to_dict, seed, sample_tensors

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        with open_dict(cfg):
            # saved_folder is where THIS run writes. init_from is where it loads starting
            # weights, used to fine-tune an existing world model into a new run directory.
            # Unset means load from saved_folder, which is the plain resume behaviour.
            cfg["init_from"] = cfg.get("init_from", None)
            cfg["saved_folder"] = os.getcwd()
            log.info(f"Model saved dir: {cfg['saved_folder']}")
            if cfg["init_from"]:
                log.info(f"Initialising weights from: {cfg['init_from']}")
        cfg_dict = cfg_to_dict(cfg)
        model_name = cfg_dict["saved_folder"].split("outputs/")[-1]
        model_name += f"_{self.cfg.env.name}_f{self.cfg.frameskip}_h{self.cfg.num_hist}_p{self.cfg.num_pred}"

        if HydraConfig.get().mode == RunMode.MULTIRUN:
            log.info(" Multirun setup begin...")
            log.info(f"SLURM_JOB_NODELIST={os.environ['SLURM_JOB_NODELIST']}")
            log.info(f"DEBUGVAR={os.environ['DEBUGVAR']}")
            #==== init ddp process group ====
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            try:
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    timeout=timedelta(minutes=5),   # Set a 5-minute timeout
                )
                log.info("Multirun setup completed.")
            except Exception as e:
                log.error(f"DDP setup failed: {e}")
                raise
            torch.distributed.barrier()
            ## ==== /init ddp process group ====

        self.accelerator = Accelerator(log_with="wandb")
        log.info(
            f"rank: {self.accelerator.local_process_index}  model_name: {model_name}"
        )
        self.device = self.accelerator.device
        log.info(f"device: {self.device}   model_name: {model_name}")
        self.base_path = os.path.dirname(os.path.abspath(__file__))

        self.num_reconstruct_samples = self.cfg.training.num_reconstruct_samples
        self.total_epochs = self.cfg.training.epochs
        self.epoch = 0

        assert cfg.training.batch_size % self.accelerator.num_processes == 0, (
            "Batch size must be divisible by the number of processes. "
            f"Batch_size: {cfg.training.batch_size} num_processes: {self.accelerator.num_processes}."
        )

        OmegaConf.set_struct(cfg, False)
        cfg.effective_batch_size = cfg.training.batch_size
        cfg.gpu_batch_size = cfg.training.batch_size // self.accelerator.num_processes
        OmegaConf.set_struct(cfg, True)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            wandb_run_id = None
            if os.path.exists("hydra.yaml"):
                existing_cfg = OmegaConf.load("hydra.yaml")
                wandb_run_id = existing_cfg["wandb_run_id"]
                log.info(f"Resuming Wandb run {wandb_run_id}")

            wandb_dict = OmegaConf.to_container(cfg, resolve=True)
            if self.cfg.debug:
                log.info("WARNING: Running in debug mode...")
                self.wandb_run = wandb.init(
                    project="dino_wm_debug",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            else:
                self.wandb_run = wandb.init(
                    project="dino_wm",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            OmegaConf.set_struct(cfg, False)
            cfg.wandb_run_id = self.wandb_run.id
            OmegaConf.set_struct(cfg, True)
            wandb.run.name = "{}".format(model_name)
            with open(os.path.join(os.getcwd(), "hydra.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))

        seed(cfg.training.seed)
        # env and dataset are chosen independently (dataset config group), warn on an obvious
        # mismatch (e.g.
        _ds_target = str(OmegaConf.select(self.cfg, "env.dataset._target_") or "")
        if self.cfg.env.name == "wall" and "wall" not in _ds_target:
            log.warning(f"env=wall but dataset loader is {_ds_target!r} -- set dataset=wall?")
        elif self.cfg.env.name == "pusht" and "pusht" not in _ds_target:
            log.warning(f"env=pusht but dataset loader is {_ds_target!r} -- set dataset=pusht/pusht_noise?")
        elif self.cfg.env.name == "puzzle" and "puzzle" not in _ds_target:
            log.warning(f"env=puzzle but dataset loader is {_ds_target!r} -- set dataset=puzzle?")
        elif self.cfg.env.name == "point_maze" and "point_maze" not in _ds_target:
            log.warning(f"env=point_maze but dataset loader is {_ds_target!r} -- set dataset=point_maze?")
        log.info(f"Loading dataset from {self.cfg.env.dataset.data_path} ...")
        self.datasets, traj_dsets = hydra.utils.call(
            self.cfg.env.dataset,
            num_hist=self.cfg.num_hist,
            num_pred=self.cfg.num_pred,
            frameskip=self.cfg.frameskip,
        )

        # Model widths come from the demo slicer. data_source can replace datasets['train'] with
        # a ConcatDataset or a Subset, neither of which forwards these attributes, so read them
        # while it is still the plain slicer.
        self.train_dims = {k: getattr(self.datasets["train"], k)
                           for k in ("proprio_dim", "action_dim", "state_dim")}
        # The demo windows, captured BEFORE _build_capped_train_set replaces datasets['train'],
        # so the 'demo' group really is the demonstrations.
        self._demo_train = self.datasets["train"]

        # What the training set is made of: up to three independently capped groups of windows
        # (demonstrations / planner-success / planner-failure), plus an orthogonal
        # See _build_capped_train_set for the full contract.
        self._build_capped_train_set()


        self.train_traj_dset = traj_dsets["train"]
        self.val_traj_dset = traj_dsets["valid"]

        # The capped concat already sets the mix by group SIZE, so no sampler is needed. One
        # mechanism (the caps) instead of the old caps-vs-proportions precedence chain.
        _train_sampler = None


        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.datasets[x],
                batch_size=self.cfg.gpu_batch_size,
                shuffle=False,   # already shuffled in TrajSlicerDataset
                sampler=(_train_sampler if x == "train" else None),
                num_workers=self.cfg.env.num_workers,
                collate_fn=None,
            )
            for x in ["train", "valid"]
        }

        log.info(f"dataloader batch size: {self.cfg.gpu_batch_size}")

        self.dataloaders["train"], self.dataloaders["valid"] = self.accelerator.prepare(
            self.dataloaders["train"], self.dataloaders["valid"]
        )

        self.encoder = None
        self.action_encoder = None
        self.proprio_encoder = None
        self.predictor = None
        self.decoder = None
        self.train_encoder = self.cfg.model.train_encoder
        self.train_predictor = self.cfg.model.train_predictor
        self.train_decoder = self.cfg.model.train_decoder
        log.info(f"Train encoder, predictor, decoder:\
            {self.cfg.model.train_encoder}\
            {self.cfg.model.train_predictor}\
            {self.cfg.model.train_decoder}")

        self._keys_to_save = [
            "epoch",
        ]
        self._keys_to_save += (
            ["encoder", "encoder_optimizer"] if self.train_encoder else []
        )
        self._keys_to_save += (
            ["predictor", "predictor_optimizer"]
            if self.train_predictor and self.cfg.has_predictor
            else []
        )
        self._keys_to_save += (
            # also require has_decoder, with has_decoder=false there is no decoder_optimizer, so
            # saving it would KeyError.
            ["decoder", "decoder_optimizer"] if (self.train_decoder and self.cfg.has_decoder) else []
        )
        self._keys_to_save += ["action_encoder", "proprio_encoder"]

        self.init_models()
        self.init_optimizers()

        self.epoch_log = OrderedDict()

    def save_ckpt(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            ckpt = {}
            for k in self._keys_to_save:
                if hasattr(self.__dict__[k], "module"):
                    ckpt[k] = self.accelerator.unwrap_model(self.__dict__[k])
                else:
                    ckpt[k] = self.__dict__[k]
            torch.save(ckpt, "checkpoints/model_latest.pth")
            torch.save(ckpt, f"checkpoints/model_{self.epoch}.pth")
            log.info("Saved model to {}".format(os.getcwd()))
            ckpt_path = os.path.join(os.getcwd(), f"checkpoints/model_{self.epoch}.pth")
        else:
            ckpt_path = None
        model_name = self.cfg["saved_folder"].split("outputs/")[-1]
        model_epoch = self.epoch
        return ckpt_path, model_name, model_epoch

    def load_ckpt(self, filename="model_latest.pth"):
        ckpt = torch.load(filename, weights_only=False)
        for k, v in ckpt.items():
            self.__dict__[k] = v
        not_in_ckpt = set(self._keys_to_save) - set(ckpt.keys())
        if len(not_in_ckpt):
            log.warning("Keys not found in ckpt: %s", not_in_ckpt)

    def _build_capped_train_set(self):
        """Build datasets['train'] from up to FOUR independently capped groups.

        Groups:
          demo      the demonstration windows
          success   planned trajectories where the PLANNER reached its goal
          failure   planned trajectories where it did not
          rollout   the POLICY's own scouting episodes, recorded when
                    rollout_record_episodes > 0. Dynamics do not care who chose the action, so a
                    failed policy episode is valid world-model data -- and it is exactly where
                    the model is wrong. ~300 frames per episode against a planned trajectory's
                    26, so this is where the volume comes from.

        Each group has its own cap, in WINDOWS:
          null -> no cap, use every window the group has
          0    -> exclude the group entirely
          N    -> a seeded random N-window subset

        Any combination is legal: none, some or all capped. The training mix is whatever the caps
        leave behind, so there is exactly ONE mechanism and no precedence between options. This
        replaces data_source / original_weight / original_max_windows / plan_on_success /
        plan_on_failure / plan_on_original, which overlapped and silently overrode one another.
        """
        def _cap_of(key):
            v = self.cfg.get(key, None)
            if v is None:
                return None
            v = int(v)
            if v < 0:
                raise ValueError(f"{key} must be >= 0 or null, got {v}")
            return v

        def _demo_proprio_stats():
            """Demo proprio normalisation, so planned/rollout data reaches the model
            in the same units the world model was trained on."""
            d = self.datasets.get("train")
            for _ in range(6):
                if hasattr(d, "proprio_mean") and hasattr(d, "proprio_std"):
                    return d.proprio_mean, d.proprio_std
                d = getattr(d, "dataset", None)
                if d is None:
                    break
            log.warning("no demo proprio_mean/std found; planned and rollout windows "
                        "will use RAW proprioception")
            return None, None

        _p_mean, _p_std = _demo_proprio_stats()

        split = str(self.cfg.get("planned_split", "planner")).lower()
        if split != "planner":
            raise ValueError(
                f"planned_split must be 'planner', got {split!r}. The 'wm_error' mode was "
                f"removed 2026-11-24; see CODE_AUDIT-adaBot.md M5.")
        # Which two planned groups this mode uses, and the planned_dset filter for each.
        # The demo and rollout groups are mode-independent: demonstrations have no planner
        # outcome, and policy rollouts have no world-model prediction (the policy never runs the
        # world model), so neither can be split either way.
        # The wm_error alternative -- splitting on whether the WORLD MODEL mispredicted the
        # endpoint, at a percentile of the current pool -- was removed 2026-11-24 (M5): the
        # threshold moved every round, and the split was measured to be a near-restatement of
        # planner success/failure. planned_split therefore has only one value.
        sel_of = {"success": ("all", True), "failure": ("policy_fail", False)}

        caps = {"demo": _cap_of("max_demo_windows"),
                "rollout": _cap_of("max_rollout_windows")}
        for g in sel_of:
            caps[g] = _cap_of(f"max_{g}_windows")

        # PROPORTIONAL MODE. total_windows fixes the SIZE of the fine-tune's training set and
        # frac_<group> fixes the MIX, so successive rounds see the same amount of data in the
        # same ratio even as the pool grows. Give a fraction for all but ONE active group; the
        # omitted one takes the remainder. Fractions override the max_*_windows caps.
        _frac = None
        # B4, fixed 2026-11-25. An UNCAPPED fine-tune trains on the entire accumulated pool,
        # which grows by a round's worth of trajectories every round, so round 9 sees roughly ten
        # times round 0's data and no two rounds are comparable. DAGGER_README-adaBot.md states
        # the rule -- never leave total_windows null -- but nothing enforced it: both dagger
        # configs ship total_windows: null and every max_*_windows: null, so an arm whose override
        # list omits the cap failed silently and the only symptom was a long round.
        #
        # The guard fires only when a planner pool is in the mix. Pretraining from scratch has no
        # planned_data_path and legitimately uses all of its data. allow_uncapped_finetune=true is
        # the escape hatch, so an uncapped run is a deliberate choice rather than an oversight.
        _groups = ["demo", "rollout"] + sorted(sel_of)
        if (self.cfg.get("planned_data_path", None)
                and self.cfg.get("total_windows", None) is None
                and all(self.cfg.get(f"max_{g}_windows", None) is None for g in _groups)
                and not bool(self.cfg.get("allow_uncapped_finetune", False))):
            raise ValueError(
                "this fine-tune is UNCAPPED: planned_data_path is set but total_windows and every "
                f"max_<group>_windows ({', '.join('max_%s_windows' % g for g in _groups)}) are "
                "null, so it would train on the whole accumulated pool and grow every round. Set "
                "wm_total_windows with the wm_frac_<group> mix (e.g. wm_total_windows=500 "
                "wm_frac_failure=0.8 wm_frac_success=0.1), or a wm_max_<group>_windows cap, or "
                "pass allow_uncapped_finetune=true to do it on purpose. See "
                "CODE_AUDIT-adaBot.md B4.")

        _total = self.cfg.get("total_windows", None)
        if _total is not None:
            _total = int(_total)
            _active = ["demo", "rollout"] + sorted(sel_of)
            _fr, _missing = {}, []
            for g in _active:
                v = self.cfg.get(f"frac_{g}", None)
                if v is None:
                    _missing.append(g)
                else:
                    _fr[g] = float(v)
            # rollout data only exists when rollout_record_episodes > 0, so an unset frac_rollout
            # means 0 rather than "the remainder".
            if "rollout" in _missing and len(_missing) > 1:
                _fr["rollout"] = 0.0
                _missing.remove("rollout")
            if len(_missing) > 1:
                raise ValueError(
                    f"total_windows={_total} needs a frac_<group> for all but one of "
                    f"{_active}; {_missing} were all unset, so the remainder is ambiguous.")
            _sum = sum(_fr.values())
            if _sum > 1.0 + 1e-6:
                raise ValueError(f"the frac_<group> values sum to {_sum:.3f} > 1: {_fr}")
            if _missing:
                _fr[_missing[0]] = 1.0 - _sum
            elif abs(_sum - 1.0) > 1e-6:
                raise ValueError(
                    f"every group has a frac_<group> so they must sum to 1, got {_sum:.3f}")
            _frac = dict(_fr)
            caps = {g: int(round(_total * _fr[g])) for g in _active}
            log.info(f"total_windows={_total} with fractions "
                     + ", ".join(f"{g}={_fr[g]:.2f}" for g in _active)
                     + " -> caps " + ", ".join(f"{g}={caps[g]}" for g in _active))

        seed = int(self.cfg.get("window_subset_seed", self.cfg.training.seed))

        def _subset(ds, cap, label):
            """None = every window, 0 = drop the group, N = a random N-window subset.

            The subset is drawn fresh from window_subset_seed, which dagger_loop advances every
            round, so successive rounds train on different windows of the same group rather than
            one fixed slice. The group label is mixed into the seed so each group draws
            INDEPENDENTLY -- without it two groups of equal length got identical index sets and
            all groups' draws were correlated. Still fully reproducible from window_subset_seed.
            """
            if ds is None or len(ds) == 0 or cap == 0:
                return None
            if cap is None or cap >= len(ds):
                # FAILSAFE for re-training on identical data. A cap at or above the group's size
                # takes every window, so this round's set is last round's plus whatever is new.
                log.warning(
                    f"{label}: cap {'unset' if cap is None else cap} >= {len(ds):,} available, so "
                    f"ALL of its windows are used and this round repeats last round's data for "
                    f"this group. Set a cap below {len(ds):,} (or total_windows + frac_*).")
                return ds
            g = torch.Generator().manual_seed(seed + (zlib.crc32(label.encode()) & 0xFFFF))
            idx = torch.randperm(len(ds), generator=g)[:cap].tolist()
            log.info(f"{label}: capped {len(ds):,} -> {cap:,} random windows "
                     f"(window_subset_seed {seed})")
            return torch.utils.data.Subset(ds, sorted(idx))

        planned_path = self.cfg.get("planned_data_path", None)
        want = [g for g in sel_of if caps[g] != 0]
        if want and not planned_path:
            # a plain train.py run has no pool; demos only, same as the old data_source=original
            log.info("no planned_data_path given -> training on demonstrations only")
            want = []

        groups, sizes, built, n_pool, n_planned_traj = [], {}, [], None, 0
        if want:
            from datasets.planned_dset import PlannedTrajDataset
            from datasets.planned_dset import NoMatchingTrajectories
            from datasets.traj_dset import TrajSlicerDataset
            from datasets.img_transforms import default_transform
            n_frames = self.cfg.num_hist + self.cfg.num_pred
            for g in want:
                select, only_succ = sel_of[g]
                try:
                    dset = PlannedTrajDataset(
                        planned_path, transform=default_transform(self.cfg.img_size),
                        select=select, only_successes=only_succ,
                        max_trajectories=self.cfg.get("planned_max_trajectories", None),
                        selection=str(self.cfg.get("planned_selection", "first")),
                        selection_seed=int(self.cfg.get("planned_selection_seed",
                                                        self.cfg.training.seed)),
                        proprio_mean=_p_mean, proprio_std=_p_std)
                except NoMatchingTrajectories as e:
                    # a pool can legitimately hold none of one outcome, e.g. round 0 before the
                    # planner has failed at anything. Anything ELSE -- a bad select name, missing
                    # world-model error metadata -- propagates, so a typo cannot masquerade as
                    # an empty group and silently halve the fine-tune's data.
                    log.warning(f"group {g!r}: no matching trajectories ({e}); skipped")
                    sizes[g] = 0
                    continue
                # TrajSlicerDataset permutes with the GLOBAL numpy RNG. Restore the stream or a
                # fine-tune diverges from a plain train run started with the same seed.
                st = np.random.get_state()
                sl = TrajSlicerDataset(dset, n_frames, self.cfg.frameskip,
                                       slice_seed=int(self.cfg.training.seed))   # B19
                np.random.set_state(st)
                bad = {k: (self.train_dims[k], getattr(sl, k))
                       for k in ("proprio_dim", "action_dim")
                       if getattr(sl, k) != self.train_dims[k]}
                if bad:
                    raise ValueError(
                        f"planned data at {planned_path} disagrees with the demo dataset on "
                        + ", ".join(f"{k} (demo {a}, planned {b})" for k, (a, b) in bad.items())
                        + ". It was generated by a world model trained on different data, so it "
                          "cannot be mixed in.")
                n_pool = getattr(dset, "n_pool", n_pool)
                n_planned_traj += len(dset)
                built.append((g, sl))

        # ---- the rollout group ---------------------------------------------------------------
        # Same shard layout as the planner pool, so the same dataset class reads it. select/
        # only_successes do not apply here: these episodes have no planner outcome and no
        # world-model prediction stored, so they are taken whole.
        roll_path = self.cfg.get("rollout_data_path", None)
        if caps["rollout"] != 0 and roll_path and Path(roll_path).is_dir():
            from datasets.planned_dset import PlannedTrajDataset
            from datasets.traj_dset import TrajSlicerDataset
            from datasets.img_transforms import default_transform
            try:
                rd = PlannedTrajDataset(str(roll_path),
                                        transform=default_transform(self.cfg.img_size),
                                        select="all", only_successes=False,
                                        proprio_mean=_p_mean, proprio_std=_p_std)
                st = np.random.get_state()
                rsl = TrajSlicerDataset(rd, self.cfg.num_hist + self.cfg.num_pred,
                                        self.cfg.frameskip)
                np.random.set_state(st)
                rbad = {k: (self.train_dims[k], getattr(rsl, k))
                        for k in ("proprio_dim", "action_dim")
                        if getattr(rsl, k) != self.train_dims[k]}
                if rbad:
                    raise ValueError(
                        f"rollout data at {roll_path} disagrees with the demo dataset on "
                        + ", ".join(f"{k} (demo {a}, rollout {b})"
                                    for k, (a, b) in rbad.items()))
                built.append(("rollout", rsl))
            except Exception as e:
                # never let a missing/!malformed rollout pool kill a fine-tune that would
                # otherwise run on the planner data
                log.warning(f"group 'rollout': unusable ({type(e).__name__}: {e}); skipped")
                sizes["rollout"] = 0
        elif caps["rollout"] != 0 and roll_path:
            log.info(f"group 'rollout': {roll_path} does not exist yet; skipped")
            sizes["rollout"] = 0

        if caps["demo"] != 0 and self._demo_train is not None:
            built.append(("demo", self._demo_train))

        # REALLOCATION. total_windows is a budget and frac_<group> a target ratio, but a group may
        # not have that many windows, since early rounds have a small pool.
        # Give each group min(cap, what it has) and hand the shortfall to whoever has spare,
        # weighted by the requested fractions, so the ratio is the closest achievable one and the
        # total is preserved whenever the data exists at all. Never errors.
        if _total is not None and built:
            avail = {g: len(d) for g, d in built}
            alloc = {g: min(caps.get(g, 0), avail[g]) for g in avail}
            short = [g for g in avail if alloc[g] < caps.get(g, 0)]
            deficit = _total - sum(alloc.values())
            if deficit > 0:
                spare = {g: avail[g] - alloc[g] for g in alloc if avail[g] > alloc[g]}
                wsum = sum(_frac.get(g, 0.0) for g in spare)
                for g in sorted(spare, key=lambda x: -_frac.get(x, 0.0)):
                    if deficit <= 0:
                        break
                    share = (deficit if wsum <= 0
                             else int(round(deficit * _frac.get(g, 0.0) / wsum)))
                    take = min(spare[g], max(0, share), deficit)
                    alloc[g] += take
                    deficit -= take
                for g in sorted(spare, key=lambda x: -(avail[x] - alloc[x])):
                    if deficit <= 0:
                        break
                    take = min(avail[g] - alloc[g], deficit)
                    alloc[g] += take
                    deficit -= take
            if short:
                log.warning(
                    "total_windows=%s not met at the requested ratio: " % _total
                    + ", ".join(f"{g} wanted {caps.get(g, 0):,} had {avail[g]:,}" for g in short)
                    + " -> using "
                    + ", ".join(f"{g}={alloc[g]:,}" for g in sorted(alloc))
                    + f" ({sum(alloc.values()):,} of {_total:,})")
            caps.update(alloc)

        for _g, _d in built:
            _sub = _subset(_d, caps.get(_g, None), f"group {_g!r}")
            sizes[_g] = 0 if _sub is None else len(_sub)
            if _sub is not None:
                if _g == "demo":
                    groups.insert(0, _sub)   # demos first, index order matches the old concat
                else:
                    groups.append(_sub)

        if not groups:
            raise ValueError(
                "every group is empty or capped to 0, so there is nothing to train on. Set at "
                "least one of max_demo_windows / max_rollout_windows / "
                + " / ".join(f"max_{g}_windows" for g in sorted(sel_of))
                + " to a positive number or to null.")

        self.datasets["train"] = (groups[0] if len(groups) == 1
                                  else torch.utils.data.ConcatDataset(groups))
        total = len(self.datasets["train"])
        _order = ["demo"] + sorted(sel_of) + ["rollout"]
        log.info("train set: "
                 + ", ".join(f"{g} {sizes.get(g, 0):,}" for g in _order)
                 + f" windows -> {total:,} total")

        n_plan_w = sum(sizes.get(g, 0) for g in sel_of)
        counts = {f"data/{g}_windows": sizes.get(g, 0) for g in _order}
        counts["data/planned_split"] = split
        counts.update({
            "data/total_windows": total,
            "data/planned_windows": n_plan_w,
            "data/planned_trajectories": n_planned_traj,
            "data/pool_trajectories": n_pool,
            "data/planned_window_frac": n_plan_w / max(1, total),
        })
        counts = {k: v for k, v in counts.items() if v is not None}
        # dagger_loop reads this back so the numbers land on the loop run's round axis
        with open(os.path.join(os.getcwd(), "data_counts.json"), "w") as f:
            json.dump(counts, f, indent=2)
        if getattr(self, "wandb_run", None) is not None:
            self.wandb_run.summary.update(counts)   # sortable column in the runs table
            self.wandb_run.log(counts)


    def init_models(self):
        _src = self.cfg.get("init_from") or self.cfg.saved_folder
        # init_epoch picks WHICH checkpoint a fine-tune starts from. Without it this always loaded
        # model_latest.pth, so a world model planned with at epoch 56 was fine-tuned from epoch 100.
        _ep = self.cfg.get("init_epoch", None)
        _ck = (f"model_{_ep}.pth" if _ep is not None and str(_ep) != "latest"
               else "model_latest.pth")
        model_ckpt = Path(_src) / "checkpoints" / _ck
        if self.cfg.get("init_from") and not model_ckpt.exists():
            raise FileNotFoundError(f"init_from checkpoint not found: {model_ckpt}")
        if model_ckpt.exists():
            self.load_ckpt(model_ckpt)
            log.info(f"Resuming from epoch {self.epoch}, {model_ckpt}")
            # Fine-tuning loads the weights but starts its own epoch count, so training.epochs
            # is the number of fine-tune epochs rather than a total the checkpoint may already
            # have passed. init_optimizers runs after this and rebuilds the optimizers at the
            # configured learning rates, so the loaded optimizer state does not carry over.
            if bool(self.cfg.get("reset_epoch", False)):
                self.epoch = 0
                log.info(f"reset_epoch, training {self.cfg.training.epochs} more epochs")

        # initialize encoder
        if self.encoder is None:
            self.encoder = hydra.utils.instantiate(
                self.cfg.encoder,
            )
        if not self.train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Build fresh encoders only when no checkpoint supplied them. Rebuilding unconditionally
        # threw away the trained action/proprio encoders on every fine-tune or resume, so the
        # predictor received embeddings it had never seen and had to relearn them from scratch.
        if self.proprio_encoder is None:
            self.proprio_encoder = hydra.utils.instantiate(
                self.cfg.proprio_encoder,
                in_chans=self.train_dims["proprio_dim"],
                emb_dim=self.cfg.proprio_emb_dim,
            )
        proprio_emb_dim = self.proprio_encoder.emb_dim
        print(f"Proprio encoder type: {type(self.proprio_encoder)}")
        self.proprio_encoder = self.accelerator.prepare(self.proprio_encoder)

        if self.action_encoder is None:
            self.action_encoder = hydra.utils.instantiate(
                self.cfg.action_encoder,
                in_chans=self.train_dims["action_dim"],
                emb_dim=self.cfg.action_emb_dim,
            )
        action_emb_dim = self.action_encoder.emb_dim
        print(f"Action encoder type: {type(self.action_encoder)}")

        self.action_encoder = self.accelerator.prepare(self.action_encoder)

        # Restrict the predictor's trainable parameters to its LAST transformer block plus the
        # final norm, the approach adajepa uses (planning/adajepa.py:_select_predictor_params).
        # Their setting is lr 5e-4 with steps 1 on that slice only, i.e. the rate that wrecks a
        # full-predictor fine-tune is survivable when the slice is this small. Use it when
        # fine-tuning a converged world model on a little new data.
        if bool(self.cfg.get("finetune_last_layer_only", False)):
            _pred = self.predictor
            _pred = getattr(_pred, "module", _pred)   # unwrap accelerate/DDP
            _tr = getattr(_pred, "transformer", None)
            if _tr is None or not hasattr(_tr, "layers"):
                raise ValueError(
                    "finetune_last_layer_only needs a predictor with .transformer.layers "
                    f"(ViTPredictor); got {type(_pred).__name__}.")
            _keep = set()
            for _prm in list(_tr.layers[-1].parameters()) + list(_tr.norm.parameters()):
                _keep.add(id(_prm))
            _n_on = _n_off = 0
            for _prm in _pred.parameters():
                if id(_prm) in _keep:
                    _prm.requires_grad_(True); _n_on += _prm.numel()
                else:
                    _prm.requires_grad_(False); _n_off += _prm.numel()
            log.info(f"finetune_last_layer_only: predictor trainable {_n_on:,} params, "
                     f"frozen {_n_off:,} ({100*_n_on/max(1,_n_on+_n_off):.1f}% trainable, "
                     f"last of {len(_tr.layers)} blocks + final norm)")

        # Hold the action/proprio encoders fixed for the whole fine-tune. The predictor was
        # trained against these exact embeddings; if both the embeddings and the predictor move
        # at once the predictor is chasing a moving input representation, on the handful of
        # windows a fine-tune round supplies (154 windows in wall round 0). Separate from
        # bug-2's fix, which only made them LOAD from the checkpoint instead of being rebuilt
        # from random -- loaded is not the same as frozen. finetune_last_layer_only forces this
        # on, because there the surrounding predictor blocks are frozen too and the embeddings
        # they were trained against must not shift underneath them.
        if (bool(self.cfg.get("freeze_action_proprio_encoders", False))
                or bool(self.cfg.get("finetune_last_layer_only", False))):
            _n_enc = 0
            for _enc in (self.action_encoder, self.proprio_encoder):
                for _prm in _enc.parameters():
                    _prm.requires_grad_(False)
                    _n_enc += _prm.numel()
            log.info(f"action and proprio encoders FROZEN ({_n_enc:,} params); "
                     f"training.action_encoder_lr has no effect")

        if self.accelerator.is_main_process:
            self.wandb_run.watch(self.action_encoder)
            self.wandb_run.watch(self.proprio_encoder)

        # predictor token count. A pooled/1-token encoder (avg_pool, cls -> latent_ndim==1) is a
        # single token.
        if self.encoder.latent_ndim == 1:   # pooled / 1D feature
            num_patches = 1
        elif "dino" in getattr(self.encoder, "name", ""):   # VWorldModel resizes dino to the 16x grid
            num_patches = (self.cfg.img_size // 16) ** 2
        else:
            num_patches = getattr(self.encoder, "n_patches", (self.cfg.img_size // 16) ** 2)

        if self.cfg.concat_dim == 0:
            num_patches += 2

        # A pooled encoder (1 token) has no spatial grid for the vqvae decoder to reconstruct
        # from.
        if self.cfg.has_decoder and self.encoder.latent_ndim == 1:
            raise ValueError(
                f"has_decoder=true is incompatible with a pooled/1-token encoder "
                f"({self.cfg.encoder.get('_target_', '?')}, latent_ndim=1): the vqvae decoder "
                f"reconstructs from a spatial patch grid. Train latent-only with has_decoder=false "
                f"(planning/BC/generation don't need the decoder), or use a patch encoder "
                f"(dino / *_patch) for image reconstruction.")

        if self.cfg.has_predictor:
            if self.predictor is None:
                self.predictor = hydra.utils.instantiate(
                    self.cfg.predictor,
                    num_patches=num_patches,
                    num_frames=self.cfg.num_hist,
                    dim=self.encoder.emb_dim
                    + (
                        proprio_emb_dim * self.cfg.num_proprio_repeat
                        + action_emb_dim * self.cfg.num_action_repeat
                    )
                    * (self.cfg.concat_dim),
                )
            if not self.train_predictor:
                for param in self.predictor.parameters():
                    param.requires_grad = False

        # initialize decoder
        if self.cfg.has_decoder:
            if self.decoder is None:
                if self.cfg.env.decoder_path is not None:
                    decoder_path = os.path.join(
                        self.base_path, self.cfg.env.decoder_path
                    )
                    ckpt = torch.load(decoder_path, weights_only=False)
                    if isinstance(ckpt, dict):
                        self.decoder = ckpt["decoder"]
                    else:
                        self.decoder = torch.load(decoder_path, weights_only=False)
                    log.info(f"Loaded decoder from {decoder_path}")
                else:
                    self.decoder = hydra.utils.instantiate(
                        self.cfg.decoder,
                        emb_dim=self.encoder.emb_dim,   # 384
                    )
            if not self.train_decoder:
                for param in self.decoder.parameters():
                    param.requires_grad = False
        # prepare only the modules that exist, has_predictor=false / has_decoder=false leave them
        # None (e.g.
        self.encoder = self.accelerator.prepare(self.encoder)
        if self.predictor is not None:
            self.predictor = self.accelerator.prepare(self.predictor)
        if self.decoder is not None:
            self.decoder = self.accelerator.prepare(self.decoder)
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            encoder=self.encoder,
            proprio_encoder=self.proprio_encoder,
            action_encoder=self.action_encoder,
            predictor=self.predictor,
            decoder=self.decoder,
            proprio_dim=proprio_emb_dim,
            action_dim=action_emb_dim,
            concat_dim=self.cfg.concat_dim,
            num_action_repeat=self.cfg.num_action_repeat,
            num_proprio_repeat=self.cfg.num_proprio_repeat,
        )

    def init_optimizers(self):
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.cfg.training.encoder_lr,
        )
        self.encoder_optimizer = self.accelerator.prepare(self.encoder_optimizer)
        if self.cfg.has_predictor:
            self.predictor_optimizer = torch.optim.AdamW(
                self.predictor.parameters(),
                lr=self.cfg.training.predictor_lr,
            )
            self.predictor_optimizer = self.accelerator.prepare(
                self.predictor_optimizer
            )

            self.action_encoder_optimizer = torch.optim.AdamW(
                itertools.chain(
                    self.action_encoder.parameters(), self.proprio_encoder.parameters()
                ),
                lr=self.cfg.training.action_encoder_lr,
            )
            self.action_encoder_optimizer = self.accelerator.prepare(
                self.action_encoder_optimizer
            )

        if self.cfg.has_decoder:
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(), lr=self.cfg.training.decoder_lr
            )
            self.decoder_optimizer = self.accelerator.prepare(self.decoder_optimizer)

    def monitor_jobs(self, lock):
        """
        check planning eval jobs' status and update logs
        """
        while True:
            with lock:
                finished_jobs = [
                    job_tuple for job_tuple in self.job_set if job_tuple[2].done()
                ]
                for epoch, job_name, job in finished_jobs:
                    result = job.result()
                    print(f"Logging result for {job_name} at epoch {epoch}: {result}")
                    log_data = {
                        f"{job_name}/{key}": value for key, value in result.items()
                    }
                    log_data["epoch"] = epoch
                    self.wandb_run.log(log_data)
                    self.job_set.remove((epoch, job_name, job))
            time.sleep(1)

    def run(self):
        if self.accelerator.is_main_process:
            executor = ThreadPoolExecutor(max_workers=4)
            self.job_set = set()
            lock = threading.Lock()

            self.monitor_thread = threading.Thread(
                target=self.monitor_jobs, args=(lock,), daemon=True
            )
            self.monitor_thread.start()

        init_epoch = self.epoch + 1   # epoch starts from 1
        for epoch in range(init_epoch, init_epoch + self.total_epochs):
            self.epoch = epoch
            self.accelerator.wait_for_everyone()
            self.train()
            self.accelerator.wait_for_everyone()
            self.val()
            self.logs_flash(step=self.epoch)
            if self.epoch % self.cfg.training.save_every_x_epoch == 0:
                ckpt_path, model_name, model_epoch = self.save_ckpt()
                # main thread only, launch planning jobs on the saved ckpt
                if (
                    self.cfg.plan_settings.plan_cfg_path is not None
                    and ckpt_path is not None
                ):   # ckpt_path is only not None for main process
                    from plan import build_plan_cfg_dicts, launch_plan_jobs

                    cfg_dicts = build_plan_cfg_dicts(
                        plan_cfg_path=os.path.join(
                            self.base_path, self.cfg.plan_settings.plan_cfg_path
                        ),
                        ckpt_base_path=self.cfg.ckpt_base_path,
                        model_name=model_name,
                        model_epoch=model_epoch,
                        planner=self.cfg.plan_settings.planner,
                        goal_source=self.cfg.plan_settings.goal_source,
                        goal_H=self.cfg.plan_settings.goal_H,
                        alpha=self.cfg.plan_settings.alpha,
                    )
                    jobs = launch_plan_jobs(
                        epoch=self.epoch,
                        cfg_dicts=cfg_dicts,
                        plan_output_dir=os.path.join(
                            os.getcwd(), "submitit-evals", f"epoch_{self.epoch}"
                        ),
                    )
                    with lock:
                        self.job_set.update(jobs)

    def err_eval_single(self, z_pred, z_tgt):
        logs = {}
        for k in z_pred.keys():
            loss = self.model.emb_criterion(z_pred[k], z_tgt[k])
            logs[k] = loss
        return logs

    def err_eval(self, z_out, z_tgt, state_tgt=None):
        """
        z_pred: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        z_tgt: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        state:  (b, n_hist, dim)
        """
        logs = {}
        slices = {
            "full": (None, None),
            "pred": (-self.model.num_pred, None),
            "next1": (-self.model.num_pred, -self.model.num_pred + 1),
        }
        for name, (start_idx, end_idx) in slices.items():
            z_out_slice = slice_trajdict_with_t(
                z_out, start_idx=start_idx, end_idx=end_idx
            )
            z_tgt_slice = slice_trajdict_with_t(
                z_tgt, start_idx=start_idx, end_idx=end_idx
            )
            z_err = self.err_eval_single(z_out_slice, z_tgt_slice)

            logs.update({f"z_{k}_err_{name}": v for k, v in z_err.items()})

        return logs

    def train(self):
        for i, data in enumerate(
            tqdm(self.dataloaders["train"], desc=f"Epoch {self.epoch} Train")
        ):
            obs, act, state = data
            plot = i == 0   # only plot from the first batch
            self.model.train()
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                obs, act
            )

            self.encoder_optimizer.zero_grad()
            if self.cfg.has_decoder:
                self.decoder_optimizer.zero_grad()
            if self.cfg.has_predictor:
                self.predictor_optimizer.zero_grad()
                self.action_encoder_optimizer.zero_grad()

            self.accelerator.backward(loss)

            if self.model.train_encoder:
                self.encoder_optimizer.step()
            if self.cfg.has_decoder and self.model.train_decoder:
                self.decoder_optimizer.step()
            if self.cfg.has_predictor and self.model.train_predictor:
                self.predictor_optimizer.step()
                self.action_encoder_optimizer.step()

            loss = self.accelerator.gather_for_metrics(loss).mean()

            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }
            # latent prediction error (no decoder needed), logged for latent-only WMs too
            if plot and self.cfg.has_predictor:
                z_obs_out, z_act_out = self.model.separate_emb(z_out)
                z_gt = self.model.encode_obs(obs)
                z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)
                err_logs = self.err_eval(z_obs_out, z_tgt)
                err_logs = self.accelerator.gather_for_metrics(err_logs)
                err_logs = {
                    key: value.mean().item() for key, value in err_logs.items()
                }
                err_logs = {f"train_{k}": [v] for k, v in err_logs.items()}
                self.logs_update(err_logs)

            # image metrics + reconstruction plots (need the decoder)
            if self.cfg.has_decoder and plot:
                if visual_out is not None:
                    for t in range(
                        self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(
                            img_pred_scores
                        )
                        img_pred_scores = {
                            f"train_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = self.accelerator.gather_for_metrics(
                            img_reconstruction_scores
                        )
                        img_reconstruction_scores = {
                            f"train_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="train",
                )

            loss_components = {f"train_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)

    def val(self):
        self.model.eval()
        if len(self.train_traj_dset) > 0 and self.cfg.has_predictor:
            with torch.no_grad():
                train_rollout_logs = self.openloop_rollout(
                    self.train_traj_dset, mode="train"
                )
                train_rollout_logs = {
                    f"train_{k}": [v] for k, v in train_rollout_logs.items()
                }
                self.logs_update(train_rollout_logs)
                val_rollout_logs = self.openloop_rollout(self.val_traj_dset, mode="val")
                val_rollout_logs = {
                    f"val_{k}": [v] for k, v in val_rollout_logs.items()
                }
                self.logs_update(val_rollout_logs)

        self.accelerator.wait_for_everyone()
        for i, data in enumerate(
            tqdm(self.dataloaders["valid"], desc=f"Epoch {self.epoch} Valid")
        ):
            obs, act, state = data
            plot = i == 0
            self.model.eval()
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                obs, act
            )

            loss = self.accelerator.gather_for_metrics(loss).mean()

            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }

            # latent prediction error (no decoder needed), logged for latent-only WMs too
            if plot and self.cfg.has_predictor:
                z_obs_out, z_act_out = self.model.separate_emb(z_out)
                z_gt = self.model.encode_obs(obs)
                z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)
                err_logs = self.err_eval(z_obs_out, z_tgt)
                err_logs = self.accelerator.gather_for_metrics(err_logs)
                err_logs = {
                    key: value.mean().item() for key, value in err_logs.items()
                }
                err_logs = {f"val_{k}": [v] for k, v in err_logs.items()}
                self.logs_update(err_logs)

            # image metrics + reconstruction plots (need the decoder)
            if self.cfg.has_decoder and plot:
                if visual_out is not None:
                    for t in range(
                        self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(
                            img_pred_scores
                        )
                        img_pred_scores = {
                            f"val_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = self.accelerator.gather_for_metrics(
                            img_reconstruction_scores
                        )
                        img_reconstruction_scores = {
                            f"val_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="valid",
                )
            loss_components = {f"val_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)

    def openloop_rollout(
        self, dset, num_rollout=10, rand_start_end=True, min_horizon=2, mode="train"
    ):
        np.random.seed(self.cfg.training.seed)
        min_horizon = min_horizon + self.cfg.num_hist
        plotting_dir = f"rollout_plots/e{self.epoch}_rollout"
        if self.accelerator.is_main_process:
            os.makedirs(plotting_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        logs = {}

        # rollout with both num_hist and 1 frame as context
        num_past = [(self.cfg.num_hist, ""), (1, "_1framestart")]

        # sample traj
        for idx in range(num_rollout):
            valid_traj = False
            while not valid_traj:
                traj_idx = np.random.randint(0, len(dset))
                obs, act, state, _ = dset[traj_idx]
                act = act.to(self.device)
                if rand_start_end:
                    if obs["visual"].shape[0] > min_horizon * self.cfg.frameskip + 1:
                        start = np.random.randint(
                            0,
                            obs["visual"].shape[0] - min_horizon * self.cfg.frameskip - 1,
                        )
                    else:
                        start = 0
                    max_horizon = (obs["visual"].shape[0] - start - 1) // self.cfg.frameskip
                    if max_horizon > min_horizon:
                        valid_traj = True
                        horizon = np.random.randint(min_horizon, max_horizon + 1)
                else:
                    valid_traj = True
                    start = 0
                    horizon = (obs["visual"].shape[0] - 1) // self.cfg.frameskip

            for k in obs.keys():
                obs[k] = obs[k][
                    start :
                    start + horizon * self.cfg.frameskip + 1 :
                    self.cfg.frameskip
                ]
            act = act[start : start + horizon * self.cfg.frameskip]
            act = rearrange(act, "(h f) d -> h (f d)", f=self.cfg.frameskip)

            obs_g = {}
            for k in obs.keys():
                obs_g[k] = obs[k][-1].unsqueeze(0).unsqueeze(0).to(self.device)
            z_g = self.model.encode_obs(obs_g)
            actions = act.unsqueeze(0)

            for past in num_past:
                n_past, postfix = past

                obs_0 = {}
                for k in obs.keys():
                    obs_0[k] = (
                        obs[k][:n_past].unsqueeze(0).to(self.device)
                    )   # unsqueeze for batch, b, t, c, h, w

                z_obses, z = self.model.rollout(obs_0, actions)
                z_obs_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)
                div_loss = self.err_eval_single(z_obs_last, z_g)

                for k in div_loss.keys():
                    log_key = f"z_{k}_err_rollout{postfix}"
                    if log_key in logs:
                        logs[f"z_{k}_err_rollout{postfix}"].append(
                            div_loss[k]
                        )
                    else:
                        logs[f"z_{k}_err_rollout{postfix}"] = [
                            div_loss[k]
                        ]

                if self.cfg.has_decoder:
                    visuals = self.model.decode_obs(z_obses)[0]["visual"]
                    imgs = torch.cat([obs["visual"], visuals[0].cpu()], dim=0)
                    self.plot_imgs(
                        imgs,
                        obs["visual"].shape[0],
                        f"{plotting_dir}/e{self.epoch}_{mode}_{idx}{postfix}.png",
                    )
        logs = {
            key: sum(values) / len(values) for key, values in logs.items() if values
        }
        return logs

    def logs_update(self, logs):
        for key, value in logs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            length = len(value)
            count, total = self.epoch_log.get(key, (0, 0.0))
            self.epoch_log[key] = (
                count + length,
                total + sum(value),
            )

    def logs_flash(self, step):
        epoch_log = OrderedDict()
        for key, value in self.epoch_log.items():
            count, sum = value
            to_log = sum / count
            epoch_log[key] = to_log
        epoch_log["epoch"] = step
        log.info(f"Epoch {self.epoch}  Training loss: {epoch_log['train_loss']:.4f}  \
                Validation loss: {epoch_log['val_loss']:.4f}")

        if self.accelerator.is_main_process:
            self.wandb_run.log(epoch_log)
            # Running record of this fine-tune's loss curve, so dagger_loop can put it on the
            # LOOP run's round axis. Rewritten every epoch, so a killed run still leaves a
            # usable file. This is the series that showed validation loss rising every epoch of
            # every round (+3%, +16%, +42% on 20261008212236_wall).
            _vh = getattr(self, "_val_hist", [])
            _vl = epoch_log.get("val_loss", None)
            if _vl is not None:
                _vh.append(float(_vl))
                self._val_hist = _vh
                _best_i = int(np.argmin(_vh))
                _m = {"val_loss_per_epoch": _vh,
                      "val_loss_first": _vh[0],
                      "val_loss_final": _vh[-1],
                      "val_loss_best": _vh[_best_i],
                      "val_loss_best_epoch": _best_i + 1,
                      "val_loss_delta_pct": 100.0 * (_vh[-1] - _vh[0]) / max(1e-12, abs(_vh[0])),
                      "train_loss_final": float(epoch_log.get("train_loss", float("nan"))),
                      "epochs_run": len(_vh)}
                with open(os.path.join(os.getcwd(), "wm_metrics.json"), "w") as _f:
                    json.dump(_m, _f, indent=2)

                # World-model best-checkpoint selection was REMOVED 2026-11-23 (M1). It wrote
                # checkpoints/model_best.pth whenever the newest epoch had the lowest val_loss,
                # and dagger_loop carried that forward. The validation set is dominated by
                # demonstration windows, so the selector rejected any fine-tune that traded a
                # little demonstration loss for the states the planner visits -- which is every
                # useful fine-tune. val_loss_best_epoch is still reported above for diagnosis;
                # nothing acts on it.
        self.epoch_log = OrderedDict()

    def plot_samples(
        self,
        gt_imgs,
        pred_imgs,
        reconstructed_gt_imgs,
        epoch,
        batch,
        num_samples=2,
        phase="train",
    ):
        """
        input:  gt_imgs, reconstructed_gt_imgs: (b, num_hist + num_pred, 3, img_size, img_size)
                pred_imgs: (b, num_hist, 3, img_size, img_size)
        output:   imgs: (b, num_frames, 3, img_size, img_size)
        """
        num_frames = gt_imgs.shape[1]
        # sample num_samples images
        gt_imgs, pred_imgs, reconstructed_gt_imgs = sample_tensors(
            [gt_imgs, pred_imgs, reconstructed_gt_imgs],
            num_samples,
            indices=list(range(num_samples))[: gt_imgs.shape[0]],
        )

        num_samples = min(num_samples, gt_imgs.shape[0])

        # fill in blank images for frameskips
        if pred_imgs is not None:
            pred_imgs = torch.cat(
                (
                    torch.full(
                        (num_samples, self.model.num_pred, *pred_imgs.shape[2:]),
                        -1,
                        device=self.device,
                    ),
                    pred_imgs,
                ),
                dim=1,
            )
        else:
            pred_imgs = torch.full(gt_imgs.shape, -1, device=self.device)

        pred_imgs = rearrange(pred_imgs, "b t c h w -> (b t) c h w")
        gt_imgs = rearrange(gt_imgs, "b t c h w -> (b t) c h w")
        reconstructed_gt_imgs = rearrange(
            reconstructed_gt_imgs, "b t c h w -> (b t) c h w"
        )
        imgs = torch.cat([gt_imgs, pred_imgs, reconstructed_gt_imgs], dim=0)

        if self.accelerator.is_main_process:
            os.makedirs(phase, exist_ok=True)
        self.accelerator.wait_for_everyone()

        self.plot_imgs(
            imgs,
            num_columns=num_samples * num_frames,
            img_name=f"{phase}/{phase}_e{str(epoch).zfill(5)}_b{batch}.png",
        )

    def plot_imgs(self, imgs, num_columns, img_name):
        utils.save_image(
            imgs,
            img_name,
            nrow=num_columns,
            normalize=True,
            value_range=(0, 1),   # visuals are in [0, 1]
        )


@hydra.main(config_path="conf", config_name="train")
def main(cfg: OmegaConf):
    trainer = Trainer(cfg)
    try:
        trainer.run()
    finally:
        # Close the run explicitly, on failures too. Without this a round that crashes, or one
        # whose atexit hook does not get to run, stays marked 'running' in wandb forever.
        _run = getattr(trainer, "wandb_run", None)
        if _run is not None:
            try:
                _run.finish()
            except Exception as _e:
                print(f"wandb.finish failed: {_e}")


if __name__ == "__main__":
    main()