"""
Generate WM-planned trajectories for BC training.

Usage:
python generate_planned_trajectories.py model_name=<run> ckpt_base_path=<abs> \\
planner=cem goal_H=5 num_trajectories=dataset sample_mode=sequential \\
shard_every=50 planned_out_path=./planned_pusht.pth
# plan to each demo's final frame instead of a fixed goal_H (closed-loop)
python generate_planned_trajectories.py planner=mpc_cem plan_to_end=true ...
"""
import os
import json
# MuJoCo defaults to the GLFW backend, which needs an X display and fails on a headless box.
os.environ.setdefault("MUJOCO_GL", "egl")
# Set before any CUDA init so torch.use_deterministic_algorithms() can make cuBLAS matmuls
# deterministic when deterministic=true, harmless otherwise, respects a shell-set value.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import gym
import time
import tqdm
import torch
import hydra
import random
import numpy as np
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
from utils import cfg_to_dict, seed as set_seed
from plan import load_model, PlanWorkspace

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False


class _ConcatTrajDataset:
    """Concatenate raw trajectory datasets (source_split=both). Routes indexing, length,
    and frame reads to the right sub-dataset, dataset-level stats (means/std, action_dim,
    transform) come from the first, train and val share the same normalization."""

    def __init__(self, dsets):
        self.dsets = list(dsets)
        self._cum = [0]
        for d in self.dsets:
            self._cum.append(self._cum[-1] + len(d))
        base = self.dsets[0]
        for attr in ("action_dim", "state_dim", "proprio_dim", "action_mean", "action_std",
                     "state_mean", "state_std", "proprio_mean", "proprio_std", "transform"):
            if hasattr(base, attr):
                setattr(self, attr, getattr(base, attr))

    def __len__(self):
        return self._cum[-1]

    def _route(self, idx):
        for k in range(len(self.dsets)):
            if idx < self._cum[k + 1]:
                return self.dsets[k], idx - self._cum[k]
        raise IndexError(idx)

    def __getitem__(self, idx):
        d, j = self._route(idx)
        return d[j]

    def get_seq_length(self, idx):
        d, j = self._route(idx)
        return d.get_seq_length(j)

    def get_frames(self, idx, frames):
        d, j = self._route(idx)
        return d.get_frames(j, frames)


def _to_uint8_visual(arr):
    """(T, H, W, C) in [0, 255] (np or tensor) -> uint8 tensor."""
    t = torch.as_tensor(np.asarray(arr))
    if t.dtype != torch.uint8:
        t = t.round().clamp(0, 255).to(torch.uint8)
    return t


def _pad_actions_to(act, length):
    """Pad (T, A) to (length, A) by repeating the last action."""
    if act.shape[0] >= length:
        return act[:length]
    pad = act[-1:].repeat(length - act.shape[0], 1)
    return torch.cat([act, pad], dim=0)


def _wm_rep_features(z_vis, rep):
    """Reduce a WM visual latent (b, [1,] P, D) to the probe rep: pooled (mean over patches) or
    patches (flatten). cls is N/A (the WM predicts patch tokens, not a cls token)."""
    if rep == "pooled":
        return z_vis.mean(dim=-2).reshape(z_vis.shape[0], -1)
    if rep == "patches":
        return z_vis.reshape(z_vis.shape[0], -1)
    raise ValueError(f"probe rep={rep!r}: apply a 'pooled' or 'patches' probe to WM latents "
                     "(the WM has no cls token to probe).")


def _load_probe(path, device):
    """Load a train_probe.py checkpoint as a callable feat -> raw state. Its `rep` (pooled | patches)
    decides how the WM latent is reduced before applying it, the probe must be trained on the same
    encoder/img_size the WM uses. Returns (decode_fn, in_dim, out_dim, rep, state_names)."""
    from train_probe import Probe
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["model"]
    # infer dims/shape from the saved Sequential, Dropout at idx0, Linear at idx1[, GELU, Linear
    # idx4]
    if "net.4.weight" in sd:   # 1-hidden-layer MLP probe
        in_dim, hidden, out_dim = sd["net.1.weight"].shape[1], sd["net.1.weight"].shape[0], sd["net.4.weight"].shape[0]
    else:   # linear probe (recommended)
        in_dim, hidden, out_dim = sd["net.1.weight"].shape[1], 0, sd["net.1.weight"].shape[0]
    probe = Probe(in_dim, out_dim, hidden, 0.0).to(device).eval()
    probe.load_state_dict(sd)
    smean = ck["state_mean"].to(device).float()
    sstd = ck["state_std"].to(device).float()
    rep = ck.get("rep", "pooled")

    @torch.no_grad()
    def decode(feat):   # feat (b, in_dim) -> raw state (b, out_dim)
        return (probe(feat.to(device).float()) * sstd + smean)

    return decode, in_dim, out_dim, rep, ck.get("state_names")


@torch.no_grad()
def _dream_rollout(model, workspace, plan_actions, device, probe_decode, probe_rep):
    """Roll the WM OPEN-LOOP with the planned actions (no sim) -> the imagined trajectory.
    Returns (latents (b,T,P,D) cpu, decoded_visual (b,T,H,W,C) uint8 or None if no WM decoder,
    states (b,T,S) cpu or None if no probe). T == one entry per WM step (frameskip sim-frames)."""
    ev = workspace.evaluator
    tob = ev.preprocessor.transform_obs(ev.obs_0)
    tob = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in tob.items()}
    i_z, _ = model.rollout(tob, plan_actions.detach())
    lat = i_z["visual"].detach()   # , b, T, P, D
    b, T = lat.shape[0], lat.shape[1]
    vis = None
    if getattr(model, "decoder", None) is not None:
        dec = model.decode_obs(i_z)[0]["visual"].clamp(0, 1)   # , b, T, 3, H, W
        vis = (dec.permute(0, 1, 3, 4, 2) * 255).byte().cpu()   # , b, T, H, W, C
    states = None
    if probe_decode is not None:
        feat = _wm_rep_features(lat.reshape(b * T, *lat.shape[2:]), probe_rep)   # (b*T, in_dim)
        states = probe_decode(feat).reshape(b, T, -1).cpu()   # (b, T, S)
    return lat.cpu(), vis, states


def _build_specs(dset, cfg_dict, frameskip):
    """Decide which trajectories/goals to generate from the TRAIN dataset.

    num_trajectories: null -> n_planning_rounds*n_evals, 'dataset' -> len(train), int -> exact.
    """
    N = len(dset)
    goal_H = int(cfg_dict["goal_H"])
    plan_to_end = bool(cfg_dict.get("plan_to_end", False))
    sample_mode = str(cfg_dict.get("sample_mode", "random"))
    n_evals = int(cfg_dict["n_evals"])

    num_traj = cfg_dict.get("num_trajectories", None)
    if num_traj in ("dataset", "match", "all"):
        total = N
    elif num_traj is None:
        total = int(cfg_dict.get("n_planning_rounds", 1)) * n_evals
    else:
        total = int(num_traj)

    # shared seeded-random segment selection, matches BC's original_* selection so generation
    # plans from exactly the windows BC-on-original trains on.
    seg_seed = cfg_dict.get("original_subset_seed", None)
    seg_len_cfg = cfg_dict.get("original_segment_length", None)
    if not plan_to_end and (seg_seed is not None or seg_len_cfg is not None):
        from datasets.policy_dataset import select_original_segments
        plan_len = frameskip * goal_H   # env-steps the planned window spans
        # pass the same segment_length BC uses, None -> full trajectories, not plan_len, otherwise
        # the `valid` pool and the rng stream, offset draw vs none diverge and the two scripts
        # pick different segments.
        seg_len = int(seg_len_cfg) if seg_len_cfg is not None else None
        if seg_len is not None and seg_len < plan_len + 1:
            raise ValueError(
                f"original_segment_length={seg_len} < frameskip*goal_H+1={plan_len + 1}; the goal "
                f"frame (off+{plan_len}) would fall outside the segment. Set it >= {plan_len + 1} "
                f"(ideally == {plan_len + 1}, so BC's window == the planned window).")
        segs = select_original_segments(
            dset, max_trajectories=cfg_dict.get("original_max_trajectories", None),
            segment_length=seg_len, subset_seed=seg_seed)
        specs = []
        for i in range(total):
            tid, off, _L = segs[i % len(segs)]
            # goal is the frame at off+plan_len, so it must be a valid index (<= seq_len-1) this
            # needs original_segment_length >= plan_len+1, one more than the planned span.
            if off + plan_len >= int(dset.get_seq_length(tid)):
                raise ValueError(
                    f"trajectory {tid} (len {int(dset.get_seq_length(tid))}) has no goal frame at "
                    f"off={off}+{plan_len}. Set original_segment_length >= {plan_len + 1} "
                    f"(== frameskip*goal_H+1).")
            specs.append((tid, off, off + plan_len))
        return specs

    # source_split=eval_pairs, draw from only the (init, goal) pairs the BC eval samples, so the
    # planned data covers the evaluated tasks (overfit/upper-bound experiment).
    if not plan_to_end and str(cfg_dict.get("source_split", "train")) == "eval_pairs":
        eval_pairs = _reproduce_eval_pairs(dset, cfg_dict, frameskip, goal_H)
        plan_len = frameskip * goal_H
        # default count = cover each distinct eval task once, an int num_trajectories overrides
        # (e.g.
        pool_total = len(eval_pairs) if num_traj in ("dataset", "match", "all", None) else int(num_traj)
        rng = random.Random(cfg_dict["seed"])
        if sample_mode == "sequential":
            picks = [eval_pairs[i % len(eval_pairs)] for i in range(pool_total)]
        elif sample_mode == "random":
            pool = list(eval_pairs)
            rng.shuffle(pool)
            picks = pool[:pool_total]
            if pool_total > len(pool):   # more requested than distinct, fill w/ repeats
                picks += [rng.choice(eval_pairs) for _ in range(pool_total - len(pool))]
        elif sample_mode == "random_with_replacement":
            picks = [rng.choice(eval_pairs) for _ in range(pool_total)]
        else:
            raise ValueError("source_split=eval_pairs supports sample_mode 'sequential' | 'random' | "
                             f"'random_with_replacement', got {sample_mode!r}.")
        specs = [(tid, io, io + plan_len) for (tid, io) in picks]
        print(f"[eval_pairs] {len(eval_pairs)} distinct eval tasks -> {len(specs)} planned specs "
              f"(sample_mode={sample_mode}, plan_len={plan_len})")
        return specs

    # a trajectory must be long enough to contain init + goal
    min_len = 2 if plan_to_end else frameskip * goal_H + 1
    valid_ids = [i for i in range(N) if dset.get_seq_length(i) >= min_len]
    # restrict planning to the first `source_max_trajectories` source demos (null = all), so
    # generation can run from a fixed small set (e.g.
    src_cap = cfg_dict.get("source_max_trajectories", None)
    if src_cap is not None:
        valid_ids = [i for i in valid_ids if i < int(src_cap)]
    if not valid_ids:
        raise ValueError(
            f"No training trajectory is long enough (need >= {min_len} frames for "
            f"goal_H={goal_H}, plan_to_end={plan_to_end}) within source_max_trajectories="
            f"{src_cap}."
        )

    def goal_offset(tid, init_off):
        return (dset.get_seq_length(tid) - 1) if plan_to_end else init_off + frameskip * goal_H

    def init_offset(rng, tid):
        if plan_to_end:
            return 0
        return rng.randint(0, dset.get_seq_length(tid) - frameskip * goal_H - 1)

    specs = []
    if sample_mode == "sequential":
        i = 0
        while len(specs) < total:
            tid = valid_ids[i % len(valid_ids)]
            specs.append((tid, 0, goal_offset(tid, 0)))
            i += 1
    elif sample_mode == "random":
        rng = random.Random(cfg_dict["seed"])
        seen = set()
        max_attempts = max(1000, total * 50)
        attempts = 0
        while len(specs) < total and attempts < max_attempts:
            attempts += 1
            tid = rng.choice(valid_ids)
            io = init_offset(rng, tid)
            if (tid, io) in seen:
                continue
            seen.add((tid, io))
            specs.append((tid, io, goal_offset(tid, io)))
        if len(specs) < total:   # ran out of distinct pairs, fill with repeats
            print(f"warning: only {len(specs)} distinct (traj, offset) pairs available; "
                  f"requested {total}. Filling the remainder with repeats.")
            rng2 = random.Random(cfg_dict["seed"] + 1)
            while len(specs) < total:
                tid = rng2.choice(valid_ids)
                io = init_offset(rng2, tid)
                specs.append((tid, io, goal_offset(tid, io)))
    elif sample_mode == "random_with_replacement":
        # fully random, each (traj, offset) is drawn independently with replacement, so duplicates
        # are allowed, the exact dino-wm sampling, no dedup.
        rng = random.Random(cfg_dict["seed"])
        for _ in range(total):
            tid = rng.choice(valid_ids)
            io = init_offset(rng, tid)
            specs.append((tid, io, goal_offset(tid, io)))
    else:
        raise ValueError(f"Unknown sample_mode: {sample_mode!r} (use 'random', "
                         f"'random_with_replacement', or 'sequential')")
    return specs[:total]


def _reproduce_eval_pairs(dset, cfg_dict, frameskip, goal_H):
    """Distinct (traj_id, init_off) pairs the BC eval samples (train_policy.py:1034):
    per-episode rng=Random(eval_seed+episode_idx), traj=rng.choice(valid_ids), init_off=rng.randint(0, seq_len-plan_len-1). `dset` must be the split the eval draws goals from
    (source_split=eval_pairs loads it via cfg eval_split, default valid). eval_goal_offset, if set,
    must equal frameskip*goal_H so the planned goal frame IS the eval goal frame."""
    eval_seed = cfg_dict.get("eval_seed", None)
    n_eval = cfg_dict.get("n_eval_episodes", None)
    if eval_seed is None or n_eval is None:
        raise ValueError("source_split=eval_pairs needs eval_seed and n_eval_episodes (the BC run's "
                         "cfg.seed and n_env_evals = (n_env_evals // n_envs) * n_envs).")
    plan_len = frameskip * goal_H
    eval_goal_off = cfg_dict.get("eval_goal_offset", None)
    if eval_goal_off is not None and int(eval_goal_off) != plan_len:
        raise ValueError(f"eval_goal_offset={eval_goal_off} must equal frameskip*goal_H={plan_len} "
                         "so the planned goal frame is the eval goal frame.")
    valid_ids = [i for i in range(len(dset)) if dset.get_seq_length(i) >= plan_len + 1]
    if not valid_ids:
        raise ValueError(f"no trajectory is long enough (>= {plan_len + 1} frames) for eval_pairs.")
    seen, pairs = set(), []
    for e in range(int(n_eval)):
        rng = random.Random(int(eval_seed) + e)   # per-episode, exactly as the eval
        tid = rng.choice(valid_ids)
        io = rng.randint(0, max(0, dset.get_seq_length(tid) - plan_len - 1))
        if (tid, io) in seen:
            continue
        seen.add((tid, io))
        pairs.append((tid, io))
    return pairs


def generate(cfg_dict):
    ckpt_base_path = cfg_dict["ckpt_base_path"]
    model_path = f"{ckpt_base_path}/outputs/{cfg_dict['model_name']}/"
    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)

    # optionally override the dataset path baked into the WM's saved config.
    override_path = cfg_dict.get("override_dataset_path")
    if override_path:
        with open_dict(model_cfg):
            model_cfg.env.dataset.data_path = override_path
        print(f"Overriding WM dataset path -> {override_path}")

    # build the ENV *before* any CUDA work SubprocVectorEnv uses multiprocessing's default start
    # method, which on Linux is FORK (env/venv.py:405).
    env_kwargs = OmegaConf.to_container(model_cfg.env.kwargs, resolve=True)
    if cfg_dict.get("env_kwargs"):
        env_kwargs.update(dict(cfg_dict["env_kwargs"]))
        print(f"[env] kwargs overridden with {dict(cfg_dict['env_kwargs'])} -> {env_kwargs}")

    # serial_env=true -> build every env IN this process (env/serial_vector_env.py), no
    # multiprocessing at all.
    serial_env = bool(cfg_dict.get("serial_env", False))

    # Plain-Python copies of everything the env constructor needs.
    _env_name = str(model_cfg.env.name)
    _env_args = list(model_cfg.env.args or [])
    _n_envs = int(cfg_dict["n_evals"])
    # 'fork' (default) | 'spawn' | 'forkserver'.
    start_method = cfg_dict.get("env_start_method", None)

    def make_env():
        if serial_env or _env_name in ("wall", "deformable_env"):
            from env.serial_vector_env import SerialVectorEnv
            if serial_env and _env_name not in ("wall", "deformable_env"):
                print(f"[env] serial_env=true -> {_n_envs} envs IN-PROCESS (no subprocesses at "
                      f"all). Correct everywhere, but steps run one after another. Prefer "
                      f"env_start_method=spawn, which keeps the parallelism.")
            return SerialVectorEnv(
                [gym.make(_env_name, *_env_args, **env_kwargs) for _ in range(_n_envs)]
            )
        if start_method:
            print(f"[env] {_n_envs} envs in subprocesses via start_method={start_method}"
                  + ("  (fresh interpreters -> each initialises its own EGL display)"
                     if start_method == "spawn" else ""))
        return SubprocVectorEnv(
            [lambda: gym.make(_env_name, *_env_args, **env_kwargs) for _ in range(_n_envs)],
            start_method=start_method,
        )

    env = make_env()
    # Always release the envs, including on the crash path, generate()'s own env.close() only runs
    # if the planning loop completes.
    import atexit
    atexit.register(lambda: env.close())

    # Only now touch the GPU, torch.cuda.is_available()/get_device_name initialise the CUDA
    # driver, and the workers above must be forked before that happens, see the note above.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # make the device explicit, a silent CPU fallback, no GPU visible / wrong env is a common
    # cause of multi-hour generation, so surface it loudly.
    if torch.cuda.is_available():
        print(f"[device] using {device} -> {torch.cuda.get_device_name(0)} "
              f"(torch CUDA {torch.version.cuda})")
    else:
        print("[device] WARNING: CUDA not available -- running on CPU. This will be "
              "10-50x slower. Check the conda env / GPU allocation (nvidia-smi).")

    set_seed(cfg_dict["seed"])
    # Opt-in determinism (deterministic=true), pin cuDNN + force deterministic algorithms so the
    # WM rollouts inside CEM are reproducible -> two same-seed runs produce the same plans, and
    # thus the same planned dataset.
    if bool(cfg_dict.get("deterministic", False)):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("[determinism] ON: cuDNN deterministic + use_deterministic_algorithms(warn_only). "
              "Slower; same-seed generation should now match closely.")
    _, dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    # which env-dataset split to sample (init, goal) pairs from for planning, val/both plan from
    # held-out episodes.
    #   train (default) | val | both (concatenate train+val).
    source_split = str(cfg_dict.get("source_split", "train"))
    if source_split == "train":
        dset = dset["train"]
    elif source_split in ("val", "valid"):
        dset = dset["valid"]
    elif source_split == "both":
        dset = _ConcatTrajDataset([dset["train"], dset["valid"]])
    elif source_split == "eval_pairs":
        # draw only the eval's (init, goal) pairs (see _build_specs).
        eval_split = str(cfg_dict.get("eval_split", "valid"))
        dset = dset["train"] if eval_split == "train" else dset["valid"]
    else:
        raise ValueError(f"source_split must be 'train', 'val', 'both', or 'eval_pairs', "
                         f"got {source_split!r}")
    print(f"[source_split={source_split}] sampling from {len(dset)} trajectories")

    frameskip = model_cfg.frameskip
    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    model = load_model(model_ckpt, model_cfg, num_action_repeat, device=device)

    # optional, load a pooled position probe (train_probe.py checkpoint).
    probe_decode = None
    probe_rep = None
    probe_in_dim = None
    probe_names = None
    probe_ckpt = cfg_dict.get("probe_ckpt", None)
    if probe_ckpt:
        probe_decode, probe_in_dim, _p_out, probe_rep, probe_names = _load_probe(probe_ckpt, device)
        print(f"[probe] loaded {probe_ckpt} (rep={probe_rep}, in_dim={probe_in_dim}, out_dim={_p_out}) "
              f"-> logging WM-believed vs sim-real final state per round. components={probe_names}")

    dream = bool(cfg_dict.get("dream", False))
    if dream:
        _has_dec = getattr(model, "decoder", None) is not None
        print(f"[dream] SIM-FREE imagination ON: rolling the WM open-loop with the planned actions, "
              f"storing the imagined trajectory (decoder={'yes -> images' if _has_dec else 'no -> latents only'}). "
              f"Each frame = 1 WM step (frameskip={frameskip} sim-frames); train BC at frameskip=1 on this.")
        if not _has_dec:
            print("[dream] NOTE: no WM decoder -> only latents are stored ('dreamed_latents'); BC must "
                  "consume latents directly (latent path in train_policy), not images.")
        if bool(cfg_dict.get("planned_only_successes", False)):
            print("[dream] dreamed trajectories have no sim success, so only_successes is ignored.")

    # progress log in the Hydra run dir (plan_outputs/<ts>/), so the run dir always has visible,
    # growing output even when figures/videos are disabled.
    run_dir = os.getcwd()
    log_path = os.path.join(run_dir, "generation.log")

    def _log(msg):
        print(msg)
        with open(log_path, "a") as f:
            f.write(str(msg) + "\n")

    # wandb, on by default, logs per-round success/reached + timing.
    wandb_run = None
    if bool(cfg_dict.get("wandb_logging", True)) and _HAS_WANDB:
        wcfg = cfg_dict.get("wandb", {}) or {}
        # name/group/job_type/tags come from the config when set, else fall back to the Hydra run
        # folder.
        wandb_run = wandb.init(
            project=wcfg.get("project", "dino_wm_planning"),
            entity=wcfg.get("entity", None),
            name=wcfg.get("name") or os.path.basename(run_dir),
            group=wcfg.get("group") or None,
            job_type=wcfg.get("job_type") or None,
            tags=list(wcfg.get("tags") or []) or None,
            config=cfg_dict,
        )
        # Plot the per-round metrics against `round`, not wandb's internal step counter.
        wandb_run.define_metric("round")
        for _m in ("success_rate", "reached_cumulative", "trajectories_done",
                   "steps/*", "steps_to_success/*", "time/*", "wm/*", "wm_comp/*"):
            wandb_run.define_metric(_m, step_metric="round")

    only_successes = bool(cfg_dict.get("planned_only_successes", False))
    # how often the evaluator saves visuals, in the Hydra run dir.
    fig_every = int(cfg_dict.get("save_figure_every", 0))
    vid_every = int(cfg_dict.get("save_video_every", 0))
    n_evals = int(cfg_dict["n_evals"])
    plan_to_end = bool(cfg_dict.get("plan_to_end", False))

    # random_state mode, the planner samples random init/goal states straight from the env, wall,
    # init on one side of the wall, goal on the other, the exact dino-wm random_state behavior, so
    # there are no demo (traj_id, offset) specs to build.
    is_random_state = (not plan_to_end) and str(cfg_dict.get("goal_source", "dset")) == "random_state"

    # provided_states_path, a .pth holding explicit {init_states (N, S), goal_states (N, S)
    # env_infos, list of N dicts or []} to plan from, the DAgger relabel path.
    provided_path = cfg_dict.get("provided_states_path", None)
    provided = None
    if provided_path:
        if plan_to_end:
            raise ValueError("provided_states_path and plan_to_end are mutually exclusive "
                             "(both supply the goal).")
        provided = torch.load(provided_path, map_location="cpu", weights_only=False)
        _pi = np.asarray(provided["init_states"])
        _pg = np.asarray(provided["goal_states"])
        if _pi.shape[0] != _pg.shape[0]:
            raise ValueError(f"provided_states_path: {_pi.shape[0]} init_states but "
                             f"{_pg.shape[0]} goal_states; they must pair up.")
        # env_infos is optional and may be a list of Nones (envs with no per-trajectory layout
        # e.g.
        _ei = list(provided.get("env_infos") or [])
        if not _ei or any(e is None for e in _ei) or len(_ei) != _pi.shape[0]:
            _ei = []
        provided = {"init_states": _pi, "goal_states": _pg, "env_infos": _ei}
        is_random_state = False
        _log(f"[provided] planning from {_pi.shape[0]} caller-supplied (init, goal) pairs "
             f"({provided_path})")

    # PlanWorkspace's 'provided' path calls env.update_env(env_info) unconditionally, and
    # SerialVectorEnv.update_env indexes env_info[i] for every env, so an empty list raises
    # IndexError.
    default_env_info = None
    if provided is not None and not provided["env_infos"]:
        default_env_info = dset.get_frames(0, [0])[3]
        _log(f"[provided] no per-pair env_info given -> reusing the dataset's layout "
             f"({model_cfg.env.name}) for every pair")

    # MPC replans until every trajectory succeeds (mpc.py, `while not np.all(self.is_success)`) so
    # with max_iter=null an UNREACHABLE goal loops forever.
    is_mpc = isinstance(cfg_dict.get("planner"), dict) and "sub_planner" in cfg_dict["planner"]
    if plan_to_end or is_random_state:
        _why = "plan_to_end" if plan_to_end else "goal_source=random_state"
        if is_mpc and cfg_dict["planner"].get("max_iter") is None:
            cfg_dict["planner"]["max_iter"] = int(cfg_dict.get("mpc_max_iter", 20))
            _log(f"[{_why}] MPC max_iter was null (unbounded) -> capped at "
                 f"{cfg_dict['planner']['max_iter']} replans (mpc_max_iter). Trajectories that "
                 f"hit the cap are saved with success=False.")
        if not is_mpc and plan_to_end:
            _log("WARNING: plan_to_end=true but planner is not MPC; an open-loop planner "
                 "cannot reach a far goal. Use planner=mpc_cem.")
        if not is_mpc and is_random_state:
            _log("WARNING: goal_source=random_state with a non-MPC planner: actions are optimized "
                 "ONCE open-loop from the initial observation, so there is no real-env feedback "
                 "between replans (writeup 3.6a-c). Use planner=mpc_cem for closed-loop generation.")
    if provided is not None:
        specs = None
        total = int(provided["init_states"].shape[0])   # exactly the pairs the caller gave
    elif is_random_state:
        specs = None
        num_traj = cfg_dict.get("num_trajectories", None)
        if num_traj in ("dataset", "match", "all"):
            total = len(dset)
        elif num_traj is None:
            total = int(cfg_dict.get("n_planning_rounds", 1)) * n_evals
        else:
            total = int(num_traj)
    else:
        # pick which trajectories/goals to generate (train split only)
        specs = _build_specs(dset, cfg_dict, frameskip)
        total = len(specs)
    n_rounds = (total + n_evals - 1) // n_evals

    visual, actions_out, states_out, proprios_out, seq_lengths = [], [], [], [], []
    # per-trajectory, planned initial state, goal state, and whether the goal was reached.
    init_states_out, goal_states_out, successes_out = [], [], []
    # WM-believed final state per trajectory, probe of the imagined final latent, only filled when
    # a probe_ckpt is set.
    wm_pred_final_states_out = []
    # imagined WM latents per trajectory (dream mode only), list of (T, P, D) tensors.
    dreamed_latents_out = []
    base_seed = cfg_dict["seed"]

    # incremental sharded save, with shard_every > 0, flush accumulated trajectories to a shard
    # file every N rounds and free the RAM, so large runs don't OOM.
    shard_every = int(cfg_dict.get("shard_every", 0))
    out_path = os.path.abspath(cfg_dict.get("planned_out_path", "./planned_trajectories.pth"))
    sharded = shard_every > 0
    if sharded:
        shard_dir = out_path[:-4] if out_path.endswith(".pth") else out_path
        os.makedirs(shard_dir, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    counters = {"shard_idx": 0, "total": 0, "total_success": 0}

    def _payload():
        return {
            "visual": visual,
            "actions": actions_out,
            "states": states_out,   # full initial->...->final state per step
            "proprios": proprios_out,
            "seq_lengths": seq_lengths,
            "init_states": init_states_out,   # planned initial state per trajectory
            "goal_states": goal_states_out,   # planning goal state per trajectory
            "successes": successes_out,   # whether the goal was reached
            "wm_pred_final_states": wm_pred_final_states_out,   # WM-believed final state (probe), [] if no probe
            "dreamed_latents": dreamed_latents_out,   # (dream mode) imagined WM latents per traj, [] otherwise
            "dreamed": dream,   # true = sim-free imagined data (frame == 1 WM step)
            "env_name": model_cfg.env.name,
            "frameskip": frameskip,
        }

    def _flush_shard():
        if len(seq_lengths) == 0:
            return
        path = os.path.join(shard_dir, f"shard_{counters['shard_idx']:04d}.pth")
        torch.save(_payload(), path)
        counters["total"] += len(seq_lengths)
        counters["total_success"] += int(sum(1 for s in successes_out if s))   # None (dream) -> not a success
        _log(f"  shard {counters['shard_idx']}: saved {len(seq_lengths)} trajectories -> {path}")
        counters["shard_idx"] += 1
        for lst in (visual, actions_out, states_out, proprios_out, seq_lengths,
                    init_states_out, goal_states_out, successes_out, wm_pred_final_states_out,
                    dreamed_latents_out):
            lst.clear()

    _log(f"generation start: env={model_cfg.env.name} trajectories={total} rounds={n_rounds} "
         f"n_evals={n_evals} plan_to_end={plan_to_end} shard_every={shard_every} "
         f"out={shard_dir if sharded else out_path}")

    pbar = tqdm.trange(n_rounds, desc="planning rounds", unit="round")
    run_seen, run_reached = 0, 0   # running totals for the progress bar
    wm_err_run = {"sum": 0.0, "n": 0}   # cumulative WM endpoint-error mean (probe_ckpt only)
    # cumulative trajectory-length stats, kept separately from seq_lengths because that list is
    # cleared on every shard flush
    steps_run = {"sum": 0, "n": 0, "succ_sum": 0.0, "succ_n": 0}
    # cumulative wall-clock per phase (seconds), setup = PlanWorkspace build, samples init/goal
    # from the dset, plan = the CEM optimization, eval = execute + store.
    t_setup_total, t_plan_total, t_eval_total = 0.0, 0.0, 0.0
    t_gen_start = time.perf_counter()
    for r in pbar:
        if provided is not None:
            # this round takes the r-th block of caller-supplied pairs, the last block is padded
            # up to n_evals for the vector env and trimmed back at store time.
            lo, hi = r * n_evals, min((r + 1) * n_evals, total)
            real = hi - lo
        elif is_random_state:
            # no demo specs, every round plans a full batch of n_evals random init/goal states
            # except the last round is trimmed to hit `total` exactly.
            real = min(n_evals, total - r * n_evals)
        else:
            chunk = specs[r * n_evals: (r + 1) * n_evals]
            real = len(chunk)
            if real < n_evals:   # pad the last partial batch for the vector env, trimmed at store
                chunk = chunk + [chunk[-1]] * (n_evals - real)

        round_cfg = dict(cfg_dict)
        round_cfg["seed"] = base_seed + r * 1000   # vary seeds per round
        # eval_seed_base (optional), hand PlanWorkspace a contiguous block of episode indices for
        # this round, so episode seeds are globally unique across rounds and across successive
        # invocations of this script.
        if cfg_dict.get("eval_seed_base", None) is not None:
            round_cfg["eval_seed_base"] = int(cfg_dict["eval_seed_base"]) + r * n_evals
        if provided is not None:
            # DAgger relabel, plan from the caller's states toward the caller's goals.
            _pi = provided["init_states"][lo:hi]
            _pg = provided["goal_states"][lo:hi]
            if real < n_evals:
                _pi = np.concatenate([_pi, np.repeat(_pi[-1:], n_evals - real, axis=0)])
                _pg = np.concatenate([_pg, np.repeat(_pg[-1:], n_evals - real, axis=0)])
            round_cfg["goal_source"] = "provided"
            round_cfg["init_states"] = _pi
            round_cfg["goal_states"] = _pg
            # env_info must always be a list of exactly n_evals entries, update_env indexes
            # env_info[i] for every env, so a short list (or []) is an IndexError.
            _ei = provided["env_infos"]
            if _ei:
                _blk = list(_ei[lo:hi])
                _blk += [_blk[-1]] * (n_evals - len(_blk))
            else:
                _blk = [default_env_info] * n_evals
            round_cfg["env_info"] = _blk
        elif is_random_state:
            # PlanWorkspace's random_state path samples n_evals init/goal pairs from the env,
            # opposite sides of the wall off eval_seed, which varies with round_cfg["seed"].
            round_cfg["goal_source"] = "random_state"
        elif plan_to_end:
            # provide explicit init/goal states (goal = demo final frame) for the planner
            init_states, goal_states, env_infos = [], [], []
            for (tid, io, go) in chunk:
                _, _, s0, info0 = dset.get_frames(tid, [io])
                _, _, sg, _ = dset.get_frames(tid, [go])
                init_states.append(np.asarray(s0[0]))
                goal_states.append(np.asarray(sg[0]))
                env_infos.append(info0)
            round_cfg["goal_source"] = "provided"
            round_cfg["init_states"] = np.stack(init_states)
            round_cfg["goal_states"] = np.stack(goal_states)
            round_cfg["env_info"] = env_infos
        else:
            # fixed goal_H, PlanWorkspace's dset path replays the demo's actions to define the
            # goal, we just tell it which (traj, offset) to use.
            round_cfg["goal_source"] = "dset"
            round_cfg["traj_ids"] = [int(c[0]) for c in chunk]
            round_cfg["offsets"] = [int(c[1]) for c in chunk]

        _t0 = time.perf_counter()
        workspace = PlanWorkspace(
            cfg_dict=round_cfg,
            wm=model,
            dset=dset,
            env=env,
            env_name=model_cfg.env.name,
            frameskip=frameskip,
            # feed the generation's wandb run into the planner so its per-opt-step loss (cem.py
            # "plan_*/loss") is actually recorded, not swallowed by DummyWandbRun.
            wandb_run=wandb_run,
        )
        # Keep the per-opt-step evaluator enabled so trajectories freeze on REAL env success (see
        # cem.py), not just the imagined objective.
        for _p in (workspace.planner, getattr(workspace.planner, "sub_planner", None)):
            if _p is not None and hasattr(_p, "eval_save_plot"):
                _p.eval_save_plot = False
        # A planner with no cheap figure-free mode (e.g.
        if not is_mpc and not hasattr(workspace.planner, "eval_save_plot"):
            workspace.planner.evaluator = None
        # give each round its own wandb series (plan_r{r}/loss, plan_r{r}/success_rate) so the
        # per-opt-step curves are separate graphs, instead of every round colliding on one
        # 'plan_0' series whose step axis resets each round.
        if not is_mpc:
            for _p in (workspace.planner, getattr(workspace.planner, "sub_planner", None)):
                if _p is None:
                    continue
                if hasattr(_p, "log_step_offset"):
                    _p.log_step_offset = r * int(getattr(_p, "opt_steps", 30))
                if n_rounds <= 50 and hasattr(_p, "logging_prefix"):
                    _p.logging_prefix = f"plan_{r}"
        _t1 = time.perf_counter()
        plan_actions, action_len = workspace.planner.plan(
            obs_0=workspace.obs_0, obs_g=workspace.obs_g, actions=None
        )
        _t2 = time.perf_counter()
        save_plot = fig_every > 0 and (r % fig_every == 0)
        save_video = vid_every > 0 and (r % vid_every == 0)
        wm_final_states = None   # (b, S) WM-believed final state
        d_lat = d_vis = d_states = None   # dream-mode imagined trajectory
        if dream:
            # no sim, roll the WM open-loop with the planned actions and store the imagined traj.
            d_lat, d_vis, d_states = _dream_rollout(model, workspace, plan_actions, device,
                                                    probe_decode, probe_rep)
            logs = {"success_rate": None}
            successes = [False] * n_evals   # no sim truth, stored as None below
            e_obses = e_states = None
        elif probe_decode is not None:
            logs, successes, e_obses, e_states, i_final_z = workspace.evaluator.eval_actions(
                plan_actions.detach(), action_len, save_video=save_video, save_plot=save_plot,
                filename=f"planned_round{r}", return_imagined=True,
            )
            # imagined final latent -> probe rep (pooled | patches) -> probe -> raw state
            feat_final = _wm_rep_features(i_final_z["visual"], probe_rep)   # (b, in_dim)
            if feat_final.shape[1] != probe_in_dim:
                raise ValueError(
                    f"probe in_dim={probe_in_dim} != WM {probe_rep} feature dim {feat_final.shape[1]}; "
                    "the probe was trained on a different encoder/img_size/rep than this WM produces.")
            wm_final_states = probe_decode(feat_final).cpu().numpy()   # (b, S)
        else:
            logs, successes, e_obses, e_states = workspace.evaluator.eval_actions(
                plan_actions.detach(),
                action_len,
                save_video=save_video,
                save_plot=save_plot,
                filename=f"planned_round{r}",
            )
        _t3 = time.perf_counter()
        t_setup, t_plan, t_eval = _t1 - _t0, _t2 - _t1, _t3 - _t2
        t_setup_total += t_setup; t_plan_total += t_plan; t_eval_total += t_eval
        _log(f"[round {r}/{n_rounds}] success_rate={logs.get('success_rate')} | "
             f"plan={t_plan:.1f}s setup={t_setup:.1f}s eval={t_eval:.1f}s "
             f"({n_evals} trajs, {t_plan / max(1, n_evals):.2f}s/traj planning)")

        # planner actions, (b, H, f*d) normalized -> per-step (b, H*f, d)
        per_step = rearrange(plan_actions.detach().cpu(), "b t (f d) -> b (t f) d", f=frameskip)

        succ = np.asarray(successes).reshape(-1)   # per-env success tag
        state_0 = np.asarray(workspace.state_0)   # (b, d) planned init states
        state_g = np.asarray(workspace.state_g)   # (b, d) goal states
        # mark where this round's entries start so the length stats below cover only this round,
        # seq_lengths accumulates across rounds and is cleared on each shard flush
        _n_before = len(seq_lengths)
        for i in range(real):   # only the real trajectories (skip padding)
            if dream:
                # imagined trajectory, 1 entry per WM step, actions are the frameskip-chunk
                # (f*adim).
                T = int(d_lat.shape[1])
                act_i = _pad_actions_to(plan_actions[i].detach().cpu().float(), T)   # (T, f*adim)
                state_i = (d_states[i].float() if d_states is not None
                           else torch.zeros(T, int(state_g.shape[1])))
                if d_vis is not None:   # decoder present -> also store images
                    visual.append(_to_uint8_visual(d_vis[i]))
                dreamed_latents_out.append(d_lat[i].clone())
                actions_out.append(act_i)
                states_out.append(state_i)
                proprios_out.append(torch.zeros(T, 1))
                seq_lengths.append(T)
                init_states_out.append(torch.as_tensor(state_0[i]).float())
                goal_states_out.append(torch.as_tensor(state_g[i]).float())
                successes_out.append(None)   # no sim truth
                continue
            success_i = bool(succ[i])
            if only_successes and not success_i:
                continue
            vis_i = _to_uint8_visual(e_obses["visual"][i])   # , T, H, W, C
            T = vis_i.shape[0]
            act_i = _pad_actions_to(per_step[i].float(), T)
            prop = e_obses.get("proprio")   # not all envs expose proprio, policy is image-only
            prop_i = (torch.as_tensor(np.asarray(prop[i])).float()
                      if prop is not None else torch.zeros(T, 1))
            state_i = torch.as_tensor(np.asarray(e_states[i])).float()
            visual.append(vis_i)
            actions_out.append(act_i)
            states_out.append(state_i)
            proprios_out.append(prop_i)
            seq_lengths.append(int(T))
            init_states_out.append(torch.as_tensor(state_0[i]).float())   # == state_i[0]
            goal_states_out.append(torch.as_tensor(state_g[i]).float())   # planning target
            successes_out.append(success_i)
            if wm_final_states is not None:
                wm_pred_final_states_out.append(torch.as_tensor(wm_final_states[i]).float())

        # trajectory LENGTH stats for this round. A stored trajectory has T frames covering T-1
        # executed env actions, so "steps" = T-1.
        #   plain CEM  -> every trajectory is horizon*frameskip steps (one open-loop plan), so
        #   MPC  -> length grows with the number of replans, so the mean is the real
        _round_T = seq_lengths[_n_before:]
        _step_logs = {}   # this ROUND's stats -> wandb, so they plot as a per-round curve
        if _round_T:
            _steps = np.asarray([int(t) - 1 for t in _round_T], dtype=float)
            steps_run["sum"] += float(_steps.sum())
            steps_run["n"] += int(_steps.size)
            _step_logs.update({
                "steps/mean": float(_steps.mean()),
                "steps/median": float(np.median(_steps)),
                "steps/min": float(_steps.min()),
                "steps/max": float(_steps.max()),
                "steps/std": float(_steps.std()),
                "steps/n": int(_steps.size),
                # running mean over the whole run, for a smooth line next to the noisy per-round
                # one
                "steps/mean_cumulative": steps_run["sum"] / steps_run["n"],
            })
            _msg = (f"[round {r}] steps/trajectory: mean={_steps.mean():.1f} "
                    f"median={np.median(_steps):.1f} min={_steps.min():.0f} "
                    f"max={_steps.max():.0f} std={_steps.std():.1f} (n={_steps.size})")
            _al = np.asarray(action_len, dtype=float).reshape(-1)[:real]
            _fin = _al[np.isfinite(_al)]
            if _fin.size:
                _succ = _fin * frameskip
                steps_run["succ_sum"] += float(_succ.sum())
                steps_run["succ_n"] += int(_succ.size)
                _step_logs.update({
                    "steps_to_success/mean": float(_succ.mean()),
                    "steps_to_success/median": float(np.median(_succ)),
                    "steps_to_success/min": float(_succ.min()),
                    "steps_to_success/max": float(_succ.max()),
                    "steps_to_success/n": int(_succ.size),
                    "steps_to_success/mean_cumulative": (steps_run["succ_sum"]
                                                         / steps_run["succ_n"]),
                })
                _msg += (f" | steps_to_success: mean={_succ.mean():.1f} "
                         f"median={np.median(_succ):.1f} min={_succ.min():.0f} "
                         f"max={_succ.max():.0f} (n={_succ.size})")
            _log(_msg)

        # WM endpoint self-delusion, how far the WM thinks it ended from where it ACTUALLY ended,
        # real sim final state, and its optimism gap to the goal.
        wm_vs_real = wm_vs_goal = real_vs_goal = None
        wm_comp_log = None
        if wm_final_states is not None:
            pos = {"pusht": [0, 1, 2, 3, 4], "wall": [0, 1, 2, 3],
                   "point_maze": [0, 1]}.get(model_cfg.env.name, None)
            if pos is None and str(model_cfg.env.name) == "puzzle":
                # Puzzle state is qpos | qvel | button_states, and only the BUTTONS are task
                # state.
                _et = str(OmegaConf.select(model_cfg, "env.kwargs.env_type") or "3x3")
                _r, _c = (int(v) for v in _et.split("x"))
                pos = list(range(wm_final_states.shape[1] - _r * _c, wm_final_states.shape[1]))
            wf = wm_final_states[:real]
            rf = np.stack([np.asarray(e_states[i])[-1] for i in range(real)])   # real final (b, S)
            sg = state_g[:real]
            S = wf.shape[1]
            sl = pos if pos is not None else list(range(S))
            wm_vs_real = float(np.linalg.norm((wf - rf)[:, sl], axis=1).mean())   # endpoint error
            wm_vs_goal = float(np.linalg.norm((wf - sg)[:, sl], axis=1).mean())   # WM thinks-it-reached
            real_vs_goal = float(np.linalg.norm((rf - sg)[:, sl], axis=1).mean())   # real distance
            wm_err_run["sum"] += wm_vs_real * real   # weighted running mean over trajectories
            wm_err_run["n"] += real
            # per-component MEAN ABSOLUTE differences, all dims, incl velocity, like train_probe
            # MAE
            comp_err = np.abs(wf - rf).mean(0)   # WM-believed vs real, per component (S,)
            comp_wm_goal = np.abs(wf - sg).mean(0)   # WM-believed vs goal
            comp_real_goal = np.abs(rf - sg).mean(0)   # real vs goal
            names = (list(probe_names) if probe_names is not None and len(probe_names) == S
                     else [f"s{i}" for i in range(S)])
            wm_comp_log = {}
            for k, nm in enumerate(names):
                wm_comp_log[f"wm_comp/err_{nm}"] = float(comp_err[k])   # believed vs real
                wm_comp_log[f"wm_comp/believed_goal_{nm}"] = float(comp_wm_goal[k])
                wm_comp_log[f"wm_comp/real_goal_{nm}"] = float(comp_real_goal[k])
            _log(f"[round {r}] WM endpoint error (believed vs real)={wm_vs_real:.2f} | "
                 f"WM-vs-goal={wm_vs_goal:.2f} real-vs-goal={real_vs_goal:.2f} "
                 f"(WM {'under' if wm_vs_goal < real_vs_goal else 'over'}estimates distance to goal)")
            _log(f"[round {r}] per-component |WM-believed - real|: "
                 + ", ".join(f"{nm}={comp_err[k]:.2f}" for k, nm in enumerate(names)))

        run_seen += real
        run_reached += int(succ[:real].sum())
        pbar.set_postfix(reached=f"{run_reached}/{run_seen}",
                         last_sr=f"{float(logs.get('success_rate') or 0.0):.2f}",
                         plan=f"{t_plan:.1f}s")
        if wandb_run is not None:
            wlog = {
                "round": r,
                "success_rate": float(logs.get("success_rate") or 0.0),
                "reached_cumulative": run_reached / max(1, run_seen),
                "trajectories_done": run_seen,
                "time/plan_s": t_plan, "time/setup_s": t_setup, "time/eval_s": t_eval,
            }
            # per-round trajectory-length stats, mean/median/min/max/std, plus the running mean
            # computed above.
            wlog.update(_step_logs)
            if wm_vs_real is not None:   # probe_ckpt set, WM belief vs sim reality
                wlog.update({
                    "wm/endpoint_err": wm_vs_real,   # WM-believed final vs real final (per round)
                    "wm/believed_goal_dist": wm_vs_goal,   # WM thinks it's this far from goal
                    "wm/real_goal_dist": real_vs_goal,   # it's actually this far from goal
                    "wm/optimism": real_vs_goal - wm_vs_goal,   # >0 = WM underestimates distance (over-optimistic)
                    "wm/endpoint_err_cumulative": wm_err_run["sum"] / max(1, wm_err_run["n"]),
                })
                if wm_comp_log:   # per-component breakdown (agent_x, block_y...)
                    wlog.update(wm_comp_log)
            wandb_run.log(wlog)

        if sharded and (r + 1) % shard_every == 0:
            _flush_shard()

    env.close()

    # timing summary, plan = the CEM optimization, compare against the n_evals-batch benchmark,
    # setup/eval should be small after the dataset-decode fixes.
    t_total = time.perf_counter() - t_gen_start
    n_done = max(1, run_seen)
    _log(f"[timing] total={t_total / 60:.1f} min over {n_rounds} rounds, {run_seen} trajectories | "
         f"plan={t_plan_total / 60:.1f} min ({100 * t_plan_total / max(1e-9, t_total):.0f}%), "
         f"setup={t_setup_total / 60:.1f} min ({100 * t_setup_total / max(1e-9, t_total):.0f}%), "
         f"eval={t_eval_total / 60:.1f} min ({100 * t_eval_total / max(1e-9, t_total):.0f}%) | "
         f"{t_plan_total / n_done:.2f}s/traj planning, {t_total / n_done:.2f}s/traj total")
    if steps_run["n"]:
        _avg = steps_run["sum"] / steps_run["n"]
        _line = (f"[steps] average {_avg:.1f} env steps per trajectory over "
                 f"{steps_run['n']} trajectories ({_avg / max(1, frameskip):.1f} WM steps "
                 f"at frameskip={frameskip})")
        if steps_run["succ_n"]:
            _line += (f" | among successes: {steps_run['succ_sum'] / steps_run['succ_n']:.1f} "
                      f"steps to reach the goal (n={steps_run['succ_n']})")
        _log(_line)

    if wandb_run is not None:
        wandb_run.log({
            "final/reached_rate": run_reached / max(1, run_seen),
            "final/trajectories": run_seen,
            "final/total_min": t_total / 60,
        })
        wandb_run.finish()

    if sharded:
        _flush_shard()   # remaining trajectories since the last flush
        _log(f"Saved {counters['total']} planned trajectories "
             f"({counters['total_success']} reached goal) across {counters['shard_idx']} "
             f"shards in {shard_dir}")
        # counts the caller can read without loading the shards
        with open(os.path.join(shard_dir, "manifest.json"), "w") as _mf:
            json.dump({"trajectories": int(counters["total"]),
                       "successes": int(counters["total_success"]),
                       "shards": int(counters["shard_idx"]),
                       "rounds": int(n_rounds)}, _mf, indent=2)
        return shard_dir

    torch.save(_payload(), out_path)
    n_succ = int(sum(1 for s in successes_out if s))   # None (dream) -> not a success
    _log(f"Saved {len(seq_lengths)} planned trajectories ({n_succ} reached goal) to {out_path}")
    return out_path


@hydra.main(config_path="conf", config_name="plan")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
    cfg_dict = cfg_to_dict(cfg)
    # wandb is on by default, set wandb_logging, false in the config (or CLI) to disable.
    cfg_dict["wandb_logging"] = bool(cfg_dict.get("wandb_logging", True))
    # a relative planned_out_path lands inside the unique, timestamped Hydra run dir
    # (plan_outputs/<ts>_<model>_gH<H>/), so re-runs never mix shards and everything for a run
    # (shards, generation.log, figures) sits together.
    out = cfg_dict.get("planned_out_path") or "./planned_trajectories.pth"
    if not os.path.isabs(out):
        out = os.path.join(os.getcwd(), out)
    cfg_dict["planned_out_path"] = out
    generate(cfg_dict)


if __name__ == "__main__":
    main()
