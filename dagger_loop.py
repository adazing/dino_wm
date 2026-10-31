"""DAgger for BC policies, using the world model + planner as the expert.

collect states by driving  policy (beta<1) or planner (beta=1)
relabel those states  CEM / MPC plans from them
constant_1  planner always drives (phase 1 generation)
first_only  planner on round 0, policy after (standard DAgger)
exponential  beta = beta_p ** round
constant_x  fixed mix
"""
import os
# MuJoCo defaults to the GLFW backend, which needs an X display and fails on a headless box.
os.environ.setdefault("MUJOCO_GL", "egl")
import sys
import json
import time
import random
import re
import shutil
import signal
import subprocess
from pathlib import Path

import hydra
import torch
import numpy as np
from omegaconf import OmegaConf, open_dict

_STOP = False
_LOOP_RUN = None   # the loop's wandb run, closed by _main on every exit path


def _handle_stop(signum, frame):
    global _STOP
    _STOP = True
    print("\n[dagger] stop requested, finishing the current step then exiting.", flush=True)


def _hy(v):
    """Render a Python value for a Hydra "++key=value" override.

    Hydra parses "++key=None" as the STRING 'None', not as null. Only "++key=null" gives a real
    None. So every nullable override has to emit the literal word null, or a cap of None arrives
    downstream as the string 'None', which then fails int() or trips the "this cap belongs to the
    other mode" check. Verified: compose(overrides=["++k=None"]) -> 'None' (str);
    ["++k=null"] -> None.
    """
    return "null" if v is None else v


def _beta(schedule, r, p):
    """Fraction of this round's episodes driven by the planner rather than the policy."""
    s = str(schedule)
    if s == "constant_1":
        return 1.0
    if s == "first_only":
        return 1.0 if r == 0 else 0.0
    if s == "exponential":
        return float(p) ** r
    if s.startswith("constant_"):
        return float(s.split("_", 1)[1])
    raise ValueError(f"beta_schedule must be constant_1 | first_only | exponential | "
                     f"constant_<x>, got {schedule!r}")


def _run(cmd, log_path, desc, summary_log=None):
    """Run a subprocess, streaming its output to the terminal and to log_path."""
    header = f"[dagger] {desc}"
    print(f"{chr(10)}{'=' * 78}{chr(10)}{header}{chr(10)}[dagger] $ {' '.join(cmd)}{chr(10)}{'=' * 78}",
          flush=True)
    t0 = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:   # "w", one file per step, so re-running overwrites
        log.write(f"$ {' '.join(cmd)}{chr(10)}")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()
    dt = time.perf_counter() - t0
    if summary_log is not None:
        with open(summary_log, "a") as sl:
            sl.write(f"{header}  ->  exit {proc.returncode}  in {dt / 60:.1f} min"
                     f"  (full output: {log_path}){chr(10)}")
    return proc.returncode, dt


def _absorb_shards(gen_dir, pool_dir):
    """Move a round's shards into the pool, renumbered to continue its sequence."""
    pool_dir.mkdir(parents=True, exist_ok=True)
    n_existing = len(list(pool_dir.glob("shard_*.pth")))
    new = sorted(Path(gen_dir).glob("shard_*.pth"))
    if not new:
        raise FileNotFoundError(
            f"no shard_*.pth under {gen_dir}: generation produced nothing (shard_every must be > 0).")
    for k, src in enumerate(new):
        dst = pool_dir / f"shard_{n_existing + k:04d}.pth"
        tmp = pool_dir / f".partial_{dst.name}"
        shutil.move(str(src), str(tmp))
        os.replace(tmp, dst)
    return len(new), n_existing + len(new)


def _collect_states(cfg, rdir, prev_ckpt, base_seed, max_states):
    """Roll the current policy out and choose which visited states to relabel."""
    if prev_ckpt is None:
        print("[dagger] no policy checkpoint yet -> this round falls back to random init/goal")
        return None, 0

    import policy_rollout as pr
    import train_policy as tp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.env.goal_dim = int(cfg.encoder.output_dim) if bool(cfg.get("goal_conditional", False)) else 0
    encoder = hydra.utils.instantiate(cfg.encoder).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    policy = hydra.utils.instantiate(cfg.policy).to(device)
    payload = torch.load(prev_ckpt, map_location="cpu", weights_only=False)
    state = (payload["model"] if isinstance(payload, dict) and "model" in payload
             else payload.state_dict() if hasattr(payload, "state_dict") else payload)
    policy.load_state_dict(tp._fix_vqvae_keys(state))
    policy.eval()

    env = tp.make_eval_env(cfg, cfg.n_envs)
    try:
        _rollout_rec = []
        records = pr.rollout_policy(
            cfg, encoder, policy, device, env,
            n_episodes=int(cfg.rollout_episodes),
            # `or` rather than get(key, default), because rollout_horizon exists in the config
            # with value null and get() would return None.
            horizon=int(cfg.get("rollout_horizon") or cfg.eval_horizon),
            base_seed=base_seed,
            use_diffusion="diffusion" in cfg.policy["_target_"],
            # store this many scouting episodes in full, as world-model training data
            record_episodes=int(cfg.get("rollout_record_episodes", 0) or 0),
            record_into=_rollout_rec)
    finally:
        env.close()
        # free the encoder/policy explicitly.
        del policy, encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n_succ = sum(1 for r in records if r["success"])
    rng = random.Random(base_seed)
    inits, goals, infos, prov = pr.select_relabel_states(records, cfg, rng, max_states=max_states)
    if not inits:
        print("[dagger] no states selected for relabelling this round")
        return None, n_succ

    init_arr, goal_arr = np.stack(inits), np.stack(goals)
    if init_arr.shape[1:] != goal_arr.shape[1:]:
        # PlanWorkspace treats these as one state space.
        raise ValueError(
            f"init_states {init_arr.shape} and goal_states {goal_arr.shape} disagree past the "
            f"batch dim. The env's info['state'] and sample_random_init_goal_states() must use "
            f"the same representation for relabelling to be meaningful.")
    path = rdir / "relabel_states.pth"
    torch.save({"init_states": init_arr, "goal_states": goal_arr,
                "env_infos": infos, "provenance": prov}, path)
    torch.save(records, rdir / "policy_rollout.pth")   # scouting record, for diagnosis
    if _rollout_rec:
        # Same shard layout as the planner pool, so train.py reads it with the same dataset class.
        # Cumulative across rounds, like the planner pool, so the world model's training set grows
        # instead of being a fresh 154 windows each round.
        _rp = rdir.parent.parent / "rollout_pool"
        _rp.mkdir(parents=True, exist_ok=True)
        _shard = {k: [r[k] for r in _rollout_rec]
                  for k in ("visual", "actions", "states", "proprios", "seq_lengths",
                            "init_states", "goal_states", "successes")}
        _shard["env_name"] = str(cfg.env.name)
        _shard["frameskip"] = int(cfg.get("frameskip", 1))
        _sp = _rp / f"shard_rollout_{rdir.name}.pth"
        torch.save(_shard, _sp)
        _mb = sum(int(np.asarray(v).nbytes) for v in _shard["visual"]) / 1e6
        print(f"[dagger] recorded {len(_rollout_rec)} scouting episodes "
              f"({_mb:.0f} MB) -> {_sp}")
    print(f"[dagger] policy rollout: {n_succ}/{len(records)} succeeded; "
          f"selected {len(inits)} states to relabel -> {path}")
    return path, n_succ


def _pick_ckpt(cfg, ckpt_dir):
    """The checkpoint this round hands to the next round.

    model_best.pt is the weights that scored best on the policy's select_metric during this round's
    training; model_final.pt is merely the last epoch. Wall round 0 of 20261008010658 peaked at
    epoch ~150 and finished 10 points of goal25_valid/max_success_rate worse, so carrying the last
    epoch forward started every later round from a degraded policy. Falls back to model_final.pt
    when use_best_checkpoint is false, when select_metric is unset, and for rounds that were trained
    before best-tracking existed.
    """
    if bool(cfg.get("use_best_checkpoint", True)):
        best = ckpt_dir / "model_best.pt"
        if best.is_file():
            return best
    return ckpt_dir / "model_final.pt"


def main(cfg):
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    out_dir = Path(os.path.abspath(str(cfg.out_dir)))
    pool_dir = out_dir / "pool"
    rounds_dir = out_dir / "rounds"
    log_path = out_dir / "dagger.log"
    summary_path = out_dir / "dagger_summary.json"
    for d in (out_dir, pool_dir, rounds_dir):
        d.mkdir(parents=True, exist_ok=True)
    if not cfg.get("model_name"):
        raise ValueError("dagger_loop needs model_name=<the frozen WM run under "
                         "ckpt_base_path/outputs/>. The planner plans with that world model.")

    # validate the world model before anything expensive runs.
    _wm_dir = Path(str(cfg.ckpt_base_path)) / "outputs" / str(cfg.model_name)
    _wm_cfg = _wm_dir / "hydra.yaml"
    if not _wm_cfg.is_file():
        _root = Path(str(cfg.ckpt_base_path)) / "outputs"
        _found = sorted(q.parent.relative_to(_root).as_posix()
                        for q in _root.glob("*/hydra.yaml"))
        _found += sorted(q.parent.relative_to(_root).as_posix()
                         for q in _root.glob("*/*/hydra.yaml"))
        print(f"[dagger] model_name={cfg.model_name!r} -> {_wm_cfg} does not exist.")
        if _found:
            print(f"[dagger] world-model runs available under {_root} (last 25):")
            for _f in _found[-25:]:
                print(f"[dagger]     {_f}")
        else:
            print(f"[dagger] no */hydra.yaml anywhere under {_root}"
                  f". Is ckpt_base_path right?")
        raise FileNotFoundError(f"world model not found: {_wm_cfg}")
    _ep = str(cfg.model_epoch)
    _wm_ckpt = _wm_dir / "checkpoints" / f"model_{_ep}.pth"
    if not _wm_ckpt.is_file():
        _eps = sorted(q.stem.replace("model_", "")
                      for q in (_wm_dir / "checkpoints").glob("model_*.pth"))
        print(f"[dagger] model_epoch={_ep!r} -> {_wm_ckpt} does not exist.")
        print(f"[dagger] epochs available: {', '.join(_eps) or '(none)'}")
        raise FileNotFoundError(f"world model checkpoint not found: {_wm_ckpt}")
    print(f"[dagger] world model OK: {_wm_dir} (epoch {_ep})")

    # Hydra resolves --config-name only when the child starts, so a missing conf/<name>.yaml would
    # not surface until a full round of generation has already run.
    _conf_dir = Path(__file__).resolve().parent / "conf"
    for _key, _default in (("gen_config_name", None), ("train_config_name", "train_policy")):
        _name = cfg.get(_key, _default)
        if not _name:
            continue
        if not (_conf_dir / f"{_name}.yaml").is_file():
            _have = sorted(q.stem for q in _conf_dir.glob(f"{str(_key).split('_')[0]}*.yaml"))
            print(f"[dagger] {_key}={_name!r} -> {_conf_dir / (str(_name) + '.yaml')} does not exist.")
            print(f"[dagger] similar configs present: {', '.join(_have) or '(none)'}")
            raise FileNotFoundError(
                f"{_key}={_name!r}: conf/{_name}.yaml is missing. Hydra would only fail after "
                f"a full round of generation, so this is checked up front.")
    print(f"[dagger] configs OK: gen={cfg.get('gen_config_name')} "
          f"train={cfg.get('train_config_name', 'train_policy')}")

    # Demo folders. The world model and the policy can want different ones, so they are
    # resolved separately.
    #   override_dataset_path  generation and the world-model fine-tune, so it must be the
    #                          folder the world model was trained on
    #   policy_dataset_path    policy training, its demo-goal evals and the scouting rollout.
    #                          null falls back to override_dataset_path
    override_dataset_path = cfg.get("override_dataset_path", None)
    if override_dataset_path:
        override_dataset_path = str(override_dataset_path)
        if not Path(override_dataset_path).is_dir():
            raise FileNotFoundError(
                f"override_dataset_path={override_dataset_path!r} is not a directory. It must "
                f"point at the demo dataset folder the WM was trained on.")
    policy_dataset_path = cfg.get("policy_dataset_path", None) or override_dataset_path
    if policy_dataset_path:
        policy_dataset_path = str(policy_dataset_path)
        if not Path(policy_dataset_path).is_dir():
            raise FileNotFoundError(
                f"policy_dataset_path={policy_dataset_path!r} is not a directory.")
        with open_dict(cfg):
            cfg.dataset.data_path = policy_dataset_path
    print(f"[dagger] demo data: world model <- {override_dataset_path}, "
          f"policy <- {policy_dataset_path}")

    # A timestamp created once per out_dir and then remembered, so a rerun with different settings
    # gets a distinct wandb identity while resuming the same out_dir rejoins its existing group.
    run_id_path = out_dir / "run_id.txt"
    if run_id_path.is_file():
        run_id = run_id_path.read_text().strip()
        print(f"[dagger] resuming run id {run_id} (from {run_id_path})")
    else:
        run_id = time.strftime("%Y%m%d-%H%M%S")
        run_id_path.write_text(run_id + chr(10))
        print(f"[dagger] new run id {run_id}")
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    summary = (json.loads(summary_path.read_text()) if summary_path.is_file()
               else {"rounds": []})
    prev_ckpt = cfg.get("init_checkpoint", None)   # optional warm start
    # running pool totals, rebuilt from the summary so a resume continues the counts
    pool_traj = sum(int(x.get("trajectories_added") or 0) for x in summary["rounds"])
    pool_succ = sum(int(x.get("successes_added") or 0) for x in summary["rounds"])
    base_seed = int(cfg.seed)
    # the world model generation plans with. wm_finetune replaces it after every round.
    wm_base, wm_name, wm_epoch = cfg.ckpt_base_path, cfg.model_name, cfg.model_epoch
    # Immutable copy. wm_base/wm_name/wm_epoch get rebound to each round's fine-tuned output
    # so the PLANNER plans with the newest world model; wm_finetune_from=base needs the
    # original pretrained checkpoint to still be reachable after that rebinding.
    wm_base0, wm_name0, wm_epoch0 = wm_base, wm_name, wm_epoch

    print(f"[dagger] out_dir={out_dir}\n"
          f"[dagger]   resume this run with: out_dir={out_dir}\n"
          f"[dagger] pool={pool_dir}\n"
          f"[dagger] rounds={cfg.rounds}  beta_schedule={cfg.beta_schedule}\n"
          f"[dagger] planner-driven episodes/round={cfg.trajectories_per_round}, "
          f"policy rollouts/round={cfg.rollout_episodes}")

    # The policy's data_source is derived from its two caps rather than configured, so there is
    # one place that decides what the policy trains on. A cap of 0 excludes that source.
    _co = cfg.get("policy_max_original_windows", None)
    _cp = cfg.get("policy_max_planned_windows", None)
    if _co == 0 and _cp == 0:
        raise ValueError("policy_max_original_windows and policy_max_planned_windows are both 0, "
                         "so the policy would have nothing to train on.")
    _policy_data_source = ("planned" if _co == 0 else
                           "original" if _cp == 0 else "both")
    print(f"[dagger] policy data: max_original_windows={_co} max_planned_windows={_cp} "
          f"-> data_source={_policy_data_source}")

    # Each round launches two subprocesses. Name them explicitly and put them in one wandb group
    # so the UI collapses a whole dagger run into a single row.
    _env_name = str(OmegaConf.select(cfg, "env.name") or "env")
    # out_dir is timestamped per launch and already names the env and beta schedule, so its leaf
    # is the group.
    wandb_group = str(cfg.get("wandb_group") or out_dir.name)
    # Every run name carries the launch timestamp, so a run is identifiable on its own rather than
    # only inside its group.
    def _run_name(suffix):
        return f"{_env_name}-{run_id}-{suffix}"
    wandb_project = cfg.get("wandb_project", None)
    wandb_tags = [_env_name, str(cfg.beta_schedule), f"data={_policy_data_source}"]
    print(f"[dagger] wandb group={wandb_group}"
          + (f"  project={wandb_project} (generation AND training together)"
             if wandb_project else
             "  (generation -> dino_wm_planning, training -> dino_wm_bc; set wandb_project "
             "to put them in one project so the group is visible across both)"))

    def _san(v):
        """Make a value safe to pass through a Hydra CLI override."""
        return re.sub(r"[^A-Za-z0-9._-]", "_", str(v))

    def _wandb_args(name, job_type):
        """Hydra overrides that name/group this subprocess's wandb run."""
        a = [f"++wandb.name={_san(name)}", f"++wandb.group={_san(wandb_group)}",
             f"++wandb.job_type={_san(job_type)}",
             "++wandb.tags=[" + ",".join(_san(t) for t in wandb_tags) + "]"]
        if wandb_project:
            a.append(f"++wandb.project={_san(wandb_project)}")
        return a


    # One run for the loop itself. The per-stage subprocess runs keep the detail, this collects
    # each round's headline numbers so both generation arms, the world model and the policy can
    # be read off a single round axis.
    loop_run = None
    if wandb_project and bool(cfg.get("wandb_logging", True)):
        try:
            import wandb
            loop_run = wandb.init(project=str(wandb_project), group=_san(wandb_group),
                                  name=_san(_run_name("loop")), job_type="dagger_loop",
                                  tags=wandb_tags, dir=str(out_dir),
                                  config=OmegaConf.to_container(cfg, resolve=True))
            # 'round' is the x axis instead of wandb's implicit step counter. Logging against an
            # explicit metric commits each call straight away, so a chart fills in as every stage
            # of a round finishes rather than only once the round is over.
            loop_run.define_metric("round")
            loop_run.define_metric("*", step_metric="round")
            globals()["_LOOP_RUN"] = loop_run
            print(f"[dagger] loop summary run: {loop_run.name}, files under {out_dir}/wandb")
        except Exception as _e:
            print(f"[dagger] no loop summary run ({_e}); per-stage runs are unaffected")

    def _gen_metrics(counts):
        """Flatten the per-arm manifest numbers into chartable keys."""
        out = {}
        for _tag, _v in (counts or {}).items():
            for _k in ("trajectories", "successes", "success_rate", "coverage",
                       "wm_latent_err", "wm_latent_err_median", "wm_latent_err_p90"):
                if _v.get(_k) is not None:
                    out[f"gen_{_tag}/{_k}"] = _v[_k]
        return out

    def _loop_log(payload, rnd):
        if loop_run is None:
            return
        loop_run.log({"round": rnd, **{k: v for k, v in payload.items() if v is not None}})

    # A resume skips the rounds that already finished, so replay their summary entries. Without
    # this the charts would start at whatever round the resume happens to begin from.
    if loop_run is not None and summary["rounds"]:
        for _past in summary["rounds"]:
            _loop_log({"beta": _past.get("beta"),
                       "pool/trajectories": _past.get("pool_trajectories"),
                       "pool/successes": _past.get("pool_successes"),
                       "pool/shards": _past.get("pool_shards"),
                       "policy_rollout/successes": _past.get("policy_rollout_successes"),
                       **_gen_metrics(_past.get("generated"))},
                      int(_past.get("round", 0)))
        print(f"[dagger] backfilled {len(summary['rounds'])} finished rounds into the loop run")

    for r in range(int(cfg.rounds)):
        if _STOP:
            break
        rdir = rounds_dir / f"round_{r:02d}"
        gen_run, train_run = rdir / "gen", rdir / "train"
        done_marker, ckpt_dir = rdir / "DONE", train_run / "checkpoints"
        train_ckpt = _pick_ckpt(cfg, ckpt_dir)
        if done_marker.is_file():
            print(f"[dagger] round {r}: already done, skipping")
            prev_ckpt = train_ckpt if train_ckpt.is_file() else prev_ckpt
            continue
        rdir.mkdir(parents=True, exist_ok=True)

        beta = _beta(cfg.beta_schedule, r, cfg.get("beta_p", 0.5))
        # per-episode mixing.
        n_total = int(cfg.trajectories_per_round)
        n_planner = int(round(beta * n_total))
        n_policy = n_total - n_planner
        # The control arm is policy_max_planned_windows=0: the policy trains on demonstrations
        # only, so there is nothing for the planner to generate. data_source is DERIVED from the
        # caps now, so this reads the derived value rather than a config key that no longer exists.
        generate = _policy_data_source != "original"
        if not generate:
            n_planner = n_policy = 0
        print(f"\n[dagger] === round {r} | beta={beta:.2f} -> "
              + (f"{n_planner} planner-driven, {n_policy} policy-scouted ===" if generate
                 else "data_source=original: no generation, training on demos only ==="))

        # 1.
        states_path, n_succ = (None, None)
        if n_policy > 0:
            states_path, n_succ = _collect_states(
                cfg, rdir, prev_ckpt, base_seed + r * 100_000, max_states=n_policy)
            if states_path is None:
                n_planner, n_policy = n_total, 0   # no policy yet, so all random

        # 2.
        n_new_shards = 0
        gen_counts = {}
        tag_ran_relabel = False
        for tag, count, extra in (
                ("random", n_planner, []),
                ("relabel", n_policy, [f"++provided_states_path={states_path}"] if states_path else [])):
            if count <= 0 or (tag == "relabel" and not states_path):
                continue
            sub = gen_run / tag
            cmd = [
                sys.executable, "generate_planned_trajectories.py",
                f"--config-name={cfg.gen_config_name}",
                f"hydra.run.dir={sub}",
                f"++ckpt_base_path={wm_base}",
                f"++model_name={wm_name}",
                f"++model_epoch={wm_epoch}",
                "++planned_out_path=./planned",
                f"++shard_every={int(cfg.shard_every)}",
                f"++seed={base_seed + r * 10_000}",
                f"++eval_seed_base={base_seed + r * 10_000}",
            ] + ([f"++num_trajectories={count}"] if tag == "random" else [])
            # generation reads the demo folder from the world model's saved config, which may
            # point at a path that does not exist here.
            if override_dataset_path:
                cmd += [f"++override_dataset_path={override_dataset_path}"]
            # in-process envs, for cases where MuJoCo cannot make a GL context in a forked child.
            if bool(cfg.get("serial_env", False)):
                cmd += ["++serial_env=true"]
            if cfg.get("env_start_method", None):
                cmd += [f"++env_start_method={cfg.env_start_method}"]
            cmd += _wandb_args(_run_name(f"r{r:02d}-gen-{tag}"),
                               "generate" if tag == "random" else "relabel")
            cmd += extra + [str(o) for o in (cfg.get("gen_overrides") or [])]
            rc, dt = _run(cmd, log_dir / f"round{r:02d}_gen_{tag}.log",
                          f"round {r}: generate ({tag}, {count} trajectories)",
                          summary_log=log_path)
            if rc != 0:
                raise RuntimeError(f"round {r} generation ({tag}) failed with exit {rc}; see "
                                   f"{log_dir / f'round{r:02d}_gen_{tag}.log'}")
            tag_ran_relabel = tag_ran_relabel or (tag == "relabel")
            n_new, n_pool = _absorb_shards(sub / "planned", pool_dir)
            n_new_shards += n_new
            _mf = sub / "planned" / "manifest.json"
            if _mf.is_file():
                _m = json.loads(_mf.read_text())
                gen_counts[tag] = {"trajectories": _m.get("trajectories"),
                                   "successes": _m.get("successes"),
                                   "success_rate": _m.get("success_rate"),
                                   "wm_latent_err": _m.get("wm_latent_err"),
                                   "wm_latent_err_median": _m.get("wm_latent_err_median"),
                                   "wm_latent_err_p90": _m.get("wm_latent_err_p90")}
                print(f"[dagger] round {r} ({tag}): +{n_new} shards, "
                      f"{_m.get('trajectories')} trajectories "
                      f"({_m.get('successes')} reached goal) -> pool has {n_pool} shards")
            else:
                print(f"[dagger] round {r} ({tag}): +{n_new} shards -> pool has {n_pool}")

        # 2a. A measurement-only arm, off by default. The relabel arm's difficulty rises every
        # round as the policy improves, so its success rate and wm error cannot be compared
        # across rounds. This plans a small batch from env-sampled init/goal pairs, whose
        # difficulty is FIXED, giving a yardstick that only moves when the world model moves.
        # Its trajectories are deliberately NOT absorbed into the pool, so training is unchanged.
        _n_probe = int(cfg.get("measure_random_trajectories", 0) or 0)
        if _n_probe > 0 and generate and tag_ran_relabel:
            sub_m = gen_run / "measure"
            cmd = [
                sys.executable, "generate_planned_trajectories.py",
                f"--config-name={cfg.gen_config_name}",
                f"hydra.run.dir={sub_m}",
                f"++ckpt_base_path={wm_base}",
                f"++model_name={wm_name}",
                f"++model_epoch={wm_epoch}",
                "++planned_out_path=./planned",
                f"++shard_every={int(cfg.shard_every)}",
                f"++num_trajectories={_n_probe}",
                "++goal_source=random_state",
                f"++seed={base_seed + 777 + r * 10_000}",
                f"++eval_seed_base={base_seed + 777 + r * 10_000}",
            ]
            if override_dataset_path:
                cmd += [f"++override_dataset_path={override_dataset_path}"]
            if bool(cfg.get("serial_env", False)):
                cmd += ["++serial_env=true"]
            if cfg.get("env_start_method", None):
                cmd += [f"++env_start_method={cfg.env_start_method}"]
            cmd += _wandb_args(_run_name(f"r{r:02d}-gen-measure"), "measure")
            cmd += [str(o) for o in (cfg.get("gen_overrides") or [])]
            rc, _dt = _run(cmd, log_dir / f"round{r:02d}_gen_measure.log",
                           f"round {r}: measurement arm ({_n_probe} trajectories, not trained on)",
                           summary_log=log_path)
            _mf = sub_m / "planned" / "manifest.json"
            if rc == 0 and _mf.is_file():
                _m = json.loads(_mf.read_text())
                _loop_log({f"gen_measure/{k}": _m.get(k) for k in
                           ("trajectories", "successes", "success_rate", "coverage",
                            "wm_latent_err", "wm_latent_err_median", "wm_latent_err_p90")}, r)
                print(f"[dagger] round {r} (measure): {_m.get('trajectories')} trajectories, "
                      f"{_m.get('successes')} reached goal, wm_latent_err "
                      f"{_m.get('wm_latent_err')} -- NOT added to the pool")
            else:
                print(f"[dagger] round {r}: measurement arm failed (exit {rc}), continuing")

        # generation is done, so publish it now. Training the policy takes hours and holding
        # these back until then is what left the loop run's charts empty mid-round.
        _new_traj = sum(v["trajectories"] or 0 for v in gen_counts.values())
        _new_succ = sum(v["successes"] or 0 for v in gen_counts.values())
        _loop_log({"beta": beta, "policy_rollout/successes": n_succ,
                   "pool/trajectories": pool_traj + _new_traj,
                   "pool/successes": pool_succ + _new_succ,
                   **_gen_metrics(gen_counts)}, r)

        # 2b. optionally fine-tune the world model on the trajectories it got wrong, so the
        # next round plans with a model corrected where planning actually breaks. Off by
        # default, which leaves the frozen-world-model behaviour untouched.
        if bool(cfg.get("wm_finetune", False)) and n_new_shards > 0:
            wm_run = rdir / "outputs" / "wm"
            # Which checkpoint this round's fine-tune starts from.
            #   chain  the previous round's fine-tuned model, i.e. the one generation just planned
            #          with. Drift COMPOUNDS: each round starts from a worse model than the last.
            #   base   the original pretrained model, every round. The planner still plans with the
            #          newest fine-tune; only the fine-tune's starting point is pinned, so a round's
            #          drift cannot stack on the previous round's.
            _ft_from = str(cfg.get("wm_finetune_from", "chain")).lower()
            if _ft_from not in ("chain", "base"):
                raise ValueError(f"wm_finetune_from must be 'chain' or 'base', got {_ft_from!r}")
            if _ft_from == "base":
                _ft_base, _ft_name, _ft_epoch = wm_base0, wm_name0, wm_epoch0
            else:
                _ft_base, _ft_name, _ft_epoch = wm_base, wm_name, wm_epoch
            print(f"[dagger] round {r}: fine-tuning from "
                  f"{Path(_ft_base) / 'outputs' / str(_ft_name)} @ {_ft_epoch} "
                  f"(wm_finetune_from={_ft_from})")
            cmd = [
                sys.executable, "train.py",
                f"--config-name={cfg.wm_train_config_name}",
                f"hydra.run.dir={wm_run}",
                f"++init_from={Path(_ft_base) / 'outputs' / str(_ft_name)}",
                f"++init_epoch={_ft_epoch}",
                "++reset_epoch=true",
                f"++training.epochs={int(cfg.get('wm_finetune_epochs', 5))}",
                f"++planned_data_path={pool_dir}",
                # three independent window caps: null = all, 0 = exclude the group, N = N windows
                # planner  -> success / failure groups (did the PLANNER reach the goal)
                # wm_error -> wm_ok / wm_fail groups (did the WORLD MODEL mispredict), threshold
                #             recomputed from this round's pool at wm_finetune_err_pct
                f"++planned_split={cfg.get('wm_planned_split', 'planner')}",
                f"++max_demo_windows={_hy(cfg.get('wm_max_demo_windows', None))}",
                f"++max_rollout_windows={_hy(cfg.get('wm_max_rollout_windows', None))}",
                f"++max_success_windows={_hy(cfg.get('wm_max_success_windows', None))}",
                f"++max_failure_windows={_hy(cfg.get('wm_max_failure_windows', None))}",
                f"++max_wm_ok_windows={_hy(cfg.get('wm_max_wm_ok_windows', None))}",
                f"++max_wm_fail_windows={_hy(cfg.get('wm_max_wm_fail_windows', None))}",
                # the policy's recorded scouting episodes, written by _collect_states when
                # rollout_record_episodes > 0. Cumulative across rounds.
                f"++rollout_data_path={out_dir / 'rollout_pool'}",
                # Proportional mode: wm_total_windows fixes the SIZE of the fine-tune's set
                # and wm_frac_<group> fixes the MIX. Leave wm_total_windows null to use the caps.
                f"++total_windows={_hy(cfg.get('wm_total_windows', None))}",
                f"++frac_demo={_hy(cfg.get('wm_frac_demo', None))}",
                f"++frac_rollout={_hy(cfg.get('wm_frac_rollout', None))}",
                f"++frac_success={_hy(cfg.get('wm_frac_success', None))}",
                f"++frac_failure={_hy(cfg.get('wm_frac_failure', None))}",
                f"++frac_wm_ok={_hy(cfg.get('wm_frac_wm_ok', None))}",
                f"++frac_wm_fail={_hy(cfg.get('wm_frac_wm_fail', None))}",
                # orthogonal filter on the two planned groups
                f"++planned_wm_err_pct={float(cfg.get('wm_finetune_err_pct', 50.0))}",
                # which windows a cap keeps; varied per round so successive fine-tunes anchor
                # against different demo data instead of one fixed subset
                f"++window_subset_seed={base_seed + r}",
            ]
            # adajepa restricts adaptation to the predictor's last transformer block, which is why
            # it survives lr 5e-4 where a full-predictor fine-tune at that rate collapsed.
            if bool(cfg.get("wm_finetune_last_layer_only", False)):
                cmd += ["++finetune_last_layer_only=true"]
            # Keep the action/proprio encoders fixed across the fine-tune. Default true: the
            # predictor was trained against those embeddings and a 5-epoch fine-tune on ~150
            # windows is not the place to move them.
            if bool(cfg.get("wm_finetune_freeze_encoders", True)):
                cmd += ["++freeze_action_proprio_encoders=true"]
            if cfg.get("wm_finetune_lr", None) is not None:
                cmd += [f"++training.predictor_lr={cfg.wm_finetune_lr}",
                        f"++training.decoder_lr={cfg.wm_finetune_lr}",
                        f"++training.action_encoder_lr={cfg.wm_finetune_lr}"]
            # the same demo folder generation and policy training use. Without it the fine-tune
            # falls back to the dataset group's own path, which for pusht is a different folder.
            if override_dataset_path:
                cmd += [f"++dataset.data_path={override_dataset_path}"]
            cmd += _wandb_args(_run_name(f"r{r:02d}-wm"), "wm_finetune")
            cmd += [str(o) for o in (cfg.get("wm_overrides") or [])]
            rc, dt = _run(cmd, log_dir / f"round{r:02d}_wm.log",
                          f"round {r}: fine-tune the world model", summary_log=log_path)
            if rc != 0:
                raise RuntimeError(f"round {r} world-model fine-tune failed with exit {rc}, see "
                                   f"{log_dir / f'round{r:02d}_wm.log'}")
            # Carry the fine-tune's BEST epoch forward, not its last. On wall every round of
            # every fine-tuning arm reported val_loss_best_epoch == 1, so handing the last epoch
            # forward meant adopting the worst of the five just produced.
            _wm_best = wm_run / "checkpoints" / "model_best.pth"
            _use_best = bool(cfg.get("wm_use_best_checkpoint", True)) and _wm_best.is_file()
            wm_base, wm_name, wm_epoch = str(rdir), "wm", ("best" if _use_best else "latest")
            print(f"[dagger] round {r}: carrying forward model_{wm_epoch}.pth"
                  + ("" if _use_best else "  (no model_best.pth; using the last epoch)"))
            _wm_log = {"wm/finetuned": 1, "wm/finetune_seconds": dt}
            # how much data the fine-tune actually saw, written by train.py
            _dc = wm_run / "data_counts.json"
            if _dc.is_file():
                _wm_log.update({f"wm_{k}": v for k, v in json.loads(_dc.read_text()).items()})
            # the fine-tune's loss curve, so "is the world model degrading?" is answerable from
            # wandb instead of from this round's log file
            _wmm = wm_run / "wm_metrics.json"
            if _wmm.is_file():
                for _k, _v in json.loads(_wmm.read_text()).items():
                    if not isinstance(_v, list):          # the per-epoch list is not chartable here
                        _wm_log[f"wm/{_k}"] = _v
            _loop_log(_wm_log, r)
            print(f"[dagger] round {r}: next round plans with {wm_run}")

        # 3. retrain on the whole pool. DAgger aggregates rather than replaces.
        _cap = cfg.get("train_max_trajectories", None)
        _cap = "null" if _cap is None else int(_cap)
        cmd = [
            sys.executable, "train_policy.py",
            # which train config to compose, as conf/<name>.yaml.
            f"--config-name={cfg.get('train_config_name', 'train_policy')}",
            f"hydra.run.dir={train_run}",
            # data_source is DERIVED from the caps, not configured: a cap of 0 excludes a
            # source. train_policy still needs it to decide what to LOAD, and the train_policy
            # configs interpolate ${data_source} into the experiment name and output dir.
            f"++data_source={_policy_data_source}",
            f"++planned_data_path={pool_dir}",
            # the two independent WINDOW caps: null = all, 0 = exclude, N = seeded random N
            f"++max_original_windows={_hy(cfg.get('policy_max_original_windows', None))}",
            f"++max_planned_windows={_hy(cfg.get('policy_max_planned_windows', None))}",
            # which windows a cap keeps; varied per round so successive rounds see different
            # subsets rather than one fixed slice
            f"++window_subset_seed={base_seed + r}",
            # the pool grows every round and planned_max_trajectories raises when it exceeds what
            # exists, so a fixed number would kill early rounds.
            f"++planned_max_trajectories={_cap}",
            f"++planned_selection={cfg.get('train_selection', 'first')}",
            # reseed the subset each round so 'random' selection actually resamples
            f"++planned_selection_seed={base_seed + r}",
        ] + [str(o) for o in (cfg.get("train_overrides") or [])]
        # Per-arm policy-training scalars, appended AFTER train_overrides so they win over the
        # values baked into that list. `epochs` in particular is NOT comparable across
        # data_source settings: on planned-only data one epoch is a few hundred optimizer steps,
        # while data_source=both with uncapped demos makes one epoch over a hundred thousand.
        # Leaving these null keeps whatever train_overrides says.
        for _key, _cfg_key in (("epochs", "policy_epochs"),
                               ("eval_on_env_freq", "policy_eval_on_env_freq"),
                               ("windows_per_epoch", "policy_windows_per_epoch")):
            _val = cfg.get(_cfg_key, None)
            if _val is not None:
                cmd += [f"++{_key}={_val}"]
        # demo dataset for action normalisation stats and demo-goal eval passes.
        if policy_dataset_path:
            cmd += [f"++dataset.data_path={policy_dataset_path}"]
        if bool(cfg.get("serial_env", False)):
            cmd += ["++serial_env=true"]
        if cfg.get("env_start_method", None):
            cmd += [f"++env_start_method={cfg.env_start_method}"]
        cmd += _wandb_args(_run_name(f"r{r:02d}-train"), "train")
        if bool(cfg.get("warm_start", False)) and prev_ckpt is not None:
            # weights only, fresh optimizer.
            cmd += [f"++checkpoint={prev_ckpt}", "++resume=false"]
        rc, dt = _run(cmd, log_dir / f"round{r:02d}_train.log",
                      f"round {r}: train on the pool", summary_log=log_path)
        if rc != 0:
            raise RuntimeError(f"round {r} training failed with exit {rc}; see "
                               f"{log_dir / f'round{r:02d}_train.log'}")
        # re-resolve: model_best.pt exists only now that this round's training has run.
        train_ckpt = _pick_ckpt(cfg, ckpt_dir)
        _bj = ckpt_dir / "model_best.json"
        if train_ckpt.name == "model_best.pt" and _bj.is_file():
            try:
                _b = json.loads(_bj.read_text())
                print(f"[dagger] round {r}: carrying forward model_best.pt "
                      f"({_b['metric']}={_b['value']} at epoch {_b['epoch']})")
            except Exception:
                pass
        else:
            print(f"[dagger] round {r}: carrying forward {train_ckpt.name}")
        prev_ckpt = train_ckpt if train_ckpt.is_file() else prev_ckpt

        done_marker.write_text(f"round {r} complete\n")
        _gen_traj = sum(v["trajectories"] or 0 for v in gen_counts.values())
        _gen_succ = sum(v["successes"] or 0 for v in gen_counts.values())
        pool_traj += _gen_traj
        pool_succ += _gen_succ
        summary["rounds"].append({
            "round": r, "beta": beta,
            # requested by beta
            "planner_requested": n_planner, "policy_requested": n_policy,
            # actually produced, per arm and in total
            "generated": gen_counts,
            "trajectories_added": _gen_traj, "successes_added": _gen_succ,
            # running pool totals.
            "pool_trajectories": pool_traj, "pool_successes": pool_succ,
            "pool_shards": len(list(pool_dir.glob("shard_*.pth"))),
            "policy_rollout_successes": n_succ, "new_shards": n_new_shards,
            "checkpoint": str(train_ckpt) if train_ckpt.is_file() else None,
            "world_model": f"{wm_base}/outputs/{wm_name}",
        })
        print(f"[dagger] round {r}: +{_gen_traj} trajectories ({_gen_succ} usable) "
              f"-> pool {pool_traj} trajectories, {pool_succ} usable")
        _pl = {"pool/shards": summary["rounds"][-1]["pool_shards"], "train/seconds": dt}
        # The policy's own per-round scores, written by train_policy.py. goal25_valid/
        # max_success_rate is the reference metric: fixed difficulty every round, so it is the
        # only one comparable across rounds and runs.
        _mf = train_run / "metrics_final.json"
        if _mf.is_file():
            try:
                _m = json.loads(_mf.read_text())
                for _k, _v in (_m.get("best") or {}).items():
                    _pl[f"policy_best/{_k}"] = _v
                for _k, _v in (_m.get("final") or {}).items():
                    _pl[f"policy_final/{_k}"] = _v
                # which epoch best-checkpoint selection kept, and its score. Wall rounds 1-3
                # peaked at epochs 25, 0 and 0 of 150, so this says how much the selection saved.
                if _m.get("select_best_value") is not None:
                    _pl["policy/select_best_value"] = _m["select_best_value"]
                    _pl["policy/select_best_epoch"] = _m.get("select_best_epoch")
                    _pl["policy/select_metric_name"] = _m.get("select_metric")
            except Exception as _e:
                print(f"[dagger] round {r}: could not read {_mf}: {_e}")
        else:
            print(f"[dagger] round {r}: no metrics_final.json; policy scores will not be on the "
                  f"loop run's round axis for this round")
        _loop_log(_pl, r)

        # 4. optionally score the new policy on the SAME states the planner just solved. The
        # planner's success rate is measured on the policy's failures, whose difficulty rises every
        # round, so it cannot be read against the policy's own goal25 passes. This puts both on
        # identical problems. Horizon 25 matches the planner's budget (goal_H * frameskip), 300
        # matches the goal25 passes, so the pair separates "slower" from "cannot".
        if bool(cfg.get("eval_policy_on_relabel", False)) and (rdir / "relabel_states.pth").is_file():
            _ev = [sys.executable, "tools/eval_policy_on_relabel.py",
                   f"--run_dir={out_dir}", f"--rounds={r}",
                   f"--horizons={cfg.get('relabel_eval_horizons', '25,300')}",
                   f"--out={rdir / 'relabel_eval.json'}"]
            rc, _dt = _run(_ev, log_dir / f"round{r:02d}_relabel_eval.log",
                           f"round {r}: score the policy on the planner's own states",
                           summary_log=log_path)
            _rj = rdir / "relabel_eval.json"
            if rc == 0 and _rj.is_file():
                _rows = json.loads(_rj.read_text())
                if _rows:
                    _row = _rows[-1]
                    _pl = _row.get("planner_success_rate")
                    _lg = {"relabel/planner_success": _pl,
                           "relabel/n_states": _row.get("n_states")}
                    for _k, _v in _row.items():
                        if _k.startswith("policy_"):
                            _lg[f"relabel/{_k}"] = _v
                    _h = str(cfg.get("relabel_eval_horizons", "25,300")).split(",")[0]
                    _pm = _row.get(f"policy_max_success_h{_h}")
                    if _pl is not None and _pm is not None:
                        _lg["relabel/gap"] = _pl - _pm   # how much better the expert still is
                    _loop_log(_lg, r)
                    print(f"[dagger] round {r}: on the planner's {_row.get('n_states')} states, "
                          f"planner={_pl} policy@h{_h}={_pm}")
            else:
                print(f"[dagger] round {r}: relabel eval failed (exit {rc}), continuing")
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"[dagger] round {r} done -> {summary_path}")

    if loop_run is not None:
        loop_run.finish()
        globals()["_LOOP_RUN"] = None
    print(f"\n[dagger] finished. pool={pool_dir}  summary={summary_path}\n"
          f"[dagger] DAgger returns the best round on validation, not the last. Compare the "
          f"per-round eval metrics before picking a checkpoint.")


@hydra.main(config_path="conf", config_name="dagger_wall", version_base="1.2")
def _main(cfg: OmegaConf):
    try:
        main(cfg)
    finally:
        # A failed round raises straight past the finish() at the end of main, which leaves the
        # loop run marked 'running' in wandb for good.
        if _LOOP_RUN is not None:
            try:
                _LOOP_RUN.finish(exit_code=1)
            except Exception as _e:
                print(f"[dagger] wandb.finish failed: {_e}")


if __name__ == "__main__":
    _main()