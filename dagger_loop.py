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


def _handle_stop(signum, frame):
    global _STOP
    _STOP = True
    print("\n[dagger] stop requested, finishing the current step then exiting.", flush=True)


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
        records = pr.rollout_policy(
            cfg, encoder, policy, device, env,
            n_episodes=int(cfg.rollout_episodes),
            # `or` rather than get(key, default), because rollout_horizon exists in the config
            # with value null and get() would return None.
            horizon=int(cfg.get("rollout_horizon") or cfg.eval_horizon),
            base_seed=base_seed,
            use_diffusion="diffusion" in cfg.policy["_target_"])
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
    print(f"[dagger] policy rollout: {n_succ}/{len(records)} succeeded; "
          f"selected {len(inits)} states to relabel -> {path}")
    return path, n_succ


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

    # Point the demo dataset at a specific folder.
    override_dataset_path = cfg.get("override_dataset_path", None)
    if override_dataset_path:
        override_dataset_path = str(override_dataset_path)
        if not Path(override_dataset_path).is_dir():
            raise FileNotFoundError(
                f"override_dataset_path={override_dataset_path!r} is not a directory. It must "
                f"point at the demo dataset folder the WM was trained on.")
        with open_dict(cfg):
            cfg.dataset.data_path = override_dataset_path
        print(f"[dagger] override_dataset_path -> {override_dataset_path} "
              f"(demo dataset for action-norm stats, demo-goal evals and valid/train rollouts)")

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

    print(f"[dagger] out_dir={out_dir}\n"
          f"[dagger]   resume this run with: out_dir={out_dir}\n"
          f"[dagger] pool={pool_dir}\n"
          f"[dagger] rounds={cfg.rounds}  beta_schedule={cfg.beta_schedule}\n"
          f"[dagger] planner-driven episodes/round={cfg.trajectories_per_round}, "
          f"policy rollouts/round={cfg.rollout_episodes}")

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
    wandb_tags = [_env_name, str(cfg.beta_schedule), f"data={cfg.data_source}"]
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


    for r in range(int(cfg.rounds)):
        if _STOP:
            break
        rdir = rounds_dir / f"round_{r:02d}"
        gen_run, train_run = rdir / "gen", rdir / "train"
        done_marker, train_ckpt = rdir / "DONE", train_run / "checkpoints" / "model_final.pt"
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
        # data_source=original is the control arm.
        generate = str(cfg.data_source) != "original"
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
                f"++ckpt_base_path={cfg.ckpt_base_path}",
                f"++model_name={cfg.model_name}",
                f"++model_epoch={cfg.model_epoch}",
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
            n_new, n_pool = _absorb_shards(sub / "planned", pool_dir)
            n_new_shards += n_new
            _mf = sub / "planned" / "manifest.json"
            if _mf.is_file():
                _m = json.loads(_mf.read_text())
                gen_counts[tag] = {"trajectories": _m.get("trajectories"),
                                   "successes": _m.get("successes")}
                print(f"[dagger] round {r} ({tag}): +{n_new} shards, "
                      f"{_m.get('trajectories')} trajectories "
                      f"({_m.get('successes')} reached goal) -> pool has {n_pool} shards")
            else:
                print(f"[dagger] round {r} ({tag}): +{n_new} shards -> pool has {n_pool}")

        # 3. retrain on the whole pool. DAgger aggregates rather than replaces.
        _cap = cfg.get("train_max_trajectories", None)
        _cap = "null" if _cap is None else int(_cap)
        cmd = [
            sys.executable, "train_policy.py",
            # which train config to compose, as conf/<name>.yaml.
            f"--config-name={cfg.get('train_config_name', 'train_policy')}",
            f"hydra.run.dir={train_run}",
            f"++data_source={cfg.data_source}",
            f"++planned_data_path={pool_dir}",
            # the pool grows every round and planned_max_trajectories raises when it exceeds what
            # exists, so a fixed number would kill early rounds.
            f"++planned_max_trajectories={_cap}",
            f"++planned_selection={cfg.get('train_selection', 'first')}",
            # reseed the subset each round so 'random' selection actually resamples
            f"++planned_selection_seed={base_seed + r}",
        ] + [str(o) for o in (cfg.get("train_overrides") or [])]
        # demo dataset for action normalisation stats and demo-goal eval passes.
        if override_dataset_path:
            cmd += [f"++dataset.data_path={override_dataset_path}"]
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
        })
        print(f"[dagger] round {r}: +{_gen_traj} trajectories ({_gen_succ} usable) "
              f"-> pool {pool_traj} trajectories, {pool_succ} usable")
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"[dagger] round {r} done -> {summary_path}")

    print(f"\n[dagger] finished. pool={pool_dir}  summary={summary_path}\n"
          f"[dagger] DAgger returns the best round on validation, not the last. Compare the "
          f"per-round eval metrics before picking a checkpoint.")


@hydra.main(config_path="conf", config_name="dagger_wall", version_base="1.2")
def _main(cfg: OmegaConf):
    main(cfg)


if __name__ == "__main__":
    _main()
