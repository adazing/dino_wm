"""Fixed-difficulty yardsticks for an existing DAgger run, computed after the fact.

The per-round numbers the loop logs are measured on a moving target: the planner is scored on the
policy's failures (harder every round) and the policy is scored on freshly drawn tasks. This script
re-scores every saved checkpoint of a run on ONE FIXED set of tasks, so rounds AND runs compare.

  Part A  world model: every round's world model plans the same N env-sampled tasks.
          Falls  -> the world model really got worse.
          Flat   -> any drop in the loop's planner success was just the exam getting harder.
  Part B  policy: every round's policy attempts the same M env-sampled tasks.
          Reports success ever-reached (what the loop logs) AND at the final step.
  Part C  (optional, --relabel) teacher vs student on the planner's own relabel states, via
          tools/eval_policy_on_relabel.py if that file exists.

Nothing is trained, nothing is added to the pool. Results land in <run>/yardsticks/.

Usage, from the dino_wm folder on the server:
  python eval_yardsticks.py --run 20261007-004253
  python eval_yardsticks.py --run /nas/ada/dev/dino_wm/dagger/<out_dir>
  # same seeds -> directly comparable with any other run:
  python eval_yardsticks.py --run <other run>
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys
import csv
import json
import argparse
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent
if not (REPO / "dagger_loop.py").is_file() and (REPO.parent / "dagger_loop.py").is_file():
    REPO = REPO.parent   # allow running it from tools/
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

# keys this script sets on the generation command; any gen_overrides entry for them is dropped so
# Hydra never sees two values for one key
_GEN_KEYS_WE_SET = ("goal_source", "planned_only_successes", "num_trajectories", "seed",
                    "eval_seed_base", "wandb_logging", "shard_every", "planned_out_path",
                    "ckpt_base_path", "model_name", "model_epoch", "provided_states_path")


# --------------------------------------------------------------------------- locating the run
def find_run(arg, root):
    p = Path(arg)
    if p.is_dir():
        return p.resolve()
    rid = str(arg)
    for pre in ("wall-", "pusht-"):
        if rid.startswith(pre):
            rid = rid[len(pre):]
    rid = rid.replace("-loop", "")
    hits = [d for d in Path(root).iterdir()
            if (d / "run_id.txt").is_file() and (d / "run_id.txt").read_text().strip() == rid]
    if not hits:
        raise FileNotFoundError(f"no run under {root} has run_id.txt == {rid!r}. "
                                f"Pass the out_dir path with --run instead.")
    if len(hits) > 1:
        raise RuntimeError(f"several runs share run id {rid}: {hits}")
    return hits[0].resolve()


def load_loop_cfg(out_dir, cfg_path=None):
    """The resolved config the loop itself ran with, as saved by its wandb run."""
    if cfg_path is None:
        cands = [out_dir / "wandb" / "latest-run" / "files" / "config.yaml"]
        cands += sorted((out_dir / "wandb").glob("run-*/files/config.yaml"), reverse=True)
        cfg_path = next((c for c in cands if c.is_file()), None)
        if cfg_path is None:
            raise FileNotFoundError(
                f"no wandb config under {out_dir}/wandb. Pass --cfg <a yaml with the dagger "
                f"config> (e.g. conf/dagger_wall.yaml, but note it may have changed since).")
    raw = yaml.safe_load(Path(cfg_path).read_text())
    # wandb stores each key as {"desc": ..., "value": ...}, plus a few bookkeeping entries
    # (_wandb, wandb_version) that are not wrapped. Unwrap key by key, never all-or-nothing.
    def _is_wrapped(v):
        return isinstance(v, dict) and "value" in v and set(v) <= {"desc", "value"}
    raw = {k: (v["value"] if _is_wrapped(v) else v) for k, v in raw.items()
           if k not in ("_wandb", "wandb_version")}
    cfg = OmegaConf.create(raw)
    OmegaConf.set_struct(cfg, False)
    print(f"[cfg] {cfg_path}")
    return cfg


def round_dirs(out_dir):
    return sorted((d for d in (out_dir / "rounds").glob("round_*") if d.is_dir()),
                  key=lambda d: int(d.name.split("_")[1]))


# --------------------------------------------------------------------------- part A: world models
def world_models(cfg, out_dir):
    """(label, round, ckpt_base_path, model_name, model_epoch) for the original world model and
    every fine-tuned one. 'after r3' is the model round 3 produced, i.e. what round 4 planned with."""
    out = [("original", -1, str(cfg.ckpt_base_path), str(cfg.model_name), str(cfg.model_epoch))]
    for rd in round_dirs(out_dir):
        r = int(rd.name.split("_")[1])
        if (rd / "outputs" / "wm" / "checkpoints" / "model_latest.pth").is_file():
            out.append((f"after r{r}", r, str(rd), "wm", "latest"))
    return out


def eval_world_model(cfg, label, base, name, epoch, n_tasks, seed, ydir, python):
    tag = "wm_original" if label == "original" else f"wm_{label.replace(' ', '_')}"
    sub = ydir / tag
    mf = sub / "planned" / "manifest.json"
    if mf.is_file():
        print(f"[A] {label}: cached ({mf})")
    else:
        cmd = [python, str(REPO / "generate_planned_trajectories.py"),
               f"--config-name={cfg.gen_config_name}",
               f"hydra.run.dir={sub}",
               f"++ckpt_base_path={base}", f"++model_name={name}", f"++model_epoch={epoch}",
               "++planned_out_path=./planned", "++shard_every=1",
               f"++num_trajectories={n_tasks}",
               "++goal_source=random_state",
               # keep failures too, otherwise the manifest would only count successes (rate = 1)
               "++planned_only_successes=false",
               # identical seeds for every model and every run -> identical tasks
               f"++seed={seed}", f"++eval_seed_base={seed}",
               "++wandb_logging=false"]
        if cfg.get("override_dataset_path"):
            cmd.append(f"++override_dataset_path={cfg.override_dataset_path}")
        if bool(cfg.get("serial_env", False)):
            cmd.append("++serial_env=true")
        if cfg.get("env_start_method"):
            cmd.append(f"++env_start_method={cfg.env_start_method}")
        for o in (cfg.get("gen_overrides") or []):
            key = str(o).lstrip("+").split("=", 1)[0]
            if key not in _GEN_KEYS_WE_SET:
                cmd.append(str(o))
        sub.mkdir(parents=True, exist_ok=True)
        log = ydir / f"{tag}.log"
        print(f"[A] {label}: planning {n_tasks} fixed tasks (log: {log})")
        with open(log, "w") as f:
            f.write("$ " + " ".join(cmd) + "\n\n")
            f.flush()
            rc = subprocess.run(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT).returncode
        if rc != 0 or not mf.is_file():
            lines = log.read_text(errors="replace").splitlines()
            tail = [l for l in lines if "Warning" not in l and "warn(" not in l][-25:]
            print(f"[A] {label}: FAILED (exit {rc}). Command and last lines of {log}:")
            print("    $ " + " ".join(cmd))
            print("\n".join("    | " + l for l in tail))
            return {"wm_success": None, "wm_latent_err": None, "failed": True}
    m = json.loads(mf.read_text())
    return {"wm_success": m.get("success_rate"), "wm_latent_err": m.get("wm_latent_err"),
            "wm_latent_err_median": m.get("wm_latent_err_median")}


# --------------------------------------------------------------------------- part B: policies
def policies(out_dir):
    out = []
    for rd in round_dirs(out_dir):
        ck = rd / "train" / "checkpoints" / "model_final.pt"
        if ck.is_file():
            out.append((int(rd.name.split("_")[1]), ck))
    return out


def policy_cfg(train_dir, loop_cfg):
    """The config the policy was actually TRAINED with (train_policy's own .hydra/config.yaml), with
    the loop's rollout settings laid on top. The loop's wandb config is not enough on its own: it
    lacks keys the policy side needs, e.g. encoder.output_dim."""
    import train_policy  # noqa: F401  registers the ${eval:...} resolver the policy config uses
    p = Path(train_dir) / ".hydra" / "config.yaml"
    if not p.is_file():
        print(f"[B]   no {p}, falling back to the loop config")
        return loop_cfg.copy()
    c = OmegaConf.load(p)
    OmegaConf.set_struct(c, False)
    # rollout_* belong to the loop, not to train_policy, so take them from the loop config
    for k in ("rollout_goal_source", "rollout_horizon", "rollout_goal_offset"):
        if loop_cfg.get(k) is not None:
            c[k] = loop_cfg[k]
    return c


def eval_policy(cfg, ckpt, n_episodes, seed):
    """Same rollout the loop's scouting step runs (policy_rollout.rollout_policy), on fixed seeds."""
    import hydra
    import policy_rollout as pr
    import train_policy as tp

    if cfg.get("encoder") is None or cfg.encoder.get("output_dim") is None:
        raise KeyError(f"policy config has no encoder.output_dim; keys under encoder: "
                       f"{list((cfg.get('encoder') or {}).keys())}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.env.goal_dim = int(cfg.encoder.output_dim) if bool(cfg.get("goal_conditional", False)) else 0
    encoder = hydra.utils.instantiate(cfg.encoder).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    policy = hydra.utils.instantiate(cfg.policy).to(device)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = (payload["model"] if isinstance(payload, dict) and "model" in payload
             else payload.state_dict() if hasattr(payload, "state_dict") else payload)
    policy.load_state_dict(tp._fix_vqvae_keys(state))
    policy.eval()

    env = tp.make_eval_env(cfg, cfg.n_envs)
    final_succ = None
    try:
        recs = pr.rollout_policy(
            cfg, encoder, policy, device, env, n_episodes=n_episodes,
            horizon=int(cfg.get("rollout_horizon") or cfg.eval_horizon),
            base_seed=seed, use_diffusion="diffusion" in cfg.policy["_target_"])
        # success at the LAST step, in batches of n_envs because that is what the env expects
        try:
            n_envs, final_succ = int(cfg.n_envs), []
            for i in range(0, len(recs), n_envs):
                chunk = recs[i:i + n_envs]
                g = [np.asarray(r["goal_state"]) for r in chunk]
                f = [np.asarray(r["states"][-1]) for r in chunk]
                while len(g) < n_envs:
                    g.append(g[-1]); f.append(f[-1])
                res = env.eval_state(np.stack(g), np.stack(f))
                final_succ += list(np.asarray(res["success"]).astype(bool)[:len(chunk)])
        except Exception as e:
            print(f"[B]   final-step success unavailable ({e})")
            final_succ = None
    finally:
        env.close()
        del policy, encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    dists = [r["min_dist"] for r in recs if np.isfinite(r["min_dist"])]
    return {"policy_success_ever": float(np.mean([r["success"] for r in recs])),
            "policy_success_final": (float(np.mean(final_succ)) if final_succ else None),
            "policy_closest_dist": (float(np.mean(dists)) if dists else None),
            "policy_episodes": len(recs)}


# --------------------------------------------------------------------------- part C: relabel
def eval_relabel(out_dir, r, ydir, python, horizons):
    tool = REPO / "tools" / "eval_policy_on_relabel.py"
    if not tool.is_file():
        return None
    if not (out_dir / "rounds" / f"round_{r:02d}" / "relabel_states.pth").is_file():
        return None
    out = ydir / f"relabel_r{r:02d}.json"
    if not out.is_file():
        log = ydir / f"relabel_r{r:02d}.log"
        cmd = [python, str(tool), f"--run_dir={out_dir}", f"--rounds={r}",
               f"--horizons={horizons}", f"--out={out}"]
        with open(log, "w") as f:
            rc = subprocess.run(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT).returncode
        if rc != 0 or not out.is_file():
            print(f"[C] round {r}: FAILED (exit {rc}), see {log}")
            return None
    rows = json.loads(out.read_text())
    if not rows:
        return None
    row, h = rows[-1], horizons.split(",")[0]
    return {"relabel_planner": row.get("planner_success_rate"),
            f"relabel_policy_h{h}": row.get(f"policy_max_success_h{h}"),
            "relabel_n": row.get("n_states")}


# --------------------------------------------------------------------------- main
def fmt(v, nd=3):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", required=True, help="run id (e.g. 20261007-004253) or out_dir path")
    ap.add_argument("--dagger_root", default="/nas/ada/dev/dino_wm/dagger")
    ap.add_argument("--cfg", default=None, help="override the config (default: the run's own)")
    ap.add_argument("--wm_tasks", type=int, default=50, help="fixed planning tasks per world model")
    ap.add_argument("--policy_episodes", type=int, default=200, help="fixed episodes per policy")
    # far from the loop's own seeds (seed + r*10_000 / r*100_000), and identical across runs
    ap.add_argument("--wm_seed", type=int, default=7_654_321)
    ap.add_argument("--policy_seed", type=int, default=1_234_567)
    ap.add_argument("--skip_wm", action="store_true")
    ap.add_argument("--skip_policy", action="store_true")
    ap.add_argument("--relabel", action="store_true", help="also run part C")
    ap.add_argument("--horizons", default="25,300")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    out_dir = find_run(a.run, a.dagger_root)
    cfg = load_loop_cfg(out_dir, a.cfg)
    ydir = out_dir / "yardsticks"
    ydir.mkdir(exist_ok=True)
    print(f"[run] {out_dir}\n[out] {ydir}")

    rows = {}   # keyed by round, -1 = before any training

    if not a.skip_wm:
        wms = world_models(cfg, out_dir)
        if len(wms) == 1:
            print("[A] no fine-tuned world models in this run (wm_finetune off?), "
                  "scoring only the original")
        if not cfg.get("gen_config_name"):
            print("[A] the run config has no gen_config_name, skipping part A "
                  "(pass --cfg with a dagger config that sets it)")
            wms = []
        for label, r, base, name, epoch in wms:
            res = eval_world_model(cfg, label, base, name, epoch, a.wm_tasks, a.wm_seed, ydir, a.python)
            failed = res.pop("failed", False)
            rows.setdefault(r, {"round": r}).update({"wm": label, **res})
            print(f"[A] {label:9s} success={fmt(res['wm_success'])} "
                  f"latent_err={fmt(res['wm_latent_err'], 1)}")
            if failed and label == "original":
                # the original model is the simplest case; if it fails, every other will too
                print("[A] stopping part A: fix the error above, then re-run (finished parts are cached)")
                break

    if not a.skip_policy:
        for r, ck in policies(out_dir):
            cache = ydir / f"policy_r{r:02d}.json"
            if cache.is_file():
                res = json.loads(cache.read_text())
                print(f"[B] round {r}: cached")
            else:
                print(f"[B] round {r}: {a.policy_episodes} fixed episodes ({ck})")
                res = eval_policy(policy_cfg(ck.parent.parent, cfg), ck,
                                  a.policy_episodes, a.policy_seed)
                cache.write_text(json.dumps(res, indent=2))
            rows.setdefault(r, {"round": r}).update(res)
            print(f"[B] round {r}: ever={fmt(res['policy_success_ever'])} "
                  f"final={fmt(res['policy_success_final'])}")

    if a.relabel:
        for r in sorted(k for k in rows if k >= 0):
            res = eval_relabel(out_dir, r, ydir, a.python, a.horizons)
            if res:
                rows[r].update(res)

    table = [rows[k] for k in sorted(rows)]
    keys = []
    for t in table:
        keys += [k for k in t if k not in keys]
    with open(ydir / "yardsticks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(table)
    (ydir / "yardsticks.json").write_text(json.dumps(
        {"run": str(out_dir), "wm_tasks": a.wm_tasks, "wm_seed": a.wm_seed,
         "policy_episodes": a.policy_episodes, "policy_seed": a.policy_seed,
         "rows": table}, indent=2))

    print("\n" + "=" * 92)
    print(f"{out_dir.name}   (fixed tasks: WM seed {a.wm_seed}, policy seed {a.policy_seed})")
    print("=" * 92)
    print(f"{'round':>5} | {'world model':>11} {'WM success':>10} {'WM err':>7} | "
          f"{'policy (ever)':>13} {'policy (final)':>14} {'closest dist':>12}")
    print("-" * 92)
    for t in table:
        print(f"{('pre' if t['round'] < 0 else t['round']):>5} | {t.get('wm', '—'):>11} "
              f"{fmt(t.get('wm_success')):>10} {fmt(t.get('wm_latent_err'), 1):>7} | "
              f"{fmt(t.get('policy_success_ever')):>13} {fmt(t.get('policy_success_final')):>14} "
              f"{fmt(t.get('policy_closest_dist'), 2):>12}")
    print("-" * 92)
    print("WM row r = the model fine-tuned at the end of round r (what round r+1 planned with).\n"
          "Policy row r = the policy trained in round r.\n"
          f"Written: {ydir / 'yardsticks.csv'}")


if __name__ == "__main__":
    main()