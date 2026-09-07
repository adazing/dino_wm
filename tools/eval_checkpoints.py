"""Re-evaluate every saved checkpoint of a train_policy run and log a fresh wandb curve."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
import re
import json
import hydra
import torch
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

from utils import seed as set_seed
import train_policy as tp
from eval_policy import _build_train_dataset

# Top-level blocks chosen by Hydra's `defaults:` list (config GROUPS).
_GROUP_KEYS = ("env", "dataset", "encoder", "policy", "env_vars", "action_encoder",
               "proprio_encoder", "decoder", "predictor", "planner")


def _load_run_config(run_dir, cli_overrides):
    """Rebuild cfg from the training run's own saved config, then re-apply this command's CLI
    overrides on top."""
    saved = Path(run_dir) / ".hydra" / "config.yaml"
    if not saved.is_file():
        raise FileNotFoundError(
            f"{saved} not found -- run_dir must be a Hydra run folder (containing .hydra/). "
            f"Pass +use_run_config=false to compose from conf/ instead.")
    cfg = OmegaConf.load(saved)
    OmegaConf.set_struct(cfg, False)   # allow the extra run_dir / ckpt_* keys
    applied, skipped = [], []
    for ov in cli_overrides:
        if "=" not in ov:
            skipped.append(ov)   # deletions (~key) and other non-assignments
            continue
        key, _, raw = ov.partition("=")
        key = key.lstrip("+~")
        if key.startswith("hydra."):
            continue   # launcher/run-dir plumbing, not part of the job config
        if key in _GROUP_KEYS and OmegaConf.select(cfg, key) is not None:
            skipped.append(ov)   # group selection, the saved config already has it
            continue
        try:   # parse the value as YAML so false/null/25/[{...}] land as real types
            val = OmegaConf.create(f"_v: {raw}")._v
        except Exception:
            val = raw   # not YAML-parseable, keep the literal string
        OmegaConf.update(cfg, key, val, merge=False)
        applied.append(ov)
    return cfg, applied, skipped


def _find_checkpoints(run_dir, every=1, lo=None, hi=None):
    """The model_<epoch>.pt files under <run_dir>/checkpoints, sorted by epoch."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"no checkpoints/ directory under {run_dir} -- point run_dir at a train_policy "
            f"output folder (the one containing checkpoints/ and .hydra/).")
    found = []
    for p in sorted(ckpt_dir.glob("model_*.pt")):
        m = re.fullmatch(r"model_(\d+)\.pt", p.name)
        if m is None:
            continue   # model_final.pt and anything else unnumbered
        e = int(m.group(1))
        if (lo is None or e >= int(lo)) and (hi is None or e <= int(hi)):
            found.append((e, p))
    if not found:
        raise FileNotFoundError(f"no model_<epoch>.pt in {ckpt_dir} matching lo={lo} hi={hi}")
    found.sort(key=lambda t: t[0])
    every = max(1, int(every))
    if every > 1:
        kept = found[::every]
        if found[-1] not in kept:   # always include the newest checkpoint
            kept.append(found[-1])
        found = kept
    return found


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[recheck] WARNING: no CUDA -> DINO encoder will FAIL on CPU (xFormers). Run on a GPU.")
    set_seed(int(cfg.seed))
    out_dir = Path(os.getcwd())

    run_dir = cfg.get("run_dir", None)
    if not run_dir:
        raise ValueError("eval_checkpoints needs +run_dir=<a train_policy output folder>")

    # Rebuild cfg from the run's own saved config (default), so the architecture / data settings
    # match the checkpoints without retyping them.
    if bool(cfg.get("use_run_config", True)):
        cfg, applied, skipped = _load_run_config(run_dir, list(HydraConfig.get().overrides.task))
        print(f"[recheck] config rebuilt from {Path(run_dir) / '.hydra' / 'config.yaml'}")
        if applied:
            print(f"[recheck]   CLI overrides applied on top: {applied}")
        if skipped:
            print(f"[recheck]   IGNORED (already set by the run's config): {skipped}")
        set_seed(int(cfg.seed))   # the run's seed, which may differ from the composed default
        run_dir = cfg.get("run_dir", run_dir)

    ckpts = _find_checkpoints(run_dir, cfg.get("ckpt_every", 1),
                              cfg.get("ckpt_min_epoch", None), cfg.get("ckpt_max_epoch", None))
    print(f"[recheck] {len(ckpts)} checkpoints from {run_dir}: "
          f"epochs {[e for e, _ in ckpts]}")

    # Same runtime-derived goal_dim train_policy.main sets, it lives in no yaml, without it a
    # goal-conditioned checkpoint fails to load with a wte.weight shape mismatch.
    cfg.env.goal_dim = int(cfg.encoder.output_dim) if bool(cfg.get("goal_conditional", False)) else 0
    use_diffusion = "diffusion" in cfg.policy["_target_"]

    # build everything that does not depend on the checkpoint, exactly once
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # one policy instance, per checkpoint we only load_state_dict into it (same architecture)
    model = hydra.utils.instantiate(cfg.policy).to(device)
    if use_diffusion:
        from utils.normalizer import LinearNormalizer
        norm = LinearNormalizer()
        norm.fit(_build_train_dataset(cfg).get_all_actions())
        model.set_normalizer(norm)
        print("[recheck] fit diffusion action normalizer from training data")

    eval_env = tp.make_eval_env(cfg, cfg.n_envs)
    _, eval_traj_dset = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist,
                                         num_pred=cfg.num_pred, frameskip=cfg.frameskip)
    eval_task_traj_dset = None
    _task_spec = tp._resolve_task_dataset_spec(cfg)
    if _task_spec is not None:
        _, eval_task_traj_dset = hydra.utils.call(_task_spec, num_hist=cfg.num_hist,
                                                  num_pred=cfg.num_pred, frameskip=cfg.frameskip)

    # wandb, a new run by default, resume the original only when asked env/ on a new run mirrors
    # training's key names so the two curves overlay directly.
    resume_id = cfg.get("wandb_resume_id", None)
    prefix = str(cfg.get("metric_prefix", "env_ckpt" if resume_id else "env"))
    do_wandb = bool(cfg.get("wandb_logging", True))
    wandb_run = None
    if do_wandb:
        import wandb
        init_kwargs = dict(project=cfg.wandb.project, entity=cfg.wandb.entity,
                           config=OmegaConf.to_container(cfg, resolve=True))
        if resume_id:
            init_kwargs.update(id=str(resume_id), resume="allow")
        else:
            init_kwargs.update(name=f"recheck_{os.path.basename(str(run_dir).rstrip('/'))}")
        wandb_run = wandb.init(**init_kwargs)
        # plot against the checkpoint's epoch, not wandb's internal step, required when resuming,
        # the run's step counter is already past these epochs and clearer regardless.
        wandb.define_metric("epoch")
        wandb.define_metric(f"{prefix}/*", step_metric="epoch")

    rows = []
    try:
        for epoch, path in ckpts:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state = (payload["model"] if isinstance(payload, dict) and "model" in payload
                     else payload.state_dict() if hasattr(payload, "state_dict") else payload)
            model.load_state_dict(tp._fix_vqvae_keys(state))
            model.eval()
            # the epoch stamped inside the checkpoint is authoritative, the filename should agree
            stored = int(payload["epoch"]) if isinstance(payload, dict) and "epoch" in payload else epoch
            if stored != epoch:
                print(f"[recheck] NOTE: {path.name} stores epoch={stored}, using it over the filename")
            print(f"\n[recheck] === {path.name} (epoch={stored}) ===")

            metrics = tp.run_eval_passes(cfg, encoder, model, device, use_diffusion,
                                         epoch=stored, out_dir=out_dir, env=eval_env,
                                         traj_dset=eval_traj_dset,
                                         task_traj_dset=eval_task_traj_dset)
            for k in sorted(metrics):
                print(f"  {k:<48} {metrics[k]}")
            rows.append({"epoch": stored, "checkpoint": str(path), "metrics": metrics})
            if wandb_run is not None:
                wandb_run.log({**{f"{prefix}/{k}": v for k, v in metrics.items()}, "epoch": stored})
            # write after every checkpoint so a long sweep that dies partway keeps its results
            with open(out_dir / "eval_checkpoints.json", "w") as f:
                json.dump({"run_dir": str(run_dir), "metric_prefix": prefix,
                           "eval_passes": OmegaConf.to_container(cfg.get("eval_passes", []) or [],
                                                                 resolve=True),
                           "n_env_evals": int(cfg.n_env_evals), "results": rows}, f, indent=2)
    finally:
        eval_env.close()
        if wandb_run is not None:
            wandb_run.finish()

    print(f"\n[recheck] {len(rows)} checkpoints evaluated -> {out_dir/'eval_checkpoints.json'}")


@hydra.main(config_path="../conf", config_name="train_policy", version_base="1.2")
def _main(cfg: OmegaConf):
    main(cfg)


if __name__ == "__main__":
    _main()
