"""Evaluate a trained BC checkpoint in the REAL env on the configured eval_passes, no training.

- {name: task,  type: task,  horizon: 300, goal_source: valid}
"""
import os
import json
import hydra
import torch
from pathlib import Path
from omegaconf import OmegaConf

from utils import seed as set_seed
import train_policy as tp


def _ckpt_epoch(payload):
    if isinstance(payload, dict) and "epoch" in payload:
        return int(payload["epoch"])
    return 0


def _build_train_dataset(cfg):
    """Rebuild the training dataset (original/planned per data_source), only needed to fit the
    diffusion action normalizer, mirrors train_policy.main exactly."""
    original = (tp.build_raw_dino_traj_datasets(cfg.env.dataset, cfg.num_hist, cfg.num_pred, cfg.frameskip)
                if cfg.data_source in ("original", "both") else [])
    if original:
        original = [tp._select_original(original[0], cfg)]
    planned = []
    if cfg.data_source in ("planned", "both"):
        assert cfg.planned_data_path is not None, "planned_data_path required for planned/both"
        from datasets.planned_dset import PlannedTrajDataset
        from datasets.img_transforms import default_transform
        planned = [PlannedTrajDataset(
            cfg.planned_data_path, transform=default_transform(cfg.img_size),
            only_successes=bool(cfg.get("planned_only_successes", False)),
            max_trajectories=cfg.get("planned_max_trajectories", None))]
    return tp.make_raw_policy_traj(original, planned, cfg.data_source)


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[eval] WARNING: no CUDA -> DINO encoder will FAIL on CPU (xFormers). Run on a GPU.")
    set_seed(int(cfg.seed))
    out_dir = Path(os.getcwd())

    ckpt_path = cfg.get("checkpoint", None)
    if not ckpt_path:
        raise ValueError("eval_policy needs checkpoint=<path to a train_policy model_*.pt>")
    ckpt_payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    epoch = _ckpt_epoch(ckpt_payload)
    print(f"[eval] checkpoint: {ckpt_path} (epoch={epoch})")

    use_diffusion = "diffusion" in cfg.policy["_target_"]

    # goal-conditioned BC, goal_dim is not stored in any yaml, conf/env/*.yaml ships goal_dim, 0
    # train_policy.main derives it from goal_conditional at runtime, so cfg.policy's goal_dim,
    # ${env.goal_dim} resolves to the trained width.
    cfg.env.goal_dim = int(cfg.encoder.output_dim) if bool(cfg.get("goal_conditional", False)) else 0

    # frozen encoder
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # policy + load weights, same key-fix train_policy applies
    model = hydra.utils.instantiate(cfg.policy).to(device)
    state = (ckpt_payload["model"] if isinstance(ckpt_payload, dict) and "model" in ckpt_payload
             else ckpt_payload.state_dict() if hasattr(ckpt_payload, "state_dict") else ckpt_payload)
    state = tp._fix_vqvae_keys(state)
    model.load_state_dict(state)
    model.eval()
    print("[eval] policy weights loaded")

    # diffusion needs the action normalizer fit on the training data (VQ-BeT does not)
    if use_diffusion:
        from utils.normalizer import LinearNormalizer
        dataset = _build_train_dataset(cfg)
        norm = LinearNormalizer()
        norm.fit(dataset.get_all_actions())
        model.set_normalizer(norm)
        print("[eval] fit diffusion action normalizer from training data")

    # eval env + goal-sampling datasets, exactly as train_policy sets them up
    eval_env = tp.make_eval_env(cfg, cfg.n_envs)
    _, eval_traj_dset = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist,
                                         num_pred=cfg.num_pred, frameskip=cfg.frameskip)
    eval_task_traj_dset = None
    _task_spec = tp._resolve_task_dataset_spec(cfg)
    if _task_spec is not None:
        _, eval_task_traj_dset = hydra.utils.call(_task_spec, num_hist=cfg.num_hist,
                                                  num_pred=cfg.num_pred, frameskip=cfg.frameskip)

    do_wandb = bool(cfg.get("wandb_logging", False))
    wandb_run = None
    if do_wandb:
        import wandb
        wandb_run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                               name=f"eval_{os.path.basename(str(ckpt_path))}",
                               config=OmegaConf.to_container(cfg, resolve=True))

    try:
        metrics = tp.run_eval_passes(cfg, encoder, model, device, use_diffusion,
                                     epoch=epoch, out_dir=out_dir, env=eval_env,
                                     traj_dset=eval_traj_dset, task_traj_dset=eval_task_traj_dset)
    finally:
        eval_env.close()

    print("\n=== EVAL RESULTS ===")
    for k in sorted(metrics):
        print(f"  {k:<48} {metrics[k]}")
    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump({"checkpoint": str(ckpt_path), "epoch": epoch,
                   "eval_passes": OmegaConf.to_container(cfg.get("eval_passes", []) or [], resolve=True),
                   "n_env_evals": int(cfg.n_env_evals), "metrics": metrics}, f, indent=2)
    if wandb_run is not None:
        wandb_run.log({**{f"env/{k}": v for k, v in metrics.items()}, "epoch": epoch})
        wandb_run.finish()
    print(f"\n[eval] wrote {out_dir/'eval_metrics.json'}")


@hydra.main(config_path="conf", config_name="train_policy", version_base="1.2")
def _main(cfg: OmegaConf):
    main(cfg)


if __name__ == "__main__":
    _main()
