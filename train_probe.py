"""Linear-probe study: how well can a FROZEN DINO embedding decode the sim STATE, and how much
does that depend on the representation, full patch tokens vs avg-pooled vs CLS?

- checkpoints/probe_<rep>.pt  (probe weights + state-normalization stats)
- summary.json  (final per-rep, per-component MAE/R^2)
- compare.png  (grouped bars: val R^2 per state component, per rep)
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")   # headless box, no DISPLAY

import json
import hydra
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate

from utils import seed as set_seed

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

# readable per-component names (falls back to s0, s1...
STATE_NAMES = {
    "pusht": ["agent_x", "agent_y", "block_x", "block_y", "angle", "agent_vx", "agent_vy"],
    "wall":  ["dot_x", "dot_y", "target_x", "target_y"],
    "point_maze": ["x", "y", "vx", "vy"],
}


@torch.no_grad()
def extract_features(encoder, images, device):
    """images (B, C, H, W) in [0,1] -> {'patches': (B,N,D), 'pooled': (B,D), 'cls': (B,D)}.

    match what DinoV2Encoder feeds downstream (same [0,1] -> [-1,1] normalization).
    """
    x = images.to(device).float()
    if x.max() > 1.5:
        x = x / 255.0
    x = encoder.normalization(x)   # [0,1] -> [-1,1], as in DinoV2Encoder.forward
    feats = encoder.base_model.forward_features(x)
    patches = feats["x_norm_patchtokens"]   # (B, N, D)
    return {
        "patches": patches,
        "pooled": patches.mean(dim=1),   # (B, D), exactly dino_avg_pool
        "cls": feats["x_norm_clstoken"],   # (B, D)
    }


def gather_features(traj_dset, encoder, reps, max_frames, device, batch_size, tag):
    """Encode up to max_frames (image, state) pairs from a trajectory dataset.
    Returns ({rep: (M, ...) float cpu tensor}, states (M, state_dim))."""
    index = []
    for t in range(len(traj_dset)):
        T = int(traj_dset.get_seq_length(t))
        index += [(t, f) for f in range(T)]
    rng = np.random.default_rng(0)
    if len(index) > max_frames:
        sel = sorted(rng.choice(len(index), size=int(max_frames), replace=False))
        index = [index[i] for i in sel]
    print(f"[{tag}] encoding {len(index)} (image, state) pairs from {len(traj_dset)} trajectories...")

    feats = {r: [] for r in reps}
    states = []
    img_buf, state_buf = [], []

    def flush():
        if not img_buf:
            return
        out = extract_features(encoder, torch.stack(img_buf), device)
        for r in reps:
            feats[r].append(out[r].float().cpu())
        states.append(torch.stack(state_buf))
        img_buf.clear(); state_buf.clear()

    for (t, f) in index:
        obs, _act, state, _info = traj_dset.get_frames(t, [f])
        img = obs["visual"][0]   # (C, H, W) in [0,1]
        img_buf.append(img if torch.is_tensor(img) else torch.as_tensor(np.asarray(img)))
        state_buf.append(torch.as_tensor(np.asarray(state[0])).float())
        if len(img_buf) >= batch_size:
            flush()
    flush()
    feats = {r: torch.cat(v, 0) for r, v in feats.items()}
    states = torch.cat(states, 0).float()
    return feats, states


class Probe(nn.Module):
    """Linear probe (probe_hidden=0) or a 1-hidden-layer MLP, with optional dropout."""
    def __init__(self, in_dim, out_dim, hidden=0, dropout=0.0):
        super().__init__()
        p = float(dropout)
        if hidden and int(hidden) > 0:
            self.net = nn.Sequential(
                nn.Dropout(p), nn.Linear(in_dim, int(hidden)), nn.GELU(),
                nn.Dropout(p), nn.Linear(int(hidden), out_dim))
        else:
            self.net = nn.Sequential(nn.Dropout(p), nn.Linear(in_dim, out_dim))

    def forward(self, x):
        return self.net(x)


def _flatten(f):
    return f.reshape(f.shape[0], -1)   # patches (M,N,D)->(M,N*D), pooled/cls (M,D)->(M,D)


def _eval_split(probe, x_d, y_raw_d, s_mean_d, s_std_d, y_n_d, loss_fn, ss_tot):
    """Evaluate the probe on one split -> (per-component MAE in RAW units, per-component R^2,
    normalized-MSE loss). Assumes probe.eval() + no_grad already set by the caller."""
    pred_n = probe(x_d)
    pred_raw = pred_n * s_std_d + s_mean_d
    mae = (pred_raw - y_raw_d).abs().mean(0)
    r2 = 1.0 - ((pred_raw - y_raw_d) ** 2).sum(0) / ss_tot
    return mae.cpu().numpy(), r2.cpu().numpy(), loss_fn(pred_n, y_n_d).item()


def train_one_probe(rep, tr_x, tr_y_n, va_x, va_y_n, s_mean, s_std, va_y_raw, names,
                    cfg, device, wandb_run, ckpt_dir):
    """Fit one probe (rep). Logs a TRAIN equivalent of every VAL metric (loss, per-component MAE,
    R^2), so the train-vs-val gap = overfitting is directly visible on wandb. best is on VAL."""
    in_dim, out_dim = tr_x.shape[1], tr_y_n.shape[1]
    dropout = float(cfg.get("probe_dropout", 0.0))
    probe = Probe(in_dim, out_dim, cfg.probe_hidden, dropout).to(device)
    n_params = sum(p.numel() for p in probe.parameters())
    opt = torch.optim.Adam(probe.parameters(), lr=cfg.optim.lr,
                           weight_decay=cfg.optim.weight_decay, betas=tuple(cfg.optim.betas))
    loss_fn = nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tr_x, tr_y_n), batch_size=int(cfg.batch_size),
        shuffle=True, drop_last=False)
    s_mean_d, s_std_d = s_mean.to(device), s_std.to(device)
    va_x_d, va_y_raw_d, va_y_n_d = va_x.to(device), va_y_raw.to(device), va_y_n.to(device)
    va_ss_tot = ((va_y_raw_d - va_y_raw_d.mean(0)) ** 2).sum(0).clamp_min(1e-8)
    # fixed TRAIN subset, same size as val for the train-vs-val comparison, the full train set is
    # too big to eval at once (esp.
    g = torch.Generator().manual_seed(0)
    n_te = min(len(tr_x), max(len(va_x), 2048))
    te_i = torch.randperm(len(tr_x), generator=g)[:n_te]
    te_x_d = tr_x[te_i].to(device)
    te_y_n_d = tr_y_n[te_i].to(device)
    te_y_raw_d = te_y_n_d * s_std_d + s_mean_d
    te_ss_tot = ((te_y_raw_d - te_y_raw_d.mean(0)) ** 2).sum(0).clamp_min(1e-8)

    print(f"[{rep}] probe in_dim={in_dim} params={n_params:,} train={len(tr_x)} val={len(va_x)} "
          f"(dropout={dropout} weight_decay={cfg.optim.weight_decay} lr={cfg.optim.lr})")
    best = {"val_mae_mean": float("inf"), "epoch": -1}
    last = {}   # metrics at the LAST eval (the TRAINED probe) -> over/under-fit is visible in summary
    for epoch in range(int(cfg.epochs) + 1):   # +1 so we log an epoch-0 (untrained) baseline
        # eval TRAIN(subset) + VAL, also at epoch 0, before any step
        if epoch % int(cfg.eval_freq) == 0 or epoch == int(cfg.epochs):
            probe.eval()
            with torch.no_grad():
                va_mae, va_r2, va_loss = _eval_split(
                    probe, va_x_d, va_y_raw_d, s_mean_d, s_std_d, va_y_n_d, loss_fn, va_ss_tot)
                tr_mae, tr_r2, tr_loss = _eval_split(
                    probe, te_x_d, te_y_raw_d, s_mean_d, s_std_d, te_y_n_d, loss_fn, te_ss_tot)
            last = {"epoch": epoch,
                    "train_r2_mean": float(tr_r2.mean()), "val_r2_mean": float(va_r2.mean()),
                    "train_mae_mean": float(tr_mae.mean()), "val_mae_mean": float(va_mae.mean()),
                    "train_r2": tr_r2.tolist(), "val_r2": va_r2.tolist()}
            if float(va_mae.mean()) < best["val_mae_mean"]:
                best = {"val_mae_mean": float(va_mae.mean()), "epoch": epoch,
                        "val_mae": va_mae.tolist(), "val_r2": va_r2.tolist(),
                        "train_mae": tr_mae.tolist(), "train_r2": tr_r2.tolist()}
                _save_probe(probe, s_mean, s_std, rep, names, ckpt_dir)   # keep the BEST (by val) checkpoint
            if wandb_run is not None:
                log = {"epoch": epoch,
                       f"{rep}/val_loss": va_loss, f"{rep}/val_mae_mean": float(va_mae.mean()),
                       f"{rep}/val_r2_mean": float(va_r2.mean()),
                       f"{rep}/train_loss": tr_loss, f"{rep}/train_mae_mean": float(tr_mae.mean()),
                       f"{rep}/train_r2_mean": float(tr_r2.mean()),
                       f"{rep}/overfit_gap_r2": float(tr_r2.mean() - va_r2.mean())}
                for k, nm in enumerate(names):
                    log[f"{rep}/val_mae_{nm}"] = float(va_mae[k])
                    log[f"{rep}/val_r2_{nm}"] = float(va_r2[k])
                    log[f"{rep}/train_mae_{nm}"] = float(tr_mae[k])
                    log[f"{rep}/train_r2_{nm}"] = float(tr_r2[k])
                wandb_run.log(log)
            print(f"[{rep}] epoch {epoch:4d}  train_r2={tr_r2.mean():+.3f} val_r2={va_r2.mean():+.3f}"
                  f"  train_mae={tr_mae.mean():.2f} val_mae={va_mae.mean():.2f}   (train>>val = overfit)")
        # snapshot the same (pre-train) model just evaluated, so probe_<rep>_e{N}.pt matches
        # epoch-N metrics
        if epoch % int(cfg.save_every) == 0:
            _save_probe(probe, s_mean, s_std, rep, names, ckpt_dir, name=f"probe_{rep}_e{epoch}.pt")
        if epoch == int(cfg.epochs):
            break
        # train one epoch
        probe.train()
        tot, nb = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(probe(xb.to(device)), yb.to(device))
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if wandb_run is not None:   # running SGD loss (dropout ON), separate from the clean train_loss above
            wandb_run.log({f"{rep}/train_running_loss": tot / max(1, nb), "epoch": epoch})

    # over/under-fit, compare the TRAINED probe's train vs val R^2 (last eval).
    _fit = ("OVERFIT" if last.get("train_r2_mean", -9) - last.get("val_r2_mean", 9) > 0.15
            else ("UNDERFIT(can't fit train)" if last.get("train_r2_mean", -9) < 0.2 else "healthy"))
    print(f"[{rep}] BEST val_mae_mean={best['val_mae_mean']:.3f} @ epoch {best['epoch']}  |  "
          f"TRAINED (epoch {last.get('epoch')}): train_r2={last.get('train_r2_mean'):+.3f} "
          f"val_r2={last.get('val_r2_mean'):+.3f} -> {_fit}")
    return {"rep": rep, "in_dim": in_dim, "n_params": n_params, **best, "final": last, "names": names}


def _save_probe(probe, s_mean, s_std, rep, names, ckpt_dir, name=None):
    torch.save({"model": probe.state_dict(), "state_mean": s_mean, "state_std": s_std,
                "rep": rep, "state_names": names}, ckpt_dir / (name or f"probe_{rep}.pt"))


def _compare_plot(results, names, path):
    import matplotlib.pyplot as plt
    reps = [r["rep"] for r in results]
    x = np.arange(len(names))
    w = 0.8 / max(1, len(reps))
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(names)), 4.5))
    for i, r in enumerate(results):
        ax.bar(x + i * w, np.clip(r["val_r2"], -0.1, 1.0), w, label=r["rep"])
    ax.set_xticks(x + w * (len(reps) - 1) / 2)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("val R^2 (higher = better decodable)")
    ax.set_title("How well each representation decodes the sim state\n(patches vs pooled vs cls)")
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150)


def main(cfg):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[probe] WARNING: no CUDA visible -> the DINO encoder will run on CPU and FAIL "
              "(xFormers attention has no CPU kernel). Run on a GPU (check CUDA_VISIBLE_DEVICES).")
    set_seed(int(cfg.seed))
    out_dir = Path(os.getcwd())
    ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    print(f"[probe] device={device}  out_dir={out_dir}")

    do_wandb = _HAS_WANDB and bool(cfg.get("wandb_logging", True))
    wandb_run = None
    if do_wandb:
        wandb_run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                               name=os.path.basename(str(out_dir)),
                               config=OmegaConf.to_container(cfg, resolve=True))

    # frozen encoder, must be DINO-family, exposes base_model + cls/patch tokens
    encoder = instantiate(cfg.encoder).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    if not hasattr(encoder, "base_model") or not hasattr(encoder, "normalization"):
        raise ValueError("probe needs a DINO-family encoder (base_model + normalization); "
                         f"got {cfg.encoder.get('_target_', '?')}. Use encoder=dino / dino_avg_pool.")
    n_enc = sum(p.numel() for p in encoder.parameters())
    n_enc_grad = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    # the encoder is FROZEN, eval() + requires_grad=False, and features are precomputed once below
    # so probe training never touches it.
    print(f"[probe] encoder FROZEN: {n_enc:,} params, {n_enc_grad} trainable (must be 0); "
          f"features precomputed once -> encoder never updated.")
    assert n_enc_grad == 0, "encoder is not fully frozen"

    # (image, state) pairs from the dataset's own train / valid splits
    _, traj_dset = hydra.utils.call(cfg.dataset, num_hist=cfg.num_hist,
                                    num_pred=cfg.num_pred, frameskip=cfg.frameskip)
    reps = list(cfg.reps)
    tr_feats, tr_states = gather_features(
        traj_dset["train"], encoder, reps, int(cfg.max_frames), device,
        int(cfg.encode_batch_size), "train")
    va_feats, va_states = gather_features(
        traj_dset["valid"], encoder, reps, int(cfg.max_frames) // 5, device,
        int(cfg.encode_batch_size), "valid")

    # target names + normalization (from TRAIN states)
    sd = tr_states.shape[1]
    names = STATE_NAMES.get(str(cfg.env_name), None)
    if names is None and str(cfg.env_name) == "puzzle":
        # Puzzle state width depends on the board size, so it cannot live in STATE_NAMES.
        _n_btn = int(getattr(traj_dset["train"], "meta", {}).get("num_buttons", 0))
        if _n_btn:
            names = [f"q{i}" for i in range(sd - _n_btn)] + [f"btn_{i}" for i in range(_n_btn)]
    if names is None or len(names) != sd:
        names = [f"s{i}" for i in range(sd)]
    s_mean = tr_states.mean(0)
    s_std = tr_states.std(0).clamp_min(1e-6)
    tr_y_n = (tr_states - s_mean) / s_std
    va_y_n = (va_states - s_mean) / s_std

    results = []
    for rep in reps:
        trx, vax = _flatten(tr_feats[rep]), _flatten(va_feats[rep])
        res = train_one_probe(
            rep, trx, tr_y_n, vax, va_y_n, s_mean, s_std, va_states, names,
            cfg, device, wandb_run, ckpt_dir)
        results.append(res)

    # summary + comparison plot
    _compare_plot(results, names, str(out_dir / "compare.png"))
    summary = {"env_name": str(cfg.env_name), "state_names": names,
               "n_train": int(len(tr_states)), "n_val": int(len(va_states)),
               "results": results}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== PROBE COMPARISON (val R^2, higher=better) ===")
    header = "rep".ljust(10) + "".join(n[:9].rjust(11) for n in names) + "     mean_mae"
    print(header)
    for r in results:
        row = r["rep"].ljust(10) + "".join(f"{v:11.3f}" for v in r["val_r2"])
        print(row + f"     {r['val_mae_mean']:8.3f}")
    # the headline number, patches-vs-pooled gap on position components
    print("\nRead the block_x/block_y (or dot) columns: if `pooled` << `patches`, avg-pooling is "
          "dropping the object position -> it caps BC/planning that run on the pooled rep.")

    if wandb_run is not None:
        for r in results:
            for nm, v in zip(names, r["val_r2"]):
                wandb_run.summary[f"{r['rep']}/final_val_r2_{nm}"] = float(v)
            wandb_run.summary[f"{r['rep']}/final_val_mae_mean"] = float(r["val_mae_mean"])
        wandb_run.log({"compare": wandb.Image(str(out_dir / "compare.png"))})
        wandb_run.finish()
    print(f"\n[probe] wrote {out_dir/'summary.json'} and {out_dir/'compare.png'}")


@hydra.main(config_path="conf", config_name="probe", version_base="1.2")
def _main(cfg):
    main(cfg)


if __name__ == "__main__":
    _main()
