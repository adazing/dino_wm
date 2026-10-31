"""Sim-free world-model drift meter.

1. Fit a FROZEN linear probe (WM latent -> sim state) on the WM's own encoder space, so the probe
3. Probe both -> predicted positions. Report, per rollout step t:
- err_imagined(t) = |probe(z_imag_t) - true_state_t|  (WM drift + probe error)
- err_real(t)  = |probe(z_real_t) - true_state_t|  (probe error FLOOR, the baseline)
- wm_drift(t)  = |probe(z_imag_t) - probe(z_real_t)| (the WM's own contribution, probe-consistent)
- latent_drift(t) = ||z_imag_t - z_real_t||  (raw latent, the usual z_err_rollout)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import json
import hydra
import torch
import numpy as np
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf

from plan import load_model
from utils import seed as set_seed

STATE_NAMES = {
    "pusht": ["agent_x", "agent_y", "block_x", "block_y", "angle", "agent_vx", "agent_vy"],
    "wall":  ["dot_x", "dot_y", "target_x", "target_y"],
    "point_maze": ["x", "y", "vx", "vy"],
}
# state dims to summarize drift over, exclude single-frame-undecodable velocity for pusht and
# point_maze, a still image cannot show how fast the point mass is moving
POS_DIMS = {"pusht": [0, 1, 2, 3, 4], "wall": [0, 1, 2, 3], "point_maze": [0, 1]}


def puzzle_labels(model_cfg, sdim):
    """Names + task dims for the puzzle env, whose state width depends on the board size.

    Puzzle state is qpos | qvel | button_states. Only the BUTTONS are task state, the
    """
    env_type = str(OmegaConf.select(model_cfg, "env.kwargs.env_type") or "3x3")
    rows, cols = (int(v) for v in env_type.split("x"))
    n_btn = rows * cols
    names = [f"q{i}" for i in range(sdim - n_btn)] + [f"btn_{i}" for i in range(n_btn)]
    return names, list(range(sdim - n_btn, sdim))


def _load_wm(cfg, device):
    """Load the WM exactly as generate_planned_trajectories / plan do."""
    model_path = f"{cfg.ckpt_base_path}/outputs/{cfg.model_name}/"
    model_cfg = OmegaConf.load(os.path.join(model_path, "hydra.yaml"))
    if cfg.get("override_dataset_path"):
        from omegaconf import open_dict
        with open_dict(model_cfg):
            model_cfg.env.dataset.data_path = cfg.override_dataset_path
    model_ckpt = Path(model_path) / "checkpoints" / f"model_{cfg.model_epoch}.pth"
    model = load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device=device)
    model.eval()
    return model, model_cfg


def _rep_features(z_vis, rep):
    """Reduce a WM visual latent (..., P, D) to the chosen probe rep:
       pooled  -> (..., D)      mean over patches (== dino_avg_pool, also returns the single token
                                of a CLS/avg_pool WM where P==1)
       patches -> (..., P*D)    flatten all patch tokens (keeps spatial layout)
    cls is not a separate option: the WM predictor outputs patch tokens, not a cls token."""
    if rep == "pooled":
        return z_vis.mean(dim=-2)
    if rep == "patches":
        return z_vis.reshape(*z_vis.shape[:-2], z_vis.shape[-2] * z_vis.shape[-1])
    raise ValueError(f"rep must be 'pooled' or 'patches' (got {rep!r}); the WM latent has no separate "
                     "cls token to probe.")


@torch.no_grad()
def _encode_rep(model, visual, proprio, device, rep):
    """visual (b, t, C, H, W), proprio (b, t, Dp) -> visual latent reduced to `rep` (b, t, feat) via
    the WM's own encoder, so it matches the WM's rollout latent space exactly."""
    z = model.encode_obs({"visual": visual.to(device), "proprio": proprio.to(device)})
    return _rep_features(z["visual"], rep)


@torch.no_grad()
def fit_ridge_probe(model, dset, cfg, device):
    """Fit a frozen ridge probe (pooled WM latent -> state) on the WM's encoder space.
    Returns (probe dict, per-component train R^2, state_dim)."""
    rep = str(cfg.get("rep", "pooled"))
    index = [(t, f) for t in range(len(dset)) for f in range(int(dset.get_seq_length(t)))]
    rng = np.random.default_rng(int(cfg.seed))
    if len(index) > int(cfg.n_probe_frames):
        sel = sorted(rng.choice(len(index), int(cfg.n_probe_frames), replace=False))
        index = [index[i] for i in sel]
    print(f"[probe] fitting rep={rep} probe on {len(index)} (image, state) pairs "
          f"from {len(dset)} trajectories...")

    feats, states = [], []
    vbuf, pbuf, sbuf = [], [], []

    def flush():
        if not vbuf:
            return
        visual = torch.stack(vbuf).unsqueeze(1)   # , B, 1, C, H, W
        proprio = torch.stack(pbuf).unsqueeze(1)   # (B, 1, Dp)
        feat = _encode_rep(model, visual, proprio, device, rep).squeeze(1)   # (B, feat)
        feats.append(feat.float().cpu())
        states.append(torch.stack(sbuf))
        vbuf.clear(); pbuf.clear(); sbuf.clear()

    for (t, f) in index:
        obs, _act, state, _info = dset.get_frames(t, [f])
        vbuf.append(obs["visual"][0] if torch.is_tensor(obs["visual"]) else torch.as_tensor(obs["visual"][0]))
        pbuf.append(torch.as_tensor(np.asarray(obs["proprio"][0])).float())
        sbuf.append(torch.as_tensor(np.asarray(state[0])).float())
        if len(vbuf) >= int(cfg.probe_batch):
            flush()
    flush()

    X = torch.cat(feats, 0)   # (N, D)
    Y = torch.cat(states, 0).float()   # (N, S)
    Xm = X.mean(0)
    Xs = X.std(0).clamp_min(1e-6)
    Xn = (X - Xm) / Xs
    Xa = torch.cat([Xn, torch.ones(len(Xn), 1)], dim=1)   # bias column
    A = Xa.T @ Xa
    reg = float(cfg.ridge_lambda) * torch.eye(A.shape[0])
    reg[-1, -1] = 0.0   # do not regularize the bias
    W = torch.linalg.solve(A + reg, Xa.T @ Y)   # (D+1, S)
    Yhat = Xa @ W
    ss_res = ((Yhat - Y) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0).clamp_min(1e-8)
    r2 = (1.0 - ss_res / ss_tot).numpy()
    probe = {"Xm": Xm.to(device), "Xs": Xs.to(device), "W": W.to(device)}
    return probe, r2, Y.shape[1]


def apply_probe(probe, pooled):
    """pooled (n, D) -> predicted state (n, S), in raw state units."""
    Xn = (pooled - probe["Xm"]) / probe["Xs"]
    Xa = torch.cat([Xn, torch.ones(len(Xn), 1, device=Xn.device)], dim=1)
    return Xa @ probe["W"]


@torch.no_grad()
def rollout_drift(model, dset, probe, cfg, frameskip, num_hist, device):
    """For each of num_rollout trajectories, roll the WM open-loop and record per-step drift.
    Returns dict of per-step lists (indexed by rollout step t) for each metric."""
    rep = str(cfg.get("rep", "pooled"))
    rng = np.random.default_rng(int(cfg.seed) + 1)
    by_step = {"err_imag": {}, "err_real": {}, "wm_drift": {}, "latent_drift": {}}

    def _push(store, t, val):
        store.setdefault(t, []).append(val)

    done = 0
    tries = 0
    while done < int(cfg.num_rollout) and tries < 20 * int(cfg.num_rollout):
        tries += 1
        ti = int(rng.integers(0, len(dset)))
        obs, act, state, _ = dset[ti]
        T = obs["visual"].shape[0]
        full_h = (T - 1) // frameskip
        if full_h < int(cfg.min_horizon) + num_hist:
            continue
        horizon = full_h if cfg.get("max_steps") is None else min(full_h, int(cfg.max_steps))
        sl = slice(0, horizon * frameskip + 1, frameskip)
        vis = obs["visual"][sl]   # , horizon+1, C, H, W
        pro = obs["proprio"][sl]
        st = torch.as_tensor(np.asarray(state))[sl].float()   # (horizon+1, S)
        a = act[: horizon * frameskip]
        a = rearrange(a, "(h f) d -> h (f d)", f=frameskip)   # (horizon, f*Da)

        obs_0 = {"visual": vis[:num_hist].unsqueeze(0).to(device),
                 "proprio": pro[:num_hist].unsqueeze(0).to(device)}
        z_obses, _ = model.rollout(obs_0, a.unsqueeze(0).to(device))   # visual (1, horizon+1, P, D)
        pooled_imag = _rep_features(z_obses["visual"], rep).squeeze(0)   # (horizon+1, feat)
        pooled_real = _encode_rep(model, vis.unsqueeze(0), pro.unsqueeze(0), device, rep).squeeze(0)
        # align lengths (rollout returns num_hist + (horizon-num_hist) + 1 == horizon+1 frames)
        n = min(pooled_imag.shape[0], pooled_real.shape[0], st.shape[0])
        pred_imag = apply_probe(probe, pooled_imag[:n]).cpu()   # (n, S) raw units
        pred_real = apply_probe(probe, pooled_real[:n]).cpu()
        true = st[:n]
        for t in range(n):
            _push(by_step["err_imag"], t, (pred_imag[t] - true[t]).abs().numpy())
            _push(by_step["err_real"], t, (pred_real[t] - true[t]).abs().numpy())
            _push(by_step["wm_drift"], t, (pred_imag[t] - pred_real[t]).abs().numpy())
            _push(by_step["latent_drift"], t,
                  float((pooled_imag[t].cpu() - pooled_real[t].cpu()).norm()))
        done += 1
    print(f"[drift] rolled {done} trajectories (num_hist={num_hist} context frames)")
    return by_step


def _agg(step_dict, pos_dims):
    """{t: [per-component array]} -> (steps, mean over pos dims per step, per-component means)."""
    steps = sorted(step_dict.keys())
    pos_mean, comp_mean = [], []
    for t in steps:
        arr = np.stack(step_dict[t])   # (n_traj, S)
        comp_mean.append(arr.mean(0))
        pos_mean.append(arr[:, pos_dims].mean())
    return steps, np.array(pos_mean), np.array(comp_mean)


def _plot(steps, pos_imag, pos_real, wm, lat_steps, lat, names, pos_dims, path):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(steps, pos_imag, "-o", ms=3, label="imagined vs true (WM drift + probe err)")
    ax1.plot(steps, pos_real, "-s", ms=3, label="real-latent probe vs true (probe floor)")
    ax1.plot(steps, wm, "-^", ms=3, label="imagined vs real-latent (pure WM drift)")
    ax1.set_xlabel("open-loop rollout step"); ax1.set_ylabel("mean |pos error| (state units)")
    pos_names = ", ".join(names[d] for d in pos_dims)
    ax1.set_title(f"Position drift over open-loop rollout\n(mean over: {pos_names})")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.plot(lat_steps, lat, "-o", ms=3, color="tab:red")
    ax2.set_xlabel("open-loop rollout step"); ax2.set_ylabel("mean ||z_imag - z_real||")
    ax2.set_title("Raw latent drift (z_err_rollout)")
    ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150)


def main(cfg):
    device = torch.device(str(cfg.device) if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[drift] WARNING: no CUDA -> DINO encoder will FAIL on CPU (xFormers has no CPU "
              "kernel). Run on a GPU.")
    set_seed(int(cfg.seed))
    out_dir = Path(os.getcwd())

    model, model_cfg = _load_wm(cfg, device)
    frameskip = int(model_cfg.frameskip)
    num_hist = int(model_cfg.num_hist)
    env_name = str(model_cfg.env.name)
    print(f"[drift] WM loaded: env={env_name} frameskip={frameskip} num_hist={num_hist}")

    _, traj_dset = hydra.utils.call(model_cfg.env.dataset, num_hist=model_cfg.num_hist,
                                    num_pred=model_cfg.num_pred, frameskip=frameskip)
    probe_dset = traj_dset["train"] if str(cfg.probe_split) == "train" else traj_dset["valid"]
    drift_dset = traj_dset["train"] if str(cfg.source_split) == "train" else traj_dset["valid"]

    probe, r2, sdim = fit_ridge_probe(model, probe_dset, cfg, device)
    if env_name == "puzzle":
        names, pos_dims = puzzle_labels(model_cfg, sdim)
    else:
        names = STATE_NAMES.get(env_name, [f"s{i}" for i in range(sdim)])
        if len(names) != sdim:
            names = [f"s{i}" for i in range(sdim)]
        pos_dims = POS_DIMS.get(env_name, list(range(sdim)))
    print("[probe] train R^2 per component:")
    for nm, v in zip(names, r2):
        print(f"    {nm:<10} {v:+.3f}")
    if np.mean([r2[d] for d in pos_dims]) < 0.5:
        print("[probe] WARNING: low position R^2 -> the probe is a weak position decoder, so the "
              "drift numbers below are noisy. Increase n_probe_frames or check the encoder.")

    by_step = rollout_drift(model, drift_dset, probe, cfg, frameskip, num_hist, device)
    steps, pos_imag, comp_imag = _agg(by_step["err_imag"], pos_dims)
    _, pos_real, comp_real = _agg(by_step["err_real"], pos_dims)
    _, pos_wm, comp_wm = _agg(by_step["wm_drift"], pos_dims)
    lat_steps = sorted(by_step["latent_drift"].keys())
    lat = np.array([np.mean(by_step["latent_drift"][t]) for t in lat_steps])

    _plot(steps, pos_imag, pos_real, pos_wm, lat_steps, lat, names, pos_dims,
          str(out_dir / "drift_curve.png"))

    summary = {
        "model_name": str(cfg.model_name), "env": env_name,
        "frameskip": frameskip, "num_hist": num_hist,
        "probe_split": str(cfg.probe_split), "source_split": str(cfg.source_split),
        "n_rollouts": len(by_step["latent_drift"].get(0, [])),
        "state_names": names, "pos_dims": pos_dims,
        "probe_train_r2": {nm: float(v) for nm, v in zip(names, r2)},
        "steps": [int(s) for s in steps],
        "pos_err_imagined": pos_imag.tolist(),   # WM drift + probe floor
        "pos_err_real_floor": pos_real.tolist(),   # probe error floor
        "pos_wm_drift": pos_wm.tolist(),   # pure WM contribution
        "latent_drift": lat.tolist(),
        "per_component_wm_drift": {nm: comp_wm[:, k].tolist() for k, nm in enumerate(names)},
    }
    with open(out_dir / "drift.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== DRIFT (mean over position dims, by rollout step) ===")
    print("step   imagined_vs_true   probe_floor   pure_wm_drift   latent_drift")
    for i, t in enumerate(steps):
        lt = lat[lat_steps.index(t)] if t in lat_steps else float("nan")
        print(f"{t:4d}   {pos_imag[i]:14.3f}   {pos_real[i]:11.3f}   {pos_wm[i]:13.3f}   {lt:12.3f}")
    print(f"\n[drift] wrote {out_dir/'drift.json'} and {out_dir/'drift_curve.png'}")


@hydra.main(config_path="../conf", config_name="drift", version_base="1.2")
def _main(cfg):
    main(cfg)


if __name__ == "__main__":
    _main()
