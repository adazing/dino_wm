"""Score a finished DAgger run's policy on the SAME start/goal pairs the planner was given.

The per-round planner success rate is measured on the states the policy failed at, and those
states get harder every round as the policy improves. The policy's own goal25 pass is measured on
ordinary demo segments, whose difficulty never changes. Comparing the two numbers therefore says
little. This replays each round's saved policy against that round's saved relabel states, so both
sides are scored on identical problems.

Reads only, nothing is written back into the run directory.

    python tools/eval_policy_on_relabel.py --run_dir /nas/ada/dev/dino_wm/dagger/<run>
    python tools/eval_policy_on_relabel.py --run_dir <run> --rounds 0,4,9 --horizons 25,300
"""
import argparse
import json
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from omegaconf import OmegaConf


def load_run_cfg(run_dir: Path, rnd=None):
    """The config the POLICY was trained with, for the round being replayed.

    B8, fixed 2026-11-24. This used to read only the wandb snapshot at
    wandb/*/files/config.yaml, and raised when it was absent -- which is every offline run.
    Verified on a live arm: the wandb directories exist but their files/ holds only
    conda-environment.yaml, requirements.txt and wandb-metadata.json. The sbatch script sets
    WANDB_MODE=offline by default, so this diagnostic had been failing with exit 1 on every round
    of every arm, losing the only planner-versus-policy comparison on identical tasks.

    The training subprocess writes its own Hydra snapshot to
    rounds/round_NN/train/.hydra/config.yaml, and every key this tool reads is a train_policy-level
    setting (action_window_size, window_size, num_hist, num_pred, frameskip, img_size, encoder,
    policy, env.dataset, env.goal_dim, n_envs, goal_conditional). So that snapshot is both
    available and more authoritative than the wandb copy, because it records the settings the
    checkpoint was actually built with. Prefer the round being replayed, fall back to any round,
    and keep the wandb path for runs recorded online.
    """
    cands = []
    if rnd is not None:
        cands.append(run_dir / "rounds" / f"round_{int(rnd):02d}" / "train" / ".hydra"
                     / "config.yaml")
    cands += sorted(run_dir.glob("rounds/round_*/train/.hydra/config.yaml"), reverse=True)
    for c in cands:
        if c.is_file():
            print(f"[cfg] loaded from {c}")
            return OmegaConf.load(c)

    # Online runs: wandb writes its own copy, with entries shaped {key: {desc, value}}.
    hits = sorted(run_dir.glob("wandb/*/files/config.yaml"))
    if hits:
        print(f"[cfg] loaded from {hits[-1]} (wandb snapshot)")
        raw = OmegaConf.to_container(OmegaConf.load(hits[-1]), resolve=False)
        flat = {k: (v.get("value") if isinstance(v, dict) and "value" in v else v)
                for k, v in raw.items() if not k.startswith("_")}
        return OmegaConf.create(flat)

    raise FileNotFoundError(
        f"no launch config found under {run_dir}. Looked for "
        f"rounds/round_*/train/.hydra/config.yaml (written by the training subprocess) and "
        f"wandb/*/files/config.yaml (written only by ONLINE wandb runs). Without one of these "
        f"the run cannot be replayed.")


def planner_rate(run_dir: Path, rnd: int):
    """The planner's success rate for this round, from the run's own summary."""
    f = run_dir / "dagger_summary.json"
    if not f.is_file():
        return None
    for rec in json.loads(f.read_text()).get("rounds", []):
        if int(rec.get("round", -1)) != rnd:
            continue
        gen = rec.get("generated") or {}
        n = sum(v.get("trajectories") or 0 for v in gen.values())
        s = sum(v.get("successes") or 0 for v in gen.values())
        return (s / n) if n else None
    return None


def rollout(cfg, policy, encoder, env, device, init_states, goal_states, env_infos, horizon,
            use_diffusion, act_mean, act_std, seed0=0):
    """Run the policy from each init toward its goal. Mirrors train_policy.eval_on_env's loop.

    Returns (ever_reached, reached_at_end), each a bool array over the episodes.
    """
    import train_policy as tp

    n_envs = int(cfg.n_envs)
    n = len(init_states)
    ever = np.zeros(n, dtype=bool)
    final = np.zeros(n, dtype=bool)

    # Identical to train_policy.eval_on_env's embed. The resize/normalize transform is not
    # optional, the encoder asserts a [0,1] range and was trained on transformed frames.
    from einops import rearrange
    from datasets.img_transforms import default_transform
    tf = default_transform(cfg.img_size)

    def embed(vis):
        with torch.no_grad():
            v = torch.as_tensor(np.ascontiguousarray(vis)).float()
            if v.max() > 1.5:          # envs return [0,255] or [0,1], detect which
                v = v / 255.0
            v = rearrange(v, "b h w c -> b c h w")
            v = tf(v).to(device)
            return encoder(v).reshape(v.shape[0], 1, encoder.n_patches, encoder.emb_dim)

    for lo in range(0, n, n_envs):
        hi = min(lo + n_envs, n)
        real = hi - lo
        # the vector env always needs exactly n_envs entries, pad and discard the tail
        pad = lambda a: np.concatenate([a, np.repeat(a[-1:], n_envs - real, axis=0)]) \
            if real < n_envs else a
        ini, gol = pad(np.asarray(init_states[lo:hi])), pad(np.asarray(goal_states[lo:hi]))
        seeds = [seed0 + lo + i for i in range(n_envs)]

        infos = env_infos[lo:hi] if env_infos else []
        if infos:
            blk = list(infos) + [infos[-1]] * (n_envs - real)
            if all(isinstance(e, dict) and "fix_door_location" in e for e in blk):
                env.update_env(blk)

        goal_emb = None
        if int(cfg.env.goal_dim) > 0:
            goal_obs, _ = env.prepare(seeds, gol)
            goal_emb = embed(goal_obs["visual"])

        obs, _ = env.prepare(seeds, ini)
        stack = deque([embed(obs["visual"])], maxlen=int(cfg.window_size))
        action_list, steps = [], 0
        hit = np.zeros(n_envs, dtype=bool)
        last = None

        while steps < horizon:
            obs_seq = torch.cat(list(stack), dim=1)
            gseq = None if goal_emb is None else goal_emb.expand(-1, int(cfg.window_size), -1, -1)
            with torch.no_grad():
                action, _, _ = policy(obs_seq, gseq, None)
            if use_diffusion:
                chunk = [action[:, t].cpu().numpy() for t in range(action.shape[1])]
            else:
                if int(cfg.action_window_size) > 1:
                    action_list.append(action[:, -1].cpu().numpy())
                    if len(action_list) > int(cfg.action_window_size):
                        action_list = action_list[1:]
                    cur = np.array(action_list).mean(axis=0)[:, 0]
                    action_list = [np.concatenate(
                        (c[:, 1:], np.zeros((c.shape[0], 1, c.shape[-1]))), axis=1)
                        for c in action_list]
                else:
                    cur = action[:, -1, 0, :].cpu().numpy()
                chunk = [cur]
            for act in chunk:
                # the policy emits normalized actions, the env wants raw ones
                obs, _, _, info = env.step(np.asarray(act) * act_std + act_mean)
                last = info
                stack.append(embed(obs["visual"]))
                steps += 1
                cur_state = np.stack([info[i]["state"] for i in range(n_envs)])
                hit |= np.asarray(env.eval_state(gol, cur_state)["success"]).astype(bool)
                if steps >= horizon:
                    break

        cur_state = np.stack([last[i]["state"] for i in range(n_envs)])
        end = np.asarray(env.eval_state(gol, cur_state)["success"]).astype(bool)
        ever[lo:hi] = hit[:real]
        final[lo:hi] = end[:real]
    return ever, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="a dagger/<run> directory")
    ap.add_argument("--rounds", default=None, help="comma list, default every round with data")
    ap.add_argument("--horizons", default="25,300",
                    help="env steps allowed. 25 matches the planner's budget "
                         "(goal_H * frameskip); 300 matches the goal25 eval passes.")
    ap.add_argument("--out", default=None, help="json output path (default <run_dir>/relabel_eval.json)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    # resolved BEFORE the config, so load_run_cfg can read the snapshot belonging to the round
    # actually being replayed rather than an arbitrary one (B8).
    rounds = ([int(x) for x in args.rounds.split(",")] if args.rounds
              else sorted(int(p.parent.name.split("_")[1])
                          for p in run_dir.glob("rounds/round_*/relabel_states.pth")))
    if not rounds:
        raise SystemExit(f"no rounds/round_*/relabel_states.pth under {run_dir}")
    cfg = load_run_cfg(run_dir, rounds[0])
    horizons = [int(h) for h in args.horizons.split(",")]

    import hydra
    import train_policy as tp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OmegaConf.set_struct(cfg, False)
    # The dagger config's wandb snapshot was taken before dagger_loop assigns env.goal_dim, so
    # policy.goal_dim resolved to 0 in it. The checkpoint was built by the train_policy subprocess
    # with goal_dim = encoder.output_dim, so set both or load_state_dict sees the wrong width.
    cfg.env.goal_dim = int(cfg.encoder.output_dim) if bool(cfg.get("goal_conditional", False)) else 0
    OmegaConf.update(cfg, "policy.goal_dim", int(cfg.env.goal_dim), force_add=True)
    print(f"[cfg] goal_conditional={bool(cfg.get('goal_conditional', False))} "
          f"policy.goal_dim={cfg.policy.goal_dim} obs_dim={cfg.policy.get('obs_dim')}")

    # Action stats come from the demo dataset's valid split, the same source train_policy uses
    # to denormalize at eval. The policy's actions are meaningless in the env without them.
    _, traj_dset = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist,
                                    num_pred=cfg.num_pred, frameskip=cfg.frameskip)
    _vb = getattr(traj_dset["valid"], "dataset", traj_dset["valid"])
    act_mean, act_std = _vb.action_mean.numpy(), _vb.action_std.numpy()
    print(f"[stats] action_mean={act_mean.tolist()} action_std={act_std.tolist()}")

    encoder = hydra.utils.instantiate(cfg.encoder).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    env = tp.make_eval_env(cfg, int(cfg.n_envs))
    use_diffusion = "diffusion" in cfg.policy["_target_"]

    results = []
    try:
        for rnd in rounds:
            rd = run_dir / f"rounds/round_{rnd:02d}"
            # score the checkpoint dagger_loop actually carried forward, not the last epoch
            _cd = rd / "train" / "checkpoints"
            ckpt = _cd / "model_best.pt"
            if not ckpt.is_file():
                ckpt = _cd / "model_final.pt"
            states = rd / "relabel_states.pth"
            if not ckpt.is_file() or not states.is_file():
                print(f"[round {rnd}] skipped (checkpoint or relabel states missing)")
                continue

            blob = torch.load(states, map_location="cpu", weights_only=False)
            init_states = np.asarray(blob["init_states"])
            goal_states = np.asarray(blob["goal_states"])
            env_infos = blob.get("env_infos") or []

            policy = hydra.utils.instantiate(cfg.policy).to(device)
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd = (payload["model"] if isinstance(payload, dict) and "model" in payload
                  else payload.state_dict() if hasattr(payload, "state_dict") else payload)
            policy.load_state_dict(tp._fix_vqvae_keys(sd))
            policy.eval()

            row = {"round": rnd, "n_states": int(len(init_states)),
                   "planner_success_rate": planner_rate(run_dir, rnd)}
            for h in horizons:
                ever, final = rollout(cfg, policy, encoder, env, device, init_states, goal_states,
                                      env_infos, h, use_diffusion, act_mean, act_std,
                                      seed0=1000 * (rnd + 1))
                row[f"policy_max_success_h{h}"] = float(ever.mean())
                row[f"policy_final_success_h{h}"] = float(final.mean())
                print(f"[round {rnd}] horizon={h:<4} policy max={ever.mean():.3f} "
                      f"final={final.mean():.3f}")
            pr = row["planner_success_rate"]
            print(f"[round {rnd}] planner={'n/a' if pr is None else f'{pr:.3f}'} "
                  f"over {row['n_states']} states")
            results.append(row)
            del policy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        env.close()

    out = Path(args.out) if args.out else run_dir / "relabel_eval.json"
    out.write_text(json.dumps(results, indent=2))

    hs = horizons[0]
    print(f"\n{'round':>6} {'states':>7} {'planner':>8} {'policy_max':>11} {'policy_final':>13}"
          f"   (horizon {hs}, matched to the planner)")
    for r in results:
        pr = r["planner_success_rate"]
        print(f"{r['round']:>6} {r['n_states']:>7} {'n/a' if pr is None else f'{pr:>8.3f}'}"
              f" {r[f'policy_max_success_h{hs}']:>11.3f} {r[f'policy_final_success_h{hs}']:>13.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
