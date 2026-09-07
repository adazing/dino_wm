"""Roll a trained BC policy out in the real env and record the states it visits."""
import os
# MuJoCo defaults to the GLFW backend, which needs an X display and fails on a headless box.
os.environ.setdefault("MUJOCO_GL", "egl")
import json
import random
import hydra
import torch
import numpy as np
from pathlib import Path
from collections import deque
from einops import rearrange
from omegaconf import OmegaConf

from utils import seed as set_seed
import train_policy as tp


def _sample_states(n_visited, n_take, mode, rng, skew):
    """Pick which visited-state indices to hand the planner.

    'late'  -> spread over the LAST `skew` fraction of the episode. Default, because
    'uniform'  -> evenly spaced across the whole episode.
    'random'  -> uniform at random without replacement (each state at most once).
    'random_with_replacement' -> independent draws, duplicates allowed, lets n_take
    """
    lo = 1   # never the episode's own start
    if n_visited <= lo:
        return []
    if mode == "late":
        lo = max(1, int(n_visited * (1.0 - float(skew))))
    pool = list(range(lo, n_visited))
    if not pool:
        return []
    if mode in ("late", "uniform"):
        # Evenly spaced BIN CENTRES, not endpoints, with n_take=1, linspace(0, len-1, 1) would
        # return index 0, i.e.
        n = min(n_take, len(pool))
        idx = (np.arange(n) + 0.5) / n * len(pool)
        return sorted({pool[min(int(i), len(pool) - 1)] for i in idx})
    if mode == "random":
        return sorted(rng.sample(pool, min(n_take, len(pool))))
    if mode == "random_with_replacement":
        return sorted(rng.choice(pool) for _ in range(n_take))
    raise ValueError(f"state_sample_mode must be 'late' | 'uniform' | 'random' | "
                     f"'random_with_replacement', got {mode!r}")


@torch.no_grad()
def rollout_policy(cfg, encoder, policy, device, env, n_episodes, horizon, base_seed,
                   traj_dset=None, use_diffusion=False):
    """Run the policy for n_episodes and return one record per episode:
        {seed, init_state, goal_state, env_info, states (T+1, S), success, min_dist}

    'random_state' -> init/goal sampled from the ENV (wall: opposite sides of the wall).
    'valid'|'train'-> init/goal read from a DEMO trajectory in that split: random start
    """
    n_envs = int(cfg.n_envs)
    goal_source = str(cfg.get("rollout_goal_source", "random_state"))
    rs_mode = goal_source == "random_state"
    if goal_source not in ("random_state", "valid", "train"):
        raise ValueError(f"rollout_goal_source must be 'random_state' | 'valid' | 'train', "
                         f"got {goal_source!r}")
    from datasets.img_transforms import default_transform
    tf = default_transform(cfg.img_size)

    # Action denormalization stats come from the main dataset's valid split, the policy emits
    # actions in that normalization, so the env must be stepped with those stats, identical to
    # eval_on_env, a mismatch here silently scales every action.
    if traj_dset is None:
        _, traj_dset = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist,
                                        num_pred=cfg.num_pred, frameskip=cfg.frameskip)
    valid_base = getattr(traj_dset["valid"], "dataset", traj_dset["valid"])
    act_mean, act_std = valid_base.action_mean.numpy(), valid_base.action_std.numpy()

    # wall, pin the maze before sampling init/goal, so the sampled states are valid for it.
    wall_fixed_info = None
    if cfg.env.name == "wall":
        _ref = traj_dset.get("valid") or traj_dset.get("train")
        if _ref is not None:
            wall_fixed_info = _ref.get_frames(0, [0])[3]

    # demo-goal mode, the split to draw (init, goal) pairs from, and the frames-ahead offset.
    dset = goal_off = valid_ids = None
    if not rs_mode:
        dset = traj_dset[goal_source]
        goal_off = int(cfg.get("rollout_goal_offset") or 25)
        # a trajectory must be long enough to hold a goal_off-ahead goal (eval_on_env's filter) if
        # none are, fall back to all and let the goal index clamp to the end
        valid_ids = [j for j in range(len(dset))
                     if dset.get_seq_length(j) >= goal_off + 1] or list(range(len(dset)))
        print(f"[rollout] tasks from the {goal_source} split: {len(valid_ids)}/{len(dset)} "
              f"trajectories long enough for goal_offset={goal_off}")

    def embed(visual_np):
        v = torch.as_tensor(np.ascontiguousarray(visual_np)).float()
        if v.max() > 1.5:   # env renders may be [0,255] or [0,1]
            v = v / 255.0
        v = rearrange(v, "b h w c -> b c h w")
        return encoder(tf(v).to(device)).reshape(
            v.shape[0], 1, encoder.n_patches, encoder.emb_dim)

    records = []
    n_batches = max(1, int(np.ceil(n_episodes / n_envs)))
    for b in range(n_batches):
        real = min(n_envs, n_episodes - b * n_envs)
        seeds = [int(base_seed) + b * n_envs + i for i in range(n_envs)]
        if rs_mode:
            # env-sampled task.
            if wall_fixed_info is not None:
                env.update_env([wall_fixed_info] * n_envs)
            init_states, goal_states = env.sample_random_init_goal_states(seeds)
            init_states, goal_states = np.asarray(init_states), np.asarray(goal_states)
            ep_env_infos = [wall_fixed_info] * n_envs
            # render + embed the goal, prepare leaves the env AT the goal, so re-prepare to init
            # afterwards, same order plan.py and eval_on_env use
            goal_obs, _ = env.prepare(seeds, goal_states)
            goal_emb = (embed(np.asarray(goal_obs["visual"]))
                        if int(cfg.env.goal_dim) > 0 else None)
        else:
            # demo task, random trajectory + random start offset, goal goal_off frames later.
            idxs, init_off = [], []
            for i in range(n_envs):
                rng_i = random.Random(int(base_seed) + b * n_envs + i)
                j = rng_i.choice(valid_ids)
                idxs.append(j)
                init_off.append(rng_i.randint(0, max(0, dset.get_seq_length(j) - goal_off - 1)))
            frames = [dset.get_frames(idxs[k], [init_off[k]]) for k in range(n_envs)]
            init_states = np.stack([np.asarray(f[2][0]) for f in frames])
            ep_env_infos = [f[3] for f in frames]   # per-trajectory layout (wall), pusht, dict
            goal_idx = [min(init_off[k] + goal_off, dset.get_seq_length(idxs[k]) - 1)
                        for k in range(n_envs)]
            gframes = [dset.get_frames(idxs[k], [goal_idx[k]]) for k in range(n_envs)]
            goal_states = np.stack([np.asarray(g[2][0]) for g in gframes])
            # dataset visuals are ALREADY transformed to [0,1] at img_size, so they go straight to
            # the encoder, running `tf` again (as embed() does for env renders) would be a second
            # resize.
            goal_emb = None
            if int(cfg.env.goal_dim) > 0:
                goal_imgs = torch.stack([g[0]["visual"][0] for g in gframes]).to(device)
                goal_emb = encoder(goal_imgs).reshape(
                    n_envs, 1, encoder.n_patches, encoder.emb_dim)
            # apply the per-trajectory layout before reset, so the goal and the rollout share it
            if all(e is not None for e in ep_env_infos):
                env.update_env(ep_env_infos)

        obs, state_0 = env.prepare(seeds, init_states)
        obs_stack = deque([embed(obs["visual"])], maxlen=cfg.window_size)
        # Seed the trajectory with the ENV's own state at reset, not the sampled init_states, for
        # some envs sample_random_init_goal_states returns a shorter vector than info["state"],
        # wall, 2D position vs the env's full state, and np.stack below would then fail on
        # inhomogeneous shapes.
        visited = [np.asarray(state_0, dtype=float)]   # (n_envs, S) per step
        best = np.full(n_envs, np.inf)
        ever = np.zeros(n_envs, dtype=bool)
        action_list, steps = [], 0
        done = np.array([False])
        # `and not np.all(done)` matters, without it, an env that reports done keeps getting
        # stepped until the horizon is burned.
        while steps < horizon and not np.all(done):
            obs_seq = torch.cat(list(obs_stack), dim=1)
            gseq = None if goal_emb is None else goal_emb.expand(-1, cfg.window_size, -1, -1)
            action, _, _ = policy(obs_seq, gseq, None)
            if use_diffusion:
                chunk = [action[:, t].cpu().numpy() for t in range(action.shape[1])]
            else:
                # receding-horizon ensembling, identical to eval_on_env
                if cfg.action_window_size > 1:
                    action_list.append(action[:, -1].cpu().numpy())
                    if len(action_list) > cfg.action_window_size:
                        action_list = action_list[1:]
                    curr = np.array(action_list).mean(axis=0)[:, 0]
                    action_list = [np.concatenate(
                        (c[:, 1:], np.zeros((c.shape[0], 1, c.shape[-1]))), axis=1)
                        for c in action_list]
                else:
                    curr = action[:, -1, 0, :].cpu().numpy()
                chunk = [curr]
            for a in chunk:
                obs, _, done, info = env.step(a * act_std + act_mean)
                done = np.asarray(done)
                obs_stack.append(embed(obs["visual"]))
                steps += 1
                if info is not None and "state" in info[0]:
                    cur = np.stack([info[i]["state"] for i in range(n_envs)])
                    visited.append(np.asarray(cur, dtype=float))
                    r = env.eval_state(goal_states, cur)
                    ever |= np.asarray(r["success"]).astype(bool)
                    dk = "state_dist" if "state_dist" in r else None
                    if dk is not None:
                        best = np.minimum(best, np.asarray(r[dk], dtype=float))
                if steps >= horizon or np.all(done):
                    break

        traj = np.stack(visited, axis=1)   # (n_envs, T+1, S)
        for i in range(real):
            records.append({
                "seed": seeds[i],
                "init_state": init_states[i],
                "goal_state": goal_states[i],
                "env_info": ep_env_infos[i],   # per-episode layout, so relabelling replays it
                "states": traj[i],
                "success": bool(ever[i]),
                "min_dist": float(best[i]),
            })
        print(f"[rollout] batch {b}: {int(ever[:real].sum())}/{real} reached the goal")
    return records


def select_relabel_states(records, cfg, rng, max_states=None):
    """Turn per-episode visited states into flat (init, goal, env_info) triples to relabel."""
    pool = records
    if bool(cfg.get("relabel_failures_only", True)):
        failed = [r for r in pool if not r["success"]]
        if failed:
            pool = failed
        else:
            print("[relabel] every rollout succeeded -> relabelling from all of them instead")
    n_pool_all = len(pool)
    n_ep = cfg.get("episodes_to_relabel", None)
    if n_ep is not None and int(n_ep) < len(pool):
        # hardest first, the furthest the policy ever got from the goal
        if str(cfg.get("episode_pick", "worst")) == "worst":
            pool = sorted(pool, key=lambda r: -r["min_dist"])[: int(n_ep)]
        else:
            pool = rng.sample(pool, int(n_ep))

    mode = str(cfg.get("state_sample_mode", "late"))
    per_ep = int(cfg.get("states_per_episode", 1))
    skew = float(cfg.get("state_sample_late_frac", 0.5))
    init_states, goal_states, env_infos, provenance = [], [], [], []
    for r in pool:
        for k in _sample_states(len(r["states"]), per_ep, mode, rng, skew):
            init_states.append(r["states"][k])
            goal_states.append(r["goal_state"])
            env_infos.append(r["env_info"])
            provenance.append({"seed": int(r["seed"]), "step": int(k),
                               "episode_success": bool(r["success"])})
    if max_states is not None:
        want = int(max_states)
        if len(init_states) > want:
            # subsample rather than truncate, truncating would drop whole late episodes and
            # over-represent the first few
            pick = sorted(rng.sample(range(len(init_states)), want))
            init_states = [init_states[i] for i in pick]
            goal_states = [goal_states[i] for i in pick]
            env_infos = [env_infos[i] for i in pick]
            provenance = [provenance[i] for i in pick]
        elif len(init_states) < want:
            # Report it, beta asked for `want` policy-scouted trajectories and the rollout could
            # not supply that many, so this round's actual planner/policy split is not what beta
            # specifies.
            print(f"[relabel] WARNING: only {len(init_states)} states available but beta asked "
                  f"for {want}. Sources: {n_pool_all} candidate episodes"
                  + (f" (capped to {int(n_ep)} by episodes_to_relabel)" if n_ep is not None else "")
                  + f", states_per_episode={per_ep}"
                  + (", relabel_failures_only=true" if bool(cfg.get("relabel_failures_only", True))
                     else "")
                  + ". This round's beta split will not be honoured.")
    return init_states, goal_states, env_infos, provenance


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg.seed))
    cfg.env.goal_dim = int(cfg.encoder.output_dim) if bool(cfg.get("goal_conditional", False)) else 0
    ckpt = cfg.get("checkpoint", None)
    if not ckpt:
        raise ValueError("policy_rollout needs +checkpoint=<a train_policy model_*.pt>")

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
    try:
        records = rollout_policy(
            cfg, encoder, policy, device, env,
            n_episodes=int(cfg.get("rollout_episodes", 20)),
            horizon=int(cfg.get("rollout_horizon", cfg.eval_horizon)),
            base_seed=int(cfg.get("rollout_base_seed", cfg.seed)),
            use_diffusion="diffusion" in cfg.policy["_target_"])
    finally:
        env.close()

    out = cfg.get("rollout_out", "./visited.pth")
    torch.save(records, out)
    n_succ = sum(1 for r in records if r["success"])
    print(f"[rollout] {len(records)} episodes, {n_succ} reached the goal -> {out}")


@hydra.main(config_path="conf", config_name="train_policy", version_base="1.2")
def _main(cfg: OmegaConf):
    main(cfg)


if __name__ == "__main__":
    _main()
