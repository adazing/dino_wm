"""Generate OGBench Puzzle demonstration data in dino_wm's dataset format.

expert  ButtonPlanOracle, non-Markovian. Interpolates 5 keyframes per press
(approach -> press -> release -> retreat) and tracks the resulting spline, with
noisy  ButtonMarkovOracle, closed-loop 4-phase state machine, plus per-episode i.i.d.
Example:
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
import pickle

# MuJoCo needs a GL backend chosen before the env is imported.
os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np   # noqa: E402
import torch   # noqa: E402

from env.puzzle.puzzle_wrapper import VALID_ENV_TYPES, PuzzleWrapper   # noqa: E402


def build_oracle(env, dataset_type, args):
    from ogbench.manipspace.oracles.markov.button_markov import ButtonMarkovOracle
    from ogbench.manipspace.oracles.plan.button_plan import ButtonPlanOracle

    if dataset_type == "expert":
        return ButtonPlanOracle(
            env=env,
            noise=args.noise,
            noise_smoothing=args.noise_smoothing,
            gripper_always_closed=True,
        )
    return ButtonMarkovOracle(
        env=env, min_norm=args.min_norm, gripper_always_closed=True
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="Output directory (created).")
    p.add_argument("--env_type", default="3x3", choices=VALID_ENV_TYPES)
    p.add_argument("--dataset_type", default="expert", choices=["expert", "noisy"])
    p.add_argument("--num_episodes", type=int, default=500)
    p.add_argument("--episode_steps", type=int, default=200,
                   help="Fixed env steps per episode. One button press is ~34 steps, so 200 "
                        "steps is roughly 6 presses of coverage per episode.")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise", type=float, default=0.1,
                   help="expert: plan noise scale (0 = noiseless expert). "
                        "noisy: upper bound of the per-episode action-noise level.")
    p.add_argument("--noise_smoothing", type=float, default=0.5,
                   help="expert only: Gaussian smoothing sigma for the correlated plan noise.")
    p.add_argument("--min_norm", type=float, default=0.4, help="noisy only: MarkovOracle min action norm.")
    p.add_argument("--p_random_action", type=float, default=None,
                   help="Fraction of fully random actions. Default: 0.0 for expert, 0.2 for noisy.")
    p.add_argument("--transparent_arm", action="store_true", default=True,
                   help="Render the arm nearly transparent (OGBench's default for pixels).")
    p.add_argument("--opaque_arm", dest="transparent_arm", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    if args.p_random_action is None:
        args.p_random_action = 0.0 if args.dataset_type == "expert" else 0.2

    out = Path(args.out)
    obs_dir = out / "obses"
    obs_dir.mkdir(parents=True, exist_ok=True)

    est_gb = args.num_episodes * args.episode_steps * args.img_size**2 * 3 / 1e9
    print(f"Writing to {out} (~{est_gb:.1f} GB of images)", flush=True)

    rng = np.random.RandomState(args.seed)
    # Both OGBench oracles draw their retreat poses and plan jitter from the GLOBAL np.random so
    # seeding it is what actually makes a generated dataset reproducible.
    np.random.seed(args.seed)

    env = PuzzleWrapper(
        env_type=args.env_type,
        img_size=args.img_size,
        transparent_arm=args.transparent_arm,
        full_info=True,   # the oracles read the privileged/target_* keys
    )
    env.seed(args.seed)
    oracle = build_oracle(env, args.dataset_type, args)

    all_states, all_proprios, all_actions, seq_lengths = [], [], [], []
    n_presses_total = 0

    for ep in range(args.num_episodes):
        obs, state = env.reset()
        info = env.ob_info()
        oracle.reset(obs, info)

        # 'noisy' draws a fresh noise level per episode, so the dataset spans clean-ish to very
        # noisy behaviour rather than sitting at one fixed level.
        xi = rng.uniform(0, args.noise) if args.dataset_type == "noisy" else 0.0

        ep_visual, ep_proprio, ep_state, ep_action = [], [], [], []
        prev_buttons = state[env.nq + env.nv :].copy()

        for _ in range(args.episode_steps):
            if rng.rand() < args.p_random_action:
                action = env.action_space.sample()
            else:
                action = np.asarray(oracle.select_action(obs, info), dtype=np.float64)
                if args.dataset_type == "noisy":
                    action = action + rng.normal(0, [xi, xi, xi, xi * 3, xi * 10], action.shape)
            action = np.clip(action, -1.0, 1.0)

            # Record the observation before the action, so (obs[t], action[t]) is a transition and
            # obs[t+1] is its result.
            ep_visual.append(obs["visual"])
            ep_proprio.append(obs["proprio"])
            ep_state.append(state)
            ep_action.append(action)

            obs, _reward, _done, info = env.step(action)
            state = info["state"]

            cur_buttons = state[env.nq + env.nv :]
            if not np.array_equal(cur_buttons, prev_buttons):
                n_presses_total += 1
                prev_buttons = cur_buttons.copy()

            if oracle.done:
                # Hand the oracle a fresh random target button and keep going.
                env.set_new_target(return_info=False)
                info = env.ob_info()
                oracle.reset(obs, info)

        # .npy, not torch.save, the loader memory-maps these so it can read a single window
        # instead of a whole 30 MB episode, and np.save round-trips independently of torch's
        # weights_only default, which flipped to True in torch 2.6 and rejects raw arrays.
        np.save(obs_dir / f"episode_{ep:05d}.npy", np.stack(ep_visual).astype(np.uint8))
        all_states.append(np.stack(ep_state).astype(np.float32))
        all_proprios.append(np.stack(ep_proprio).astype(np.float32))
        all_actions.append(np.stack(ep_action).astype(np.float32))
        seq_lengths.append(len(ep_state))

        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"episode {ep + 1}/{args.num_episodes} "
                f"({n_presses_total} presses so far, "
                f"{n_presses_total / (ep + 1):.1f}/episode)",
                flush=True,
            )

    max_len = max(seq_lengths)
    n_eps = len(seq_lengths)

    def pad_stack(arrays, dim):
        out_t = torch.zeros((n_eps, max_len, dim), dtype=torch.float32)
        for i, a in enumerate(arrays):
            out_t[i, : len(a)] = torch.from_numpy(a)
        return out_t

    torch.save(pad_stack(all_states, env.state_dim), out / "states.pth")
    torch.save(pad_stack(all_proprios, env.proprio_dim), out / "proprios.pth")
    torch.save(pad_stack(all_actions, env.action_dim), out / "actions.pth")

    with open(out / "seq_lengths.pkl", "wb") as f:
        pickle.dump(seq_lengths, f)

    num_rows, num_cols = env.board_shape
    meta = {
        "env_type": args.env_type,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "num_buttons": env.num_buttons,
        "nq": env.nq,
        "nv": env.nv,
        # state = qpos | qvel | button_states, this is where the board lives.
        "button_slice": [env.nq + env.nv, env.state_dim],
        "state_dim": env.state_dim,
        "proprio_dim": env.proprio_dim,
        "action_dim": env.action_dim,
        "img_size": args.img_size,
        "dataset_type": args.dataset_type,
        "episode_steps": args.episode_steps,
        "noise": args.noise,
        "p_random_action": args.p_random_action,
        "seed": args.seed,
    }
    with open(out / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(
        f"\nDone. {n_eps} episodes x {max_len} steps, "
        f"{n_presses_total} button presses total ({n_presses_total / n_eps:.1f} per episode).\n"
        f"state_dim={env.state_dim} proprio_dim={env.proprio_dim} action_dim={env.action_dim}"
    )


if __name__ == "__main__":
    main()
