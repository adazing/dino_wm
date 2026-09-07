"""OGBench Puzzle ("Lights Out") wrapped for dino_wm.

rather than condition on the goal.  Here init is random and the goal is drawn at a
pairs is reachable at all.  See `lights_out.py`.
where planning breaks.  k=1 is also the only regime where greedy latent descent is
discs -- a randomly-posed goal makes latent distance mostly about the arm.  'canonical'
instead of noise.  OGBench's own task mode does the opposite; don't copy it.
obs  = {'visual': (H, W, 3) uint8, 'proprio': (19,) float}
"""

import mujoco
import numpy as np

try:
    from ogbench.manipspace import lie
    from ogbench.manipspace.envs.puzzle_env import PuzzleEnv
except ImportError as e:   # pragma: no cover
    raise ImportError(
        "The puzzle env needs OGBench. Install it with `pip install ogbench`, or "
        "`pip install -e .` from a checkout of https://github.com/seohongpark/ogbench"
    ) from e

from env.puzzle.lights_out import apply_presses, min_presses
from utils import aggregate_dct

# proprio = joint_pos(6) + joint_vel(6) + effector_pos(3) + cos/sin yaw(2) + gripper(1) +
# contact(1)
PROPRIO_DIM = 19

VALID_ENV_TYPES = ("3x3", "4x4", "4x5", "4x6")


class PuzzleWrapper(PuzzleEnv):
    def __init__(
        self,
        env_type="3x3",
        k=1,
        img_size=224,
        goal_arm_mode="canonical",
        transparent_arm=True,
        gripper_init="closed",
        with_button_joints_in_proprio=False,
        full_info=False,
        **kwargs,
    ):
        """
        Args:
            env_type: board size, one of '3x3', '4x4', '4x5', '4x6'.
            k: press-distance from init to goal. An int for a fixed distance, or a
                (lo, hi) pair to sample uniformly per episode -- e.g. k=[1, 3] for a mixed
                curriculum.  k=0 gives goal == init (a sanity check: the planner should
                score 100% by doing nothing).
            img_size: render resolution.  OGBench's released visual datasets are 64x64;
                dino_wm's DINO encoder wants 224, so generate your own data at 224.
            goal_arm_mode: arm pose baked into the goal observation.
                'canonical' -- one fixed pose, the same every episode (recommended).
                'init'      -- identical to the init pose, so the goal image differs from
                               the init image ONLY in button colours.
                'random'    -- a fresh random pose (OGBench-native; adds arm noise to the
                               planning objective).
            transparent_arm: render the arm nearly transparent (OGBench's own default for
                pixel observations, and it works in your favour here).
            gripper_init: 'closed' (default) or 'open' -- the gripper pose baked into sampled
                init/goal states. The oracles keep it closed for the whole dataset, so
                'closed' is what the world model was trained on. Only use 'open' if you
                generated data with gripper_always_closed disabled.
            with_button_joints_in_proprio: unused hook kept so proprio width can be extended
                without changing the state layout.
            full_info: if True, `step` returns the env's complete ob_info dict (including the
                'privileged/target_*' keys the scripted oracles read) alongside 'state'.
                tools/gen_puzzle_dset.py needs this; the planner does NOT -- it aggregates info
                across a rollout, and the full dict contains a string field that will not
                stack.  Leave it False for planning.
        """
        if env_type not in VALID_ENV_TYPES:
            raise ValueError(f"env_type must be one of {VALID_ENV_TYPES}, got {env_type!r}")
        if goal_arm_mode not in ("canonical", "init", "random"):
            raise ValueError(f"goal_arm_mode must be canonical|init|random, got {goal_arm_mode!r}")
        if gripper_init not in ("closed", "open"):
            raise ValueError(f"gripper_init must be closed|open, got {gripper_init!r}")

        super().__init__(
            env_type=env_type,
            ob_type="pixels",
            mode="data_collection",
            terminate_at_goal=False,
            visualize_info=False,
            pixel_transparent_arm=transparent_arm,
            width=img_size,
            height=img_size,
            **kwargs,
        )

        self.action_dim = 5
        self.img_size = img_size
        self.goal_arm_mode = goal_arm_mode
        self.proprio_dim = PROPRIO_DIM
        self._with_button_joints_in_proprio = with_button_joints_in_proprio
        self._full_info = full_info
        self._seed = 0
        self._seed_pending = 0

        import gym as _gym   # old gym (0.23), matching the rest of dino_wm's env stack

        self._action_space = _gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self._observation_space = _gym.spaces.Box(
            low=0, high=255, shape=(img_size, img_size, 3), dtype=np.uint8
        )

        self.k_lo, self.k_hi = self._parse_k(k)
        if self.k_hi > self._num_buttons:
            raise ValueError(
                f"k up to {self.k_hi} exceeds the {self._num_buttons} buttons on a {env_type} board"
            )

        # Build the model once so nq/nv/site ids/_T_pa exist, then cache a rest-state qpos
        # template.
        super().reset(seed=self._seed)
        self.nq = int(self._model.nq)
        self.nv = int(self._model.nv)
        self.state_dim = self.nq + self.nv + self._num_buttons

        self._button_qpos_adrs = np.array(
            [self._model.joint(f"buttonbox_joint_{i}").qposadr[0] for i in range(self._num_buttons)]
        )
        # Snapshot the qpos template that every sampled init/goal state is built from.
        if gripper_init == "closed":
            self._data.ctrl[self._arm_actuator_ids] = self._data.qpos[self._arm_joint_ids]
            self._data.ctrl[self._gripper_actuator_ids] = 255.0
            for _ in range(20):
                mujoco.mj_step(self._model, self._data, nstep=self._n_steps)

        template = self._data.qpos.copy()
        template[self._button_qpos_adrs] = 0.0   # plungers at rest
        self._template_qpos = template

        # One fixed effector pose for goal_arm_mode='canonical', the centre of the arm sampling
        # slab, facing straight ahead.
        self._canonical_eff_pos = self._arm_sampling_bounds.mean(axis=0)
        self._canonical_eff_yaw = 0.0

    #------------------------------------------------------------------ helpers

    @staticmethod
    def _parse_k(k):
        if np.isscalar(k):
            lo = hi = int(k)
        else:
            k = list(k)
            if len(k) != 2:
                raise ValueError(f"k must be an int or a (lo, hi) pair, got {k!r}")
            lo, hi = int(k[0]), int(k[1])
        if lo < 0 or hi < lo:
            raise ValueError(f"need 0 <= k_lo <= k_hi, got ({lo}, {hi})")
        return lo, hi

    def _arm_qpos(self, eff_pos, yaw):
        """Inverse-kinematics an effector pose into the 6 arm joint angles."""
        eff_ori = lie.SO3.from_z_radians(yaw) @ self._effector_down_rotation
        t_wa = lie.SE3.from_rotation_and_translation(eff_ori, np.asarray(eff_pos)) @ self._T_pa
        return self._ik.solve(
            pos=t_wa.translation(),
            quat=t_wa.rotation().wxyz,
            curr_qpos=self._home_qpos,
        )

    def _full_qpos(self, eff_pos, yaw):
        qpos = self._template_qpos.copy()
        qpos[self._arm_joint_ids] = self._arm_qpos(eff_pos, yaw)
        return qpos

    def _make_state(self, qpos, qvel, button_states):
        return np.concatenate(
            [np.asarray(qpos, dtype=np.float64),
             np.asarray(qvel, dtype=np.float64),
             np.asarray(button_states, dtype=np.float64)]
        )

    def _split_state(self, state):
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        qpos = state[: self.nq]
        qvel = state[self.nq : self.nq + self.nv]
        buttons = np.rint(state[self.nq + self.nv :]).astype(np.int64)
        return qpos, qvel, buttons

    def get_state(self):
        return self._make_state(self._data.qpos, self._data.qvel, self._cur_button_states)

    def _proprio(self, ob_info):
        return np.concatenate(
            [
                ob_info["proprio/joint_pos"],
                ob_info["proprio/joint_vel"],
                ob_info["proprio/effector_pos"],
                np.cos(ob_info["proprio/effector_yaw"]),
                np.sin(ob_info["proprio/effector_yaw"]),
                ob_info["proprio/gripper_opening"],
                ob_info["proprio/gripper_contact"],
            ]
        ).astype(np.float64)

    def _obs(self, visual, ob_info):
        # COPY, do not asarray, mujoco's Renderer may hand back a reused internal buffer, and
        # asarray would return that same object.
        return {
            "visual": np.array(visual, dtype=np.uint8, copy=True),
            "proprio": self._proprio(ob_info),
        }

    @property
    def board_shape(self):
        return self._num_rows, self._num_cols

    @property
    def num_buttons(self):
        return self._num_buttons

    #------------------------------------------------- old-gym / dino_wm surface

    def seed(self, seed=None):
        """Queue a seed for the NEXT reset."""
        if seed is not None:
            self._seed = int(seed)
            self._seed_pending = int(seed)
        return [self._seed]

    @property
    def action_space(self):
        # Old-gym spaces, not gymnasium's, dino_wm's stack is gym 0.23.
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def reset(self, **kwargs):
        """Old-gym style: returns (obs, state), not gymnasium's (obs, info)."""
        seed, self._seed_pending = self._seed_pending, None
        visual, ob_info = super().reset(seed=seed, **kwargs)
        return self._obs(visual, ob_info), self.get_state()

    def ob_info(self):
        """The env's full observation-info dict (what the scripted oracles read)."""
        return self.compute_ob_info()

    def step(self, action):
        """Old-gym 4-tuple. `info` is deliberately slim -- the planner aggregates it."""
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        visual, _reward, _terminated, _truncated, ob_info = super().step(action)
        obs = self._obs(visual, ob_info)
        state = self.get_state()
        if self._full_info:
            info = dict(ob_info)
            info["state"] = state
            return obs, 0.0, False, info
        return obs, 0.0, False, {"state": state}

    #--------------------------------------------------------- planning hooks

    def sample_random_init_goal_states(self, seed):
        """Random init board + arm; goal = that board after `k` distinct presses."""
        rs = np.random.RandomState(seed)
        num_rows, num_cols = self.board_shape

        k = int(rs.randint(self.k_lo, self.k_hi + 1))
        init_buttons = rs.randint(0, 2, size=self._num_buttons).astype(np.int64)
        goal_buttons = init_buttons.copy()
        if k > 0:
            # A nonempty press set CAN be a no-op on 4x4 (nullity 4).
            for _ in range(64):
                presses = rs.choice(self._num_buttons, size=k, replace=False)
                candidate = apply_presses(init_buttons, presses, num_rows, num_cols)
                if not np.array_equal(candidate, init_buttons):
                    goal_buttons = candidate
                    break
            else:
                goal_buttons = candidate   # give up. Vanishingly unlikely

        init_eff = rs.uniform(*self._arm_sampling_bounds)
        init_yaw = rs.uniform(-np.pi, np.pi)
        init_qpos = self._full_qpos(init_eff, init_yaw)

        if self.goal_arm_mode == "init":
            goal_qpos = init_qpos.copy()
        elif self.goal_arm_mode == "random":
            goal_qpos = self._full_qpos(
                rs.uniform(*self._arm_sampling_bounds), rs.uniform(-np.pi, np.pi)
            )
        else:   # canonical
            goal_qpos = self._full_qpos(self._canonical_eff_pos, self._canonical_eff_yaw)

        zeros = np.zeros(self.nv)
        return (
            self._make_state(init_qpos, zeros, init_buttons),
            self._make_state(goal_qpos, zeros, goal_buttons),
        )

    def update_env(self, env_info):
        """No-op: board geometry is fixed by env_type at construction."""
        return

    def eval_state(self, goal_state, cur_state):
        """Success is EXACT board match -- the state is discrete, so no distance threshold.

        solver.  Report it alongside success: it is the honest measure of how far a rollout
        """
        _, _, goal_buttons = self._split_state(goal_state)
        _, _, cur_buttons = self._split_state(cur_state)
        num_rows, num_cols = self.board_shape

        n_wrong = int((goal_buttons != cur_buttons).sum())
        remaining = min_presses(cur_buttons, goal_buttons, num_rows, num_cols)
        return {
            "success": bool(n_wrong == 0),
            "state_dist": float(n_wrong),
            "n_wrong_buttons": float(n_wrong),
            "press_dist": float(remaining) if remaining is not None else -1.0,
        }

    def prepare(self, seed, init_state):
        """Reset to an exact state. obs: (H, W, C); state: (state_dim,)."""
        self.seed(seed)
        self._seed_pending = None   # nothing to consume. Do not leak into a later reset()
        mujoco.mj_resetData(self._model, self._data)

        qpos, qvel, buttons = self._split_state(init_state)
        # PuzzleEnv.set_state takes the board state too, it is not in qpos.
        self.set_state(qpos, qvel, buttons)

        # Mirror the tail of PuzzleEnv.initialize_episode so the press-edge detector starts with
        # prev == cur and does not fire a spurious press on the first step.
        self.pre_step()
        self.post_step()

        ob_info = self.compute_ob_info()
        return self._obs(self.render(), ob_info), self.get_state()

    def step_multiple(self, actions):
        obses, rewards, dones, infos = [], [], [], []
        for action in actions:
            o, r, d, info = self.step(action)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        return (
            aggregate_dct(obses),
            np.stack(rewards),
            np.stack(dones),
            aggregate_dct(infos),
        )

    def rollout(self, seed, init_state, actions):
        """obses: dict of (T+1, ...); states: (T+1, state_dim)."""
        obs, state = self.prepare(seed, init_state)
        obses, _rewards, _dones, infos = self.step_multiple(actions)
        for key in obses:
            obses[key] = np.vstack([np.expand_dims(obs[key], 0), obses[key]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        return obses, states
