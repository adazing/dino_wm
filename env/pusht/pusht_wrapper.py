import os
import numpy as np
import gym
from env.pusht.pusht_env import PushTEnv
from utils import aggregate_dct
from env.env_shift import corrupt_visual

class PushTWrapper(PushTEnv):
    def __init__(
            self,
            with_velocity=True,
            with_target=True,
            obs_corruption=None,
            obs_corruption_level=None,
            **kwargs,
        ):
        # **kwargs forwards PushTEnv's own appearance arguments -- shape, color, agent_color,
        # goal_color -- which this wrapper previously swallowed, so `env.kwargs.shape=I` was
        # silently ignored and every env rendered a T.
        #
        # None-valued entries are DROPPED rather than forwarded. conf/env/pusht.yaml lists these
        # keys as null to document them, and null has to mean "keep the default": forwarding
        # shape=None would set self.shape = None and crash add_shape, and color=None would reach
        # pygame.Color(None). Dropping them makes null mean here what it already means for wall,
        # where parse_rgb(None, default) resolves to the default.
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        super().__init__(
            with_velocity=with_velocity,
            with_target=with_target,
            **kwargs,
        )
        self.action_dim = self.action_space.shape[0]
        # Observation corruption. pusht renders straight at render_size with no resize after,
        # so this is already the frame the encoder sees. None = unchanged.
        self.obs_corruption = None if obs_corruption in (None, "none") else str(obs_corruption)
        self.obs_corruption_level = obs_corruption_level
        self._corrupt_seed = 0
        self._corrupt_tick = 0

    def _corrupt(self, visual):
        """Apply the configured observation corruption to one (H, W, C) frame."""
        if self.obs_corruption is None:
            return visual
        self._corrupt_tick += 1
        return corrupt_visual(visual, self.obs_corruption, self.obs_corruption_level,
                              seed=self._corrupt_seed + self._corrupt_tick)
    
    def sample_random_init_goal_states(self, seed):
        """
        Return two random states: one as the initial state and one as the goal state.
        """
        rs = np.random.RandomState(seed)
        
        def generate_state():
            if self.with_velocity:
                return np.array(
                    [
                        rs.randint(50, 450),
                        rs.randint(50, 450),
                        rs.randint(100, 400),
                        rs.randint(100, 400),
                        rs.randn() * 2 * np.pi - np.pi,
                        0,
                        0,  # agent velocities default 0
                    ]
                )
            else:
                return np.array(
                    [
                        rs.randint(50, 450),
                        rs.randint(50, 450),
                        rs.randint(100, 400),
                        rs.randint(100, 400),
                        rs.randn() * 2 * np.pi - np.pi,
                    ]
                )
        
        init_state = generate_state()
        goal_state = generate_state()
        
        return init_state, goal_state
    
    def update_env(self, env_info):
        # `shape` is what pusht_dset.py:171 supplies per trajectory. The appearance keys are
        # optional and only present when eval_env_override sets them; they take effect at the
        # next reset(), where the block, pusher and anchor are rebuilt.
        if 'shape' in env_info:
            self.shape = env_info['shape']
        for key, attr in (('color', 'color'), ('agent_color', 'agent_color'),
                          ('goal_color', '_goal_color_cfg')):
            if env_info.get(key) is not None:
                setattr(self, attr, env_info[key])
    
    def eval_state(self, goal_state, cur_state):
        """
        Return True if the goal is reached
        [agent_x, agent_y, T_x, T_y, angle, agent_vx, agent_vy]
        """
        # if position difference is < 20, and angle difference < np.pi/9, then success
        pos_diff = np.linalg.norm(goal_state[:4] - cur_state[:4])
        angle_diff = np.abs(goal_state[4] - cur_state[4])
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        success = pos_diff < 20 and angle_diff < np.pi / 9
        state_dist = np.linalg.norm(goal_state - cur_state)
        # Graded overlap against THIS episode's goal. Binary success only says reached/not,
        # coverage says how close, and it is scored on the goal actually being chased rather
        # than the fixed pose the green marker sits at. Block pose is state[2:5].
        try:
            coverage = self.coverage_between(goal_state[2:5], cur_state[2:5])
        except Exception:
            coverage = float('nan')   # before the first reset there is no block to measure
        return {
            'success': success,
            'state_dist': state_dist,
            'coverage': coverage,
        }

    def prepare(self, seed, init_state):
        """
        Reset with controlled init_state
        obs: (H W C)
        state: (state_dim)
        """
        self.seed(seed)
        # Tie the corruption RNG to the episode seed and restart its counter, so a rerun of
        # the same episode draws the same noise and the metric is reproducible.
        self._corrupt_seed = int(np.asarray(seed).reshape(-1)[0]) if np.size(seed) else 0
        self._corrupt_tick = 0
        self.reset_to_state = init_state
        obs, state = self.reset()
        if isinstance(obs, dict) and "visual" in obs:
            obs["visual"] = self._corrupt(obs["visual"])
        return obs, state

    def step_multiple(self, actions):
        """
        infos: dict, each key has shape (T, ...)
        """
        obses = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            o, r, d, info = self.step(action)
            if isinstance(o, dict) and "visual" in o:
                o["visual"] = self._corrupt(o["visual"])
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        """
        only returns np arrays of observations and states
        seed: int
        init_state: (state_dim, )
        actions: (T, action_dim)
        obses: dict (T, H, W, C)
        states: (T, D)
        """
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        states = np.stack(states)
        return obses, states
