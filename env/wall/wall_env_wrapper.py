from .envs.wall import DotWall
from .envs.wall import WallDatasetConfig
import numpy as np
import torch
import random
from torchvision import transforms


from utils import aggregate_dct
from .data.wall_utils import generate_wall_layouts
from ..env_shift import corrupt_visual

ENV_ACTION_DIM = 2
STATE_RANGES = np.array([
    [16.6840, 46.9885],
    [4.0083,25.2532]
])

DEFAULT_CFG = WallDatasetConfig(
    action_angle_noise=0.2,
    action_step_mean=1.0,
    action_step_std=0.4,
    action_lower_bd=0.2,
    action_upper_bd=1.8,
    batch_size=64,
    device='cuda',
    dot_std=1.7,
    border_wall_loc=5,
    fix_wall_batch_k=None,
    fix_wall=True,
    fix_door_location=30,
    fix_wall_location=32,
    exclude_wall_train='',
    exclude_door_train='',
    only_wall_val='',
    only_door_val='',
    wall_padding=20,
    door_padding=10,
    wall_width=6,
    door_space=4,
    num_train_layouts=-1,
    img_size=65,
    max_step=1,
    n_steps=17,
    n_steps_reduce_factor=1,
    size=20000,
    val_size=10000,
    train=True,
    repeat_actions=1
)

resize_transform = transforms.Resize((224, 224))
TRANSFORM = resize_transform

class WallEnvWrapper(DotWall):
    def __init__(self, rng=42, wall_config=DEFAULT_CFG, fix_wall=True, cross_wall=False, fix_wall_location=32, fix_door_location=10, device='cpu', sample_padding=None, obs_corruption=None, obs_corruption_level=None, **kwargs):
        super().__init__(rng, wall_config, fix_wall, cross_wall, fix_wall_location=fix_wall_location, fix_door_location=fix_door_location, device=device, sample_padding=sample_padding, **kwargs)
        self.action_dim = ENV_ACTION_DIM
        self.transform = TRANSFORM
        # Observation corruption. Applied AFTER the 224 resize, because corrupting the 65x65
        # render and then upsampling would smear salt-and-pepper into soft blobs instead of
        # isolated pixels, which is not the corruption AdaJEPA defines. None = unchanged.
        self.obs_corruption = None if obs_corruption in (None, "none") else str(obs_corruption)
        self.obs_corruption_level = obs_corruption_level
        self._corrupt_seed = 0      # set by seed(), so salt-and-pepper is reproducible
        self._corrupt_tick = 0      # advanced per frame, so successive frames differ

    def eval_state(self, goal_state, cur_state):
        success = np.linalg.norm(goal_state[:2] - cur_state[:2]) < 4.5
        state_dist = np.linalg.norm(goal_state - cur_state)
        return {
            'success': success,
            'state_dist': state_dist,
        }

    def sample_random_init_goal_states(self, seed):
        """
        Return a random state
        """
        return self.generate_random_state(seed)

    def update_env(self, env_info):   # change door and wall locations
        self.wall_config.fix_door_location = env_info["fix_door_location"].item()
        self.wall_config.fix_wall_location = env_info["fix_wall_location"].item()
        layouts, other_layouts = generate_wall_layouts(self.wall_config)
        self.layouts = layouts
        self.wall_x, self.hole_y = self._generate_wall()
        # left_wall_x/right_wall_x are DERIVED from wall_x but were only ever assigned in
        # DotWall.__init__ (env/wall/envs/wall.py:50-51), so without these two lines they keep
        # the faces of whatever wall existed at construction. generate_random_state bounds its
        # start/goal samples with them (same file, lines 142-143), so after a layout change the
        # samples are drawn against the OLD wall: with the wall moved to 40 but built at 32,
        # 24% of goals land inside the new wall's solid band and are unreachable, and another
        # 6% land on the start's own side so the task is not a crossing -- 31% of tasks broken,
        # which would read as a generalisation collapse rather than as a sampling bug.
        # Invisible until now only because every wall dataset uses fix_wall_location=32, so the
        # stale value equalled the fresh one.
        self.left_wall_x = self.wall_x - self.wall_config.wall_width // 2
        self.right_wall_x = self.wall_x + self.wall_config.wall_width // 2

    def _corrupt(self, visual):
        """Apply the configured observation corruption to one (H, W, C) frame."""
        if self.obs_corruption is None:
            return visual
        self._corrupt_tick += 1
        return corrupt_visual(visual, self.obs_corruption, self.obs_corruption_level,
                              seed=self._corrupt_seed + self._corrupt_tick)

    def prepare(self, seed, init_state):
        """
        Reset with controlled init_state
        """
        self.seed(seed)
        # Tie the corruption RNG to the episode seed and restart its counter, so a rerun of
        # the same episode draws the same noise and the metric is reproducible.
        self._corrupt_seed = int(np.asarray(seed).reshape(-1)[0]) if np.size(seed) else 0
        self._corrupt_tick = 0
        self.set_init_state(init_state)
        obs, state = self.reset()
        obs['visual'] = self.transform(obs['visual'])
        obs['visual'] = obs['visual'].permute(1, 2, 0)
        obs['visual'] = self._corrupt(obs['visual'])
        return obs, state

    def step_multiple(self, actions):
        obses = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            o, r, d, info = self.step(action)
            o['visual'] = self.transform(o['visual'])
            o['visual'] = o['visual'].permute(1, 2, 0)
            o['visual'] = self._corrupt(o['visual'])
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
