import os
import numpy as np
import gym
from env.pointmaze.maze_model import (
    MazeEnv, U_MAZE, U_MAZE_EVAL, SMALL_MAZE,
    MEDIUM_MAZE, MEDIUM_MAZE_EVAL, LARGE_MAZE, LARGE_MAZE_EVAL, OPEN,
)
from utils import aggregate_dct

# Selectable layouts, so ONE env (`point_maze`) covers every maze instead of needing a separate
# gym registration and env config per layout. Pick with env.kwargs.maze_name in the env config:
#     kwargs: {maze_name: medium}
# The layouts are d4rl's, already defined in maze_model.py -- nothing external to fetch.
# NOTE the coordinate frame scales with the layout, because a cell index IS a world coordinate:
# u spans ~[0.9, 3.1], medium ~[0.9, 6.1], large ~[0.9, 10.1]. sample_random_init_goal_states
# reads the open cells out of maze_arr, so it follows automatically.
MAZE_SPECS = {
    "u": U_MAZE,
    "u_eval": U_MAZE_EVAL,
    "small": SMALL_MAZE,
    "medium": MEDIUM_MAZE,
    "medium_eval": MEDIUM_MAZE_EVAL,
    "large": LARGE_MAZE,
    "large_eval": LARGE_MAZE_EVAL,
    "open": OPEN,
}

STATE_RANGES = np.array([
    [0.39318362, 3.2198412],  # Range for first dimension
    [0.62660956, 3.2187355],  # Range for second dimension
    [-5.2262554, 5.2262554],  # Range for third dimension
    [-5.2262554, 5.2262554],  # Range for fourth dimension
    # [0.90001136, 3.0999563],  # Range for first dimension of target
    # [0.9000267, 3.0999668]    # Range for second dimension of target
])

class PointMazeWrapper(MazeEnv):
    def __init__(self, maze_name=None, **kwargs):
        """maze_name selects a layout from MAZE_SPECS ('u' | 'medium' | 'large' | ...) and wins
        over any maze_spec in kwargs. Omit it to keep whatever the gym registration supplies."""
        if maze_name is not None:
            key = str(maze_name).lower()
            if key not in MAZE_SPECS:
                raise ValueError(f"maze_name={maze_name!r} unknown; "
                                 f"choose from {sorted(MAZE_SPECS)}")
            kwargs["maze_spec"] = MAZE_SPECS[key]
        super().__init__(**kwargs)
        self.maze_name = maze_name
        self.action_dim = self.action_space.shape[0]
    
    def sample_random_init_goal_states(self, seed):
        """
        Return two random states: one as the initial state and one as the goal state.

        Positions come from the maze's OWN open cells and use the same convention
        MazeEnv.reset_model does -- pick a cell from empty_and_goal_locations, then jitter by
        +-0.1, because the cell index IS the world coordinate. That makes this work for ANY
        maze_spec (U, medium, large, open).

        The previous version hardcoded the U-maze corridor as literal x/y ranges
        (0.5<=x<=1.1 or 2.5<=x<=3.1, etc). On a bigger layout those ranges cover only part of
        the maze and land inside walls, so init/goal pairs were sampled in solid geometry with
        no error raised -- just episodes that could never succeed. Velocities keep the original
        STATE_RANGES sampling, so U-maze behaviour changes only in that the position test is now
        derived from the layout instead of transcribed by hand.
        """
        rs = np.random.RandomState(seed)
        # empty_and_goal_locations = open cells + the goal cell (both traversable); fall back to
        # reset_locations (open cells only) on older MazeEnv versions.
        cells = getattr(self, "empty_and_goal_locations", None) or self.reset_locations
        if not cells:
            raise RuntimeError("maze has no open cells to sample from; check maze_spec.")

        def generate_state():
            row, col = cells[rs.randint(len(cells))]   # maze_arr index == world (x, y)
            return np.array([
                row + rs.uniform(-0.1, 0.1),
                col + rs.uniform(-0.1, 0.1),
                rs.uniform(low=STATE_RANGES[2][0], high=STATE_RANGES[2][1]),
                rs.uniform(low=STATE_RANGES[3][0], high=STATE_RANGES[3][1]),
            ])

        init_state = generate_state()
        goal_state = generate_state()
        return init_state, goal_state
    
    def update_env(self, env_info):
        pass 
    
    def eval_state(self, goal_state, cur_state):
        # state is [x, y, vx, vy]; success is POSITION ONLY, velocity is ignored.
        success = np.linalg.norm(goal_state[:2] - cur_state[:2]) < 0.5
        # state_dist spans all four dims, so it mixes position (range ~0.5-3.1) with velocity
        # (range +-5.226) -- and the goal's velocity is drawn at random by
        # sample_random_init_goal_states, so most of this number is noise unrelated to the task.
        # Kept unchanged for comparability with existing runs.
        state_dist = np.linalg.norm(goal_state - cur_state)
        # pos_dist is the quantity success actually thresholds. Log this one, not state_dist:
        # it falls monotonically as the point mass approaches the goal and hits <0.5 on success.
        pos_dist = np.linalg.norm(goal_state[:2] - cur_state[:2])
        return {
            'success': success,
            'state_dist': state_dist,
            'pos_dist': pos_dist,
        }

    def prepare(self, seed, init_state):
        """
        Reset with controlled init_state
        obs: (H W C)
        state: (state_dim)
        """
        self.prepare_for_render()
        self.seed(seed)
        self.set_init_state(init_state)
        obs, state = self.reset()
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
