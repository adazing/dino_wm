from gym.envs.registration import register

# point_maze pulls in d4rl + mujoco_py transitively (env/pointmaze/maze_model.py).
try:
    from .pointmaze import U_MAZE
    _HAS_POINTMAZE = True
except ImportError as _e:   # pragma: no cover
    U_MAZE, _HAS_POINTMAZE = None, False
    print(f"[env] point_maze unavailable ({_e}); its registration is skipped.")

register(
    id="pusht",
    entry_point="env.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
if _HAS_POINTMAZE:
    register(
        id='point_maze',
        entry_point='env.pointmaze:PointMazeWrapper',
        max_episode_steps=300,
        kwargs={
            'maze_spec':U_MAZE,
            'reward_type':'sparse',
            'reset_target': False,
            'ref_min_score': 23.85,
            'ref_max_score': 161.86,
            'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5'
        }
    )
    # maze_spec above is only the DEFAULT layout. Select a different one per-run with
    # env.kwargs.maze_name in conf/env/point_maze.yaml ('u' | 'medium' | 'large' | ...)
    # PointMazeWrapper resolves it from MAZE_SPECS and overrides maze_spec.
register(
    id="wall",
    entry_point="env.wall.wall_env_wrapper:WallEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

register(
    id="deformable_env",
    entry_point="env.deformable_env.FlexEnvWrapper:FlexEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

# OGBench Puzzle ("Lights Out"). Board size, press-distance k, resolution and goal arm pose all
# come from conf/env/puzzle.yaml -> env.kwargs, so one registration covers every variant.
register(
    id="puzzle",
    entry_point="env.puzzle.puzzle_wrapper:PuzzleWrapper",
    max_episode_steps=None,
    order_enforce=False,
    reward_threshold=1.0,
)
