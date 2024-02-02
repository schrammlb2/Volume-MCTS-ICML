# -*- coding: utf-8 -*-
"""
Custom game generation function
@author: thomas
"""
# import gym
import gymnasium
import global_config
from global_config import gym

import numpy as np
from .wrappers import (
    NormalizeWrapper,
    ReparametrizeWrapper,
    PILCOWrapper,
    ScaleRewardWrapper,
    ClipRewardWrapper,
    ScaledObservationWrapper,
    GoalWrapper,
    PositiveRewardWrapper,
)
from .SaveStateWrapper import wrap_with_save_state_wrapper

# Register deterministic FrozenLakes
from gym.envs.registration import register


register(
    id="FrozenLakeNotSlippery-v0",
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": False},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)
register(
    id="FrozenLakeNotSlippery-v1",
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={"map_name": "8x8", "is_slippery": False},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)

run_maze = True
# run_maze = False
n_divs = 7

def get_base_env(env):
    """ removes all wrappers """
    while hasattr(env, "env"):
        env = env.env
    return env


def is_atari_game(env):
    """ Verify whether game uses the Arcade Learning Environment """
    env = get_base_env(env)
    return hasattr(env, "ale")


def make_game(game, run_maze=run_maze):
    from global_config import gym
    """ Modifications to Env """
    # game = "PointMaze_Open-v3"
    # game = "FetchReach-v3"
    # game = "FetchPickAndPlace-v2"
    # game = global_config.environment_name
    # game = "dynamics_maze"
    # game = "geometric-local_maze-7"
    # game = "dubins-local_maze-4"
    # import ipdb
    # ipdb.set_trace()
    if "local_maze" in game:
        maze = True
        split = game.split("-")
        n_divs = int(split[-1])
        if len(split) == 2:
            dynamics = "geometric"
        else:
            dynamics = game.split("-")[0]
        # print(f"Making {n_divs} X {n_divs} Maze")
        from rl.envs.maze_env import maze_env
        env = maze_env(dynamics, n_divs=n_divs)
        env.reset()
        modify = ""
        env = prepare_control_env(env, game, modify)
        env.reset()
        return env
    elif "Quadrotor" in game or "Quadcopter" in game:
        print(f"Making {game} environment")
        import gym
        # from rl.envs.mobile_mujoco_environments.factory import MushrEnvironmentFactory
        # env_factory = MushrEnvironmentFactory(
        #     max_speed=0.5,
        #     max_steering_angle=0.5,
        #     max_steps=100,
        #     prop_steps=10,
        #     goal_limits=[0, 5],
        #     with_obstacles=True,
        # )
        # env_factory.register_environments_with_position_goals()
        # env = gym.make(game+"Env-v0")
        # env = gym.make("QuadrotorObsEnv-v0")
        from rl.envs.mobile_mujoco_environments.envs.quadrotor_env import QuadrotorReachEnv
        env = QuadrotorReachEnv(max_steps=30, noisy=False, use_obs=True,
                 use_orientation=False, noise_scale=0.01,
                 return_full_trajectory=False, max_speed=1.0, prop_steps=100)

        from gymnasium.wrappers import StepAPICompatibility
        from rl.wrappers import ResetCompatibilityWrapper

        env = GoalWrapper(env)
        env = PositiveRewardWrapper(env)
        env.reset()
        env = StepAPICompatibility(env, output_truncation_bool=False)
        env = wrap_with_save_state_wrapper(env)
        return env


    if "Fetch" in game:
        print(f"Making {game} environment")
        import gymnasium
        import gymnasium_robotics
        from gymnasium.wrappers.time_limit import TimeLimit as gymnasiumTimeLimit
        from rl.envs.pomp.example_problems.gymnasium_robotics_local.gymnasium_robotics.envs.fetch.reach import MujocoFetchReachEnv as FetchReachEnv
        from rl.envs.pomp.example_problems.gymnasium_robotics_local.gymnasium_robotics.envs.fetch.push import MujocoFetchPushEnv as FetchPushEnv
        from rl.envs.pomp.example_problems.gymnasium_robotics_local.gymnasium_robotics.envs.fetch.slide import MujocoFetchSlideEnv as FetchSlideEnv
        from rl.envs.pomp.example_problems.gymnasium_robotics_local.gymnasium_robotics.envs.fetch.pick_and_place import MujocoFetchPickAndPlaceEnv as FetchPickAndPlaceEnv
        
        if "Reach" in game:            
            env = FetchReachEnv()
        elif "Push" in game:
            env = FetchPushEnv()
        elif "Slide" in game:
            env = FetchSlideEnv()
        elif "PickAndPlace" in game:
            env = FetchPickAndPlaceEnv()
        else: 
            import ipdb
            ipdb.set_trace()

        # env = gymnasium.vector.make('FetchReach-v3', num_envs=3)
        max_steps = 50
        env = gymnasiumTimeLimit(env, max_episode_steps=max_steps)
        from gymnasium.wrappers import StepAPICompatibility
        from rl.wrappers import ResetCompatibilityWrapper
        # env = ResetCompatibilityWrapper(env)

        env = GoalWrapper(env)
        env = PositiveRewardWrapper(env)
        env.reset()
        env = StepAPICompatibility(env, output_truncation_bool=False)
        env = wrap_with_save_state_wrapper(env)
        env._max_episode_steps = 50
        return env

    # game = "PointMaze_Large-v3"
    # game = "AntMaze_Open-v3"
    name, version = game.rsplit("-", 1)
    if len(version) > 2:
        modify = version[2:]
        game = name + "-" + version[:2]
    else:
        modify = ""

    print("Making game {}".format(game))
    # import ipdb
    # ipdb.set_trace()

    # print(gym)
    # env = gym.make(game)
    if "Maze" in game: 
        # env = gym.make(game, continuing_task=False, max_episode_steps=global_config.max_episode_steps)
        env = gym.make(game, continuing_task=False, max_episode_steps=global_config.max_episode_length)
    else:
        env = gym.make(game)
    env.reset()
    import gymnasium
    if (type(env.observation_space) in [gym.spaces.dict.Dict, gymnasium.spaces.dict.Dict]):
        env = GoalWrapper(env)
        if not "Maze" in game:
            env = PositiveRewardWrapper(env)
        env.reset()


    if global_config.using_gymnasium:
        from gymnasium.wrappers import StepAPICompatibility
        env = StepAPICompatibility(env, output_truncation_bool=False)
    # else:
    #     env = gym.make(game)
    # remove timelimit wrapper
    
    
    if type(env) == gym.wrappers.time_limit.TimeLimit:
        env = env.env

    if is_atari_game(env):
        env = prepare_atari_env(env)
    else:
        env = prepare_control_env(env, game, modify)

    env = wrap_with_save_state_wrapper(env)
    env.reset()

    return env

def make_planner_game(problem):
    env = ProblemGymEnv(problem)
    env = prepare_control_env(env, game, "")


def prepare_control_env(env, game, modify):
    if "n" in modify and type(env.observation_space) == gym.spaces.Box:
        print("Normalizing input space")
        env = NormalizeWrapper(env)
    if "r" in modify:
        print("Reparametrizing the reward function")
        env = ReparametrizeWrapper(env)
    if "p" in modify:
        env = PILCOWrapper(env)
    if "s" in modify:
        print("Rescaled the reward function")
        # env = ScaleRewardWrapper(env)

    if "CartPole" in game:
        env.observation_space = gym.spaces.Box(
            np.array([-4.8, -10, -4.8, -10]), np.array([4.8, 10, 4.8, 10])
        )
    return env


def prepare_atari_env(Env, frame_skip=3, repeat_action_prob=0.0, reward_clip=True):
    """ Initialize an Atari environment """
    env = get_base_env(Env)
    env.ale.setFloat("repeat_action_probability".encode("utf-8"), repeat_action_prob)
    env.frame_skip = frame_skip
    Env = ScaledObservationWrapper(Env)
    if reward_clip:
        Env = ClipRewardWrapper(Env)
    return Env
