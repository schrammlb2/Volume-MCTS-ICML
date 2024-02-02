from gymnasium_robotics_local.gymnasium_robotics.envs.fetch.reach import MujocoFetchReachEnv as FetchReachEnv
from gymnasium_robotics_local.gymnasium_robotics.envs.fetch.push import MujocoFetchPushEnv as FetchPushEnv
from gymnasium_robotics_local.gymnasium_robotics.envs.fetch.slide import MujocoFetchSlideEnv as FetchSlideEnv
from gymnasium_robotics_local.gymnasium_robotics.envs.fetch.pick_and_place import MujocoFetchPickAndPlaceEnv as FetchPickAndPlaceEnv

# from pomp.example_problems.robotics.hand.reach import HandReachEnv

# from gymnasium_robotics.envs.fetch.reach import RandomFetchReachEnv as FetchReachEnv
# from gymnasium_robotics.envs.fetch.reach import RotationMujocoFetchReachEnv as FetchReachEnv

from gymnasium.wrappers.time_limit import TimeLimit

import time

def render(env, steps):
    env.reset()
    # import ipdb
    # ipdb.set_trace()
    # env.render()
    for _ in range(steps):
        # time.sleep(0.01)
        env.step(env.action_space.sample())
        # env.render()

if __name__ == '__main__':
    env_name = "FetchReach"

    if env_name == "FetchReach":
        max_episode_steps=50
        env = TimeLimit(FetchReachEnv(render_mode="human"), max_episode_steps=max_episode_steps)
    elif env_name == "FetchPush":
        max_episode_steps=50
        env = TimeLimit(FetchPushEnv(render_mode="human"), max_episode_steps=max_episode_steps)
    elif env_name == "FetchSlide":
        max_episode_steps=50
        env = TimeLimit(FetchSlideEnv(render_mode="human"), max_episode_steps=max_episode_steps)
    elif env_name == "FetchPickAndPlace":
        max_episode_steps=50
        env = TimeLimit(FetchPickAndPlaceEnv(render_mode="human"), max_episode_steps=max_episode_steps)
    # elif env_name == "HandReach":
    #     max_episode_steps=10
    #     env = TimeLimit(HandReachEnv(), max_episode_steps=max_episode_steps)

    for _ in range(10):
        render(env, max_episode_steps)

