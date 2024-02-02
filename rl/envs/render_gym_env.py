from pomp.example_problems.robotics.fetch.reach import FetchReachEnv
from pomp.example_problems.robotics.fetch.push import FetchPushEnv
from pomp.example_problems.robotics.fetch.slide import FetchSlideEnv
from pomp.example_problems.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv

from pomp.example_problems.robotics.hand.reach import HandReachEnv

from gym.wrappers.time_limit import TimeLimit

import time

def render(env, steps):
    env.reset()
    env.render()
    for _ in range(steps):
        time.sleep(0.01)
        env.step(env.action_space.sample())
        env.render()

if __name__ == '__main__':
    env_name = "FetchReach"

    if env_name == "FetchReach":
        max_episode_steps=50
        env = TimeLimit(FetchReachEnv(), max_episode_steps=max_episode_steps)
    elif env_name == "FetchPush":
        max_episode_steps=50
        env = TimeLimit(FetchPushEnv(), max_episode_steps=max_episode_steps)
    elif env_name == "FetchSlide":
        max_episode_steps=50
        env = TimeLimit(FetchSlideEnv(), max_episode_steps=max_episode_steps)
    elif env_name == "FetchPickAndPlace":
        max_episode_steps=50
        env = TimeLimit(FetchPickAndPlaceEnv(), max_episode_steps=max_episode_steps)
    elif env_name == "HandReach":
        max_episode_steps=10
        env = TimeLimit(HandReachEnv(), max_episode_steps=max_episode_steps)

    for _ in range(10):
        render(env, max_episode_steps)

