import argparse

import numpy as np
from rl.envs.mobile_mujoco_environments.factory import MushrEnvironmentFactory

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.her import ObsDictWrapper

from stable_baselines3 import HER, SAC
import gym
import os

np.set_printoptions(suppress=True)


def main(args):
    env_factory = MushrEnvironmentFactory(
        max_speed=0.5,
        max_steering_angle=0.5,
        max_steps=100,
        prop_steps=10,
        goal_limits=[0, 5],
        with_obstacles=True,
    )
    num_steps = int(3e6)

    env_name = args.env

    checkpoint_path = "./trained_models/"+env_name

    if args.has_orientation_goals:
        env_factory.register_environments_with_position_and_orientation_goals()
    else:
        env_factory.register_environments_with_position_goals()

    env = gym.make(env_name+"Env-v0")

    model = HER('MlpPolicy', env, SAC, n_sampled_goal=4,
                goal_selection_strategy='future',
                verbose=1, buffer_size=int(1e6),
                learning_rate=1e-3,
                gamma=0.95, batch_size=256,max_episode_length=env.max_steps)

    eval_env = DummyVecEnv([lambda: gym.make(env_name+'Env-v0')])
    eval_env = ObsDictWrapper(eval_env)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=checkpoint_path)
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=100,
        eval_freq=int(1e4),
        best_model_save_path=checkpoint_path+"best/",
        log_path=checkpoint_path+"logs/",
        deterministic=True
    )

    callback = CallbackList([checkpoint_callback,eval_callback])

    model.learn(num_steps, callback=callback)
    model.save(checkpoint_path)


if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--env', type=str, default='X2Obs')
    argParser.add_argument('--has_orientation_goals', default=False, action='store_true')

    main(argParser.parse_args())
