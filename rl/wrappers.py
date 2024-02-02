#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gym Wrappers
@author: thomas
"""
import gym
import numpy as np
from scipy.stats import multivariate_normal
import sklearn.preprocessing


class ObservationRewardWrapper(gym.Wrapper):
    """ My own base class - allows for both observation and reward modification """

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), self.reward(reward), done, info

    def reset(self):
        observation = self.env.reset()
        return self.observation(observation)

    def observation(self, observation):
        return observation

    def reward(self, reward):
        return reward


def get_name(env):
    while True:
        if hasattr(env, "_spec"):
            name = env._spec.id
            break
        elif hasattr(env, "spec"):
            name = env.spec.id
            break
        else:
            env = env.env
    return name


class NormalizeWrapper(ObservationRewardWrapper):
    """ normalizes the input data range """

    def __init__(self, env):
        ObservationRewardWrapper.__init__(self, env)
        # self.name = get_name(env)
        observation_examples = np.array(
            [env.observation_space.sample() for x in range(10000)]
        )
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

    def observation(self, observation):
        return self.scaler.transform([observation])[0]


class ScaleRewardWrapper(ObservationRewardWrapper):
    def __init__(self, env):
        ObservationRewardWrapper.__init__(self, env)
        self.name = get_name(env)

    def reward(self, reward):
        """ Rescale reward """
        if "Pendulum" in self.name:
            new_reward = np.float32(reward / 1000.0)
        # elif 'MountainCarContinuous' in self.name:
        #    return np.float32(reward/500.0)
        elif "Lunarlander" in self.name:
            new_reward = np.float32(reward / 250.0)
        elif "CartPole" in self.name:
            new_reward = reward / 250.0
        elif "MountainCar" in self.name:
            new_reward = reward / 250.0
        elif "Acrobot" in self.name:
            new_reward = reward / 250.0
        else:
            new_reward = reward

        return np.squeeze(new_reward)


class ReparametrizeWrapper(ObservationRewardWrapper):
    def __init__(self, env):
        ObservationRewardWrapper.__init__(self, env)
        self.name = get_name(env)

    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        return (
            self.observation(observation),
            self.reward(reward, terminal),
            terminal,
            info,
        )

    def reward(self, r, terminal):
        if "CartPole" in self.name:
            if terminal:
                r = -1
            else:
                r = 0.005
        elif "MountainCar" in self.name:
            if terminal:
                r = 1
            else:
                r = -0.005
        elif "Acrobot" in self.name:
            if terminal:
                r = 1
            else:
                r = -0.005
        return r


class DiscretizeWrapper(gym.ActionWrapper):
    def __init__(self, env):
        ObservationRewardWrapper.__init__(self, env)
        # self.name = get_name(env)
        # assert type(env.action_space) is gym.spaces.Box
        self.action_dim = env.action_space.shape[0]
        self.k = 3
        self.action_space = gym.spaces.Discrete(self.k**self.action_dim)
        self.low  = env.action_space.low
        self.high = env.action_space.high

    def action(self, action):
        action_div = action
        vals = []
        elements = []
        for i in range(self.action_dim):
            val = (1+action_div%self.k)/(self.k+1)
            vals.append(val)
            elements.append(val*self.high[i] + (1-val)*self.low[i])
            action_div = action_div//self.k

        # import ipdb
        # ipdb.set_trace()
        return np.array(elements)
        
    def reverse_action(self, action):
        k = 0
        sum_value = 0
        for i in range(self.action_dim):
            ind = int((self.k+2)*(action[i] - self.low[i])/(self.high[i] - self.low[i]))
            if ind == 0: 
                ind = 1
            elif ind == self.k+2:
                ind = self.k+1
            ind -= 1
            sum_value = sum_value*self.k + ind
        return np.array(sum_value)






class PILCOWrapper(ObservationRewardWrapper):
    def __init__(self, env):
        ObservationRewardWrapper.__init__(self, env)
        self.name = get_name(env)

    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        return self.observation(observation), self.reward(observation), terminal, info

    def reward(self, s):
        if "CartPole" in self.name:
            target = np.array([0.0, 0.0, 0.0, 0.0])
        elif "Acrobot" in self.name:
            target = np.array([1.0])
            s = -np.cos(s[0]) - np.cos(s[1] + s[0])
        elif "MountainCar" in self.name:
            target = np.array([0.5])
            s = s[0]
        elif "Pendulum" in self.name:
            target = np.array([0.0, 0.0])
        else:
            raise ValueError("no PILCO reward mofication for this game")
        return 1 - multivariate_normal.pdf(s, mean=target)


class ClipRewardWrapper(ObservationRewardWrapper):
    def __init__(self, env):
        ObservationRewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class ScaledObservationWrapper(ObservationRewardWrapper):
    def __init__(self, env):
        ObservationRewardWrapper.__init__(self, env)

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0

import gymnasium

# from global_config import gym
# from global_config import spaces
from gymnasium.core import ObservationWrapper, RewardWrapper
from gymnasium.spaces import Box
import numpy as np

import gymnasium_robotics

PointEnv = (
    gymnasium_robotics.envs.maze.point.PointEnv,
    gymnasium_robotics.envs.maze.point_maze.PointMazeEnv,
)

AntMazeEnv = gymnasium_robotics.envs.maze.ant_maze.AntMazeEnv

class GoalWrapper(ObservationWrapper):    
    def __init__(self, env):
        super().__init__(env)
        new_low = np.concatenate([
            env.observation_space['observation'].low, 
            env.observation_space['desired_goal'].low
        ])
        new_high = np.concatenate([
            env.observation_space['observation'].high, 
            env.observation_space['desired_goal'].high
        ])
        self.observation_space = Box(new_low, new_high)

    def observation(self, observation):
        return np.concatenate([
            observation['observation'], 
            # observation['desired_goal'] - observation['achieved_goal']
            observation['desired_goal']
        ])

    def get_goal_obs(self):
        if callable(self.env.unwrapped):
            return self.env.unwrapped()._get_obs()
        else:
            if isinstance(self.env.unwrapped, PointEnv) or isinstance(self.env.unwrapped, AntMazeEnv):
                return self.env.unwrapped._get_obs(self.env.unwrapped.point_env.state_vector())
            else:
                return self.env.unwrapped._get_obs()
        # return self.env.unwrapped()._get_obs()


class MazeGoalWrapper(ObservationWrapper):    
    def __init__(self, env):
        super().__init__(env)
        new_low = np.concatenate([
            env.observation_space['observation'].low, 
            env.observation_space['desired_goal'].low
        ])
        new_high = np.concatenate([
            env.observation_space['observation'].high, 
            env.observation_space['desired_goal'].high
        ])
        self.observation_space = Box(new_low, new_high)

    def observation(self, observation):
        return np.concatenate([
            observation['observation'], 
            # observation['desired_goal'] - observation['achieved_goal']
            observation['desired_goal']
        ])

    def get_goal_obs(self):
        if callable(self.env.unwrapped):
            return self.env.unwrapped()._get_obs()
        else:
            return self.env.unwrapped._get_obs()
        # return self.env.unwrapped()._get_obs()

    # def get_processed_observation(self):
    #     return self.observation(self.env.unwrapped._get_obs())

class PositiveRewardWrapper(RewardWrapper):
    def reward(self, r):
        return r + 1

class ResetCompatibilityWrapper(gymnasium.core.Wrapper):
    def reset(self, **kwargs):
        reset_val = self.env.reset(**kwargs)
        if type(reset_val) != tuple: 
            import ipdb
            ipdb.set_trace()
        return reset_val[0]


    def _get_obs(self):
        if callable(self.env.unwrapped): env = self.env.unwrapped()
        else: env = self.env.unwrapped
        return env._get_obs()