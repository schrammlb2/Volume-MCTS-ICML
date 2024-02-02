
import gymnasium as gym
from gymnasium.core import ObservationWrapper
from gymnasium.spaces import Box
import numpy as np

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
			observation['desired_goal'] - observation['achieved_goal']
		])
