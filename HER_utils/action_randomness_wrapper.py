from gym.core import Wrapper
from gym import spaces
from math import floor
import numpy as np
from numpy.linalg import norm

def extend_box(box, low, high):
	new_low =  np.append(box.low, low)
	new_high =  np.append(box.high, high)
	return spaces.Box(new_low, new_high, dtype='float32')

class ActionRandomnessWrapper(Wrapper):

	def __init__(self, env, rand):
		self.env = env
		self.rand = rand

		self._action_space = None
		self._observation_space = None
		self._reward_range = None
		self._metadata = None
		self.size = env.action_space.sample().shape

		self.action_space = env.action_space
		self._max_episode_steps = env._max_episode_steps

	def step(self, action):
		return self.env.step(action + np.random.normal(scale = self.rand, size=self.size))


class RepeatedActionWrapper(Wrapper):
	def __init__(self, env, max_steps):
		self.env = env
		self.max_steps = max_steps

		self._action_space = None
		self._observation_space = None
		self._reward_range = None
		self._metadata = None
		self.size = env.action_space.sample().shape

		# low =  np.append(env.action_space.low, [-1])
		# high =  np.append(env.action_space.high, [1])
		# self.action_space = spaces.Box(low, high, dtype='float32')
		self.action_space = extend_box(env.action_space, [-1], [1])
		self._max_episode_steps = env._max_episode_steps

	def step(self, action):
		num_steps =  floor( (1 + action[-1])*self.max_steps) + 1
		for _ in range(num_steps):
			result = self.env.step(action[:-1])
		info = result[-1]
		info["steps"] = num_steps
		result = result[:-1] + (info,)
		return result


