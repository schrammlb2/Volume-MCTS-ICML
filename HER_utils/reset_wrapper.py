
from gymnasium.core import Wrapper

class ResetWrapper(Wrapper):
	def reset(self, **kwargs):
		return self.env.reset(**kwargs)[0]

class StepNumWrapper(Wrapper):
	@property
	def _max_episode_steps(self):
		return self.env.spec.max_episode_steps
