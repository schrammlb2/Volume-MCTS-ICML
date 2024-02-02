import gymnasium# as gym
import gymnasium_robotics
from gymnasium.core import ObservationWrapper, RewardWrapper
from gymnasium.spaces import Box
from copy import deepcopy
import numpy as np
import ipdb

import mujoco

from rl.envs.maze_env import ProblemGymGoalEnv
from rl.envs.mobile_mujoco_environments.envs.quadrotor_env import QuadrotorReachEnv
from rl.envs.pomp.example_problems.gymnasium_robotics_local.gymnasium_robotics.envs.fetch import MujocoFetchEnv

gymnasium.make("FetchPush-v2") 
gymnasium.make("AdroitHandDoor-v1") 
gymnasium.make("PointMaze_Open-v3") 
gymnasium.make("AntMaze_Open-v3") 
	#No idea why, but module doesn't load in envs.fetch and envs.adroit_hand until a mujoco environment is made


FetchEnv = (
	gymnasium_robotics.envs.fetch.MujocoFetchEnv,
	MujocoFetchEnv,
)
AdroitEnv = (
	gymnasium_robotics.envs.adroit_hand.AdroitHandDoorEnv,
	gymnasium_robotics.envs.adroit_hand.AdroitHandHammerEnv,
	gymnasium_robotics.envs.adroit_hand.AdroitHandPenEnv,
	gymnasium_robotics.envs.adroit_hand.AdroitHandRelocateEnv,
)
PointEnv = (
	gymnasium_robotics.envs.maze.point.PointEnv,
	gymnasium_robotics.envs.maze.point_maze.PointMazeEnv,
)

AntMazeEnv = gymnasium_robotics.envs.maze.ant_maze.AntMazeEnv

def wrap_with_save_state_wrapper(env):
	if isinstance(env.unwrapped, FetchEnv):
		return_env= FetchSaveStateWrapper(env)
	elif isinstance(env.unwrapped, AdroitEnv):
		return_env= AdroitSaveStateWrapper(env)
	elif isinstance(env.unwrapped, PointEnv):
		return_env= PointSaveStateWrapper(env)
	elif isinstance(env.unwrapped, AntMazeEnv):
		return_env= AntSaveStateWrapper(env)
	elif isinstance(env.unwrapped, QuadrotorReachEnv):
		return_env = QuadcopterSaveStateWrapper(env)
	elif type(env.unwrapped()) == ProblemGymGoalEnv:
		return_env= ProblemSaveStateWrapper(env)		
	else: 
		print(f"No SaveStateWrapper for environment type {type(env.unwrapped)}")
		ipdb.set_trace()

	try:
		assert return_env is not None
	except:
		ipdb.set_trace()

	return return_env



class SaveStateWrapper(gymnasium.core.Wrapper):
	def get_save_state(self):
		raise NotImplementedError

	def restore_save_state(self, saved_data):
		raise NotImplementedError


class ProblemSaveStateWrapper(SaveStateWrapper):
	def get_save_state(self):
		return {
			"state": self.env.state,
			"goal" : self.env.goal,
		}

	def restore_save_state(self, saved_data):
		self.env.state = saved_data['state']
		self.env.goal  = saved_data['goal']



class FetchSaveStateWrapper(SaveStateWrapper):
	def __init__(self, env):
		super().__init__(env)
		ag_low  = np.array([-3.2 for _ in range(env.unwrapped.observation_space['achieved_goal'].shape[0])])
		ag_high = np.array([ 3.2 for _ in range(env.unwrapped.observation_space['achieved_goal'].shape[0])])
		env.unwrapped.observation_space['achieved_goal'] = Box(ag_low, ag_high)
		env.unwrapped.observation_space['desired_goal'] = Box(ag_low, ag_high)
		obs_low  = np.array([-3.2 for _ in range(env.unwrapped.observation_space['observation'].shape[0])])
		obs_high = np.array([ 3.2 for _ in range(env.unwrapped.observation_space['observation'].shape[0])])
		env.unwrapped.observation_space['observation'] = Box(obs_low, obs_high)

	def get_save_state(self):
		data = self.unwrapped.data
		return {
			'time': deepcopy(data.time),
			'qpos': np.copy(data.qpos[:]),
			'qvel': np.copy(data.qvel[:]),
			'goal': np.copy(self.unwrapped.goal)
		}

	def restore_save_state(self, saved_data):
		env = self.unwrapped
		data = env.data
		data.time = deepcopy(saved_data['time'])
		data.qpos[:] = np.copy(saved_data['qpos'])
		data.qvel[:] = np.copy(saved_data['qvel'])
		self.unwrapped.goal = np.copy(saved_data['goal'])
		if env.model.na != 0:
		    data.act[:] = None

		# if env.has_object:
		#     object_qpos = saved_data['object_qpos']
		#     assert object_qpos.shape == (7,)
		#     env._utils.set_joint_qpos(
		#         env.model, env.data, "object0:joint", object_qpos
		#     )

		env._mujoco.mj_forward(env.model, env.data)
		return True



	def _get_obs(self):
	    if callable(self.env.unwrapped): env = self.env.unwrapped()
	    else: env = self.env.unwrapped
	    return env._get_obs()
	    

class QuadcopterSaveStateWrapper(SaveStateWrapper):
	def __init__(self, env):
		super().__init__(env)
		goal_limit = env.goal_limit
		goal_min = [-goal_limit, -goal_limit, 0]
		goal_max = [goal_limit]*3
		# ag_low  = np.array([-3.2 for _ in range(env.unwrapped.observation_space['achieved_goal'].shape[0])])
		# ag_high = np.array([ 3.2 for _ in range(env.unwrapped.observation_space['achieved_goal'].shape[0])])
		ag_low  = np.array(goal_min)
		ag_high = np.array(goal_max)
		env.unwrapped.observation_space['achieved_goal'] = Box(ag_low, ag_high)
		env.unwrapped.observation_space['desired_goal'] = Box(ag_low, ag_high)
		obs_low  = np.array([-3.2 for _ in range(env.unwrapped.observation_space['observation'].shape[0])])
		obs_high = np.array([ 3.2 for _ in range(env.unwrapped.observation_space['observation'].shape[0])])
		env.unwrapped.observation_space['observation'] = Box(obs_low, obs_high)

	def get_save_state(self):
		data = self.unwrapped.quadrotor.data
		return {
			'time': deepcopy(data.time),
			'qpos': np.copy(data.qpos[:]),
			'qvel': np.copy(data.qvel[:]),
			'goal': np.copy(self.unwrapped.goal)
		}

	# def _get_obs(self):
	# 	obs = super()._get_obs()
	# 	obs['state'] = self.get_save_state()
	# 	return obs

	def restore_save_state(self, saved_data):
		env = self.unwrapped.quadrotor
		data = env.data
		# import ipdb
		# ipdb.set_trace()
		data.time = deepcopy(saved_data['time'])
		data.qpos[:] = np.copy(saved_data['qpos'])
		data.qvel[:] = np.copy(saved_data['qvel'])
		self.unwrapped.goal = np.copy(saved_data['goal'])
		if env.model.na != 0:
		    data.act[:] = None

		mujoco.mj_forward(env.model, env.data)
		return True

	def set_state(self, env, state_vec):
		split_vec = np.array_split(state_vec, 2)
		saved_data = {
			'time': 0.,
			'qpos': split_vec[0],
			'qvel': split_vec[1],
			'goal': env.goal
		}
		env.restore_save_state(saved_data)

	def _get_obs(self):
		return self.env.unwrapped._duplicate_get_obs()

class AdroitSaveStateWrapper(SaveStateWrapper):
	def get_save_state(self):
		state_dict = deepcopy(self.unwrapped.get_env_state())
		state_dict['goal'] = np.copy(self.unwrapped.goal)
		return  state_dict

	def restore_save_state(self, state_dict):
		"""
		Set the state which includes hand as well as objects and targets in the scene
		"""
		assert self._state_space.contains(
		    state_dict
		), f"The state dictionary {state_dict} must be a member of {self._state_space}."
		qp = state_dict["qpos"]
		qv = state_dict["qvel"]
		self.unwrapped.model.body_pos[self.unwrapped.door_body_id] = state_dict["door_body_pos"]
		self.unwrapped.set_state(qp, qv)
		self.unwrapped.goal = state_dict['goal']

import types

def point_fixed_get_obs(self, point_obs):# -> Dict[str, np.ndarray]:
	#There's a bug in the released version of the point maze environment, which this fixes
    achieved_goal = point_obs[:2]
    # observation = point_obs[2:]
    return {
        "observation": point_obs.copy(),
        "achieved_goal": achieved_goal.copy(),
        "desired_goal": self.goal.copy(),
    }

def ant_fixed_get_obs(self, point_obs):# -> Dict[str, np.ndarray]:
	#There's a bug in the released version of the point maze environment, which this fixes
    achieved_goal = point_obs[:2]
    # observation = point_obs[2:]
    return {
        "observation": point_obs.copy(),
        "achieved_goal": achieved_goal.copy(),
        "desired_goal": self.goal.copy(),
    }

class PointSaveStateWrapper(SaveStateWrapper):
	def __init__(self, env):
		super().__init__(env)
		env.unwrapped._get_obs = types.MethodType(point_fixed_get_obs, env.unwrapped)
		# if 
		# import ipdb
		# ipdb.set_trace()
		ag_low  = np.array([-6 for _ in range(env.unwrapped.observation_space['achieved_goal'].shape[0])])
		ag_high = np.array([ 6 for _ in range(env.unwrapped.observation_space['achieved_goal'].shape[0])])
		env.unwrapped.observation_space['achieved_goal'] = Box(ag_low, ag_high)
		env.unwrapped.observation_space['desired_goal'] = Box(ag_low, ag_high)
		obs_low  = np.array([-6 for _ in range(env.unwrapped.observation_space['observation'].shape[0])])
		obs_high = np.array([ 6 for _ in range(env.unwrapped.observation_space['observation'].shape[0])])
		env.unwrapped.observation_space['observation'] = Box(obs_low, obs_high)
		# env.unwrapped.observation_space['desired_goal'].low = 
		# env.unwrapped.observation_space['observation'].low =

	def get_save_state(self):
		data = self.unwrapped.point_env.data
		return {
			'time': deepcopy(data.time),
			'qpos': np.copy(data.qpos[:]),
			'qvel': np.copy(data.qvel[:]),
			'goal': np.copy(self.unwrapped.goal)
		}
	# def get_save_state(self):
	# 	return {
	# 		"qpos": self.unwrapped.point_env.data.qpos,
	# 		"qvel": self.unwrapped.point_env.data.qvel,
	# 		# "goal": self.unwrapped.goal
	# 	}

	def restore_save_state(self, state_dict):
		self.unwrapped.point_env.data.time = state_dict['time']
		self.unwrapped.point_env.set_state(state_dict['qpos'], state_dict['qvel'])
		self.unwrapped.goal = state_dict['goal']


class AntSaveStateWrapper(SaveStateWrapper):
	def __init__(self, env):
		super().__init__(env)
		env.unwrapped._get_obs = types.MethodType(ant_fixed_get_obs, env.unwrapped)
		new_low = np.concatenate([
			env.unwrapped.observation_space['achieved_goal'].low,
			env.unwrapped.observation_space['observation'].low, 
		])
		new_high = np.concatenate([
			env.unwrapped.observation_space['achieved_goal'].high,
			env.unwrapped.observation_space['observation'].high, 
		])
		env.unwrapped.observation_space['observation'] = Box(new_low, new_high)

	def get_save_state(self):
		return {
			"qpos": self.unwrapped.data.qpos,
			"qvel": self.unwrapped.data.qvel,
			"goal": self.unwrapped.goal
		}

	def restore_save_state(self, state_dict):
		self.unwrapped.set_state(state_dict['qpos'], state_dict['qvel'])
		self.unwrapped.goal = state_dict['goal']

