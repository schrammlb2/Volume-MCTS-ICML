from rl.envs.pomp.planners import allplanners
from rl.envs.pomp.planners import test
from rl.envs.pomp.example_problems import *
from rl.envs.pomp.spaces.objectives import *
from rl.envs.pomp.spaces.edgechecker import *
import time
import copy
import sys
import os,errno

import gymnasium as gym
from gymnasium.core import Env
from gymnasium_robotics.core import GoalEnv
from gymnasium.spaces import Box, Dict

import numpy as np
import ipdb

class ProblemGymEnv(Env):
	def __init__(self, make_problem, new_api = False):
		self.make_problem = make_problem
		problem = make_problem()
		self.include_goal = False
		self.problem = problem
		state = problem.start
		# self.geometric = (self.problem.controlSpace != None)

		if self.problem.controlSpace:
			space = problem.controlSpace
			self.space = space

			# u_lower = np.array(space.uspace.bounds()[0])
			# u_upper = np.array(space.uspace.bounds()[1])
			u_lower = np.array(space.controlSet(state).bounds()[0])
			u_upper = np.array(space.controlSet(state).bounds()[1])
			self.action_space = Box(low=u_lower, high=u_upper)
			if self.include_goal:
				c_lower = np.array(space.cspace.bounds()[0]*2)
				c_upper = np.array(space.cspace.bounds()[1]*2)
			else: 
				c_lower = np.array(space.cspace.bounds()[0])
				c_upper = np.array(space.cspace.bounds()[1])
			self.observation_space = Box(low=c_lower, high=c_upper)
			self._interpolator = space.interpolator
			import ipdb
			ipdb.set_trace()
			self.max_dist = None
			self.geometric = False
		else: 
			space = problem.configurationSpace
			self.space = space
			if self.include_goal:
				c_lower = np.array(space.bounds()[0]*2)
				c_upper = np.array(space.bounds()[1]*2)
			else: 
				c_lower = np.array(space.bounds()[0])
				c_upper = np.array(space.bounds()[1])
			self.observation_space = Box(low=c_lower, high=c_upper)
			u_lower = np.array(space.bounds()[0])*0 - 1
			u_upper = np.array(space.bounds()[0])*0 + 1
			self.action_space = Box(low=u_lower, high=u_upper)
			self.max_dist = self.problem.goal.r*2
			# self.max_dist = 0.05
			self.geometric = True
		
		self.edgeChecker = EpsilonEdgeChecker(space,0.01)
		# self.edgeChecker = EpsilonEdgeChecker(self.problem.configurationSpace, self.max_dist/10) #0.01)
		self.initialized = False
		self.new_api = new_api

	def unwrapped(self):
		return self

	def map_action(self, state, action):
		scaled_action = action*self.max_dist#*0.8
		return np.array(state) + scaled_action

	def step(self, action):
		# action =action*0.5
		# import ipdb
		# ipdb.set_trace()
		if not self.initialized:
			print("*** gymnasium.error.ResetNeeded: Cannot call env.step() before calling env.reset()")
			assert False
		if self.geometric:
			#clip to l2 norm = 1 here?
			# ipdb.set_trace()
			u = self.map_action(self.state, action).tolist()
		else: 
			u = action.tolist()
			time = np

        
		edge = self.space.interpolator(self.state,u)
		if self.edgeChecker.feasible(edge):
			new_state = edge.end()
		else: 
			new_state = self.state
		self.state = new_state

		return self.get_obs()

	def reset(self):
		# self.problem = self.make_problem()
		# ipdb.set_trace()
		self.goal = self.problem.goal.c
		self.state = self.problem.start
		obs = self.get_obs()
		self.initialized = True
		return (obs[0], obs[-1])
		# if self.new_api:
		# 	return (obs[0], obs[-1])
		# else:
		# 	return obs[0]

	def get_state(self):
		return self.state

	def set_state(self, state):
		self.state = state

	def get_obs(self):        
		if self.include_goal:
			observation = np.array(self.state + self.goal).reshape(-1)
		else:
			observation = np.array(self.state).reshape(-1)

		reward = self._reward(self.state)
		terminated = False
		truncated = False
		info = {'done': False}
		if self.include_goal:
			assert observation.shape == (4,)
		else:
			assert observation.shape == (2,)
		assert reward.shape == ()
		if self.new_api:
			return observation, reward, terminated, truncated, info
		else: 
			return observation, reward, terminated, info

	def _reward(self, state):
		# return np.array(0.) if self.problem.goal.contains(state) else np.array(-1.)
		distance_func = lambda x, y: sum(((xi - yi)**2) for xi, yi in zip(x, y))**0.5
		# return -np.array(distance_func(self.goal, state))
		# return np.array(0.) if self.problem.goal.contains(state) else np.array(-1.)
		return np.array(1.) if self.problem.goal.contains(state) else np.array(0.)
		# pass



# class ProblemGymGoalEnv(GoalEnv):
# 	def __init__(self, make_problem, new_api = False):
# 		self.make_problem = make_problem
# 		problem = make_problem()
# 		self.include_goal = False
# 		self.problem = problem
# 		state = problem.start
# 		# self.geometric = (self.problem.controlSpace != None)

# 		if self.problem.controlSpace:
# 			print("We haven't addressed the control space problem yet")
# 			raise NotImplementedError
# 			space = problem.controlSpace
# 			self.space = space

# 			u_lower = np.array(space.controlSet(state).bounds()[0])
# 			u_upper = np.array(space.controlSet(state).bounds()[1])
# 			self.action_space = Box(low=u_lower, high=u_upper)
# 			if self.include_goal:
# 				c_lower = np.array(space.cspace.bounds()[0]*2)
# 				c_upper = np.array(space.cspace.bounds()[1]*2)
# 			else: 
# 				c_lower = np.array(space.cspace.bounds()[0])
# 				c_upper = np.array(space.cspace.bounds()[1])
# 			self.observation_space = Box(low=c_lower, high=c_upper)
# 			self._interpolator = space.interpolator
# 			self.max_dist = None
# 			self.geometric = False
# 		else: 
# 			space = problem.configurationSpace
# 			self.space = space
# 			c_lower = np.array(space.bounds()[0])
# 			c_upper = np.array(space.bounds()[1])
# 			self.observation_space = Dict({
# 				"observation" : Box(low=c_lower, high=c_upper),
# 				"achieved_goal" : Box(low=c_lower, high=c_upper),
# 				"desired_goal" : Box(low=c_lower, high=c_upper)
# 			})
# 			u_lower = np.array(space.bounds()[0])*0 - 1
# 			u_upper = np.array(space.bounds()[0])*0 + 1
# 			self.action_space = Box(low=u_lower, high=u_upper)
# 			self.max_dist = self.problem.goal.r*2
# 			# self.max_dist = 0.05
# 			self.geometric = True
		
# 		# self.edgeChecker = EpsilonEdgeChecker(space,0.01)
# 		self.edgeChecker = EpsilonEdgeChecker(self.problem.configurationSpace, self.max_dist/10) #0.01)
# 		self.initialized = False
# 		self.new_api = new_api
# 		self.reward_range = (0,1)

# 	def unwrapped(self):
# 		return self

# 	def map_action(self, state, action):
# 		scaled_action = action*self.max_dist#*0.8
# 		return np.array(state) + scaled_action

# 	def step(self, action):
# 		if not self.initialized:
# 			print("*** gymnasium.error.ResetNeeded: Cannot call env.step() before calling env.reset()")
# 			assert False
# 		if self.geometric:
# 			u = self.map_action(self.state, action).tolist()
# 		else: 
# 			u = action.tolist()
# 			time = np

        
# 		edge = self.space.interpolator(self.state,u)
# 		if self.edgeChecker.feasible(edge):
# 			new_state = edge.end()
# 		else: 
# 			new_state = self.state
# 		self.state = new_state

# 		return self.get_obs()

# 	def reset(self, seed=0, options={}):
# 		self.goal = self.problem.goal.c
# 		self.state = self.problem.start
# 		obs = self.get_obs()
# 		self.initialized = True
# 		return (obs[0], obs[-1])

# 	def get_state(self):
# 		return self.state

# 	def set_state(self, state):
# 		self.state = state

# 	def get_obs(self):    
# 		observation = self._get_obs()
# 		reward = self.compute_reward(self.state, self.goal, {})
# 		terminated = False
# 		truncated = False
# 		info = {
# 			'done': False, 
# 			'is_success': reward > 0.5, 
# 		}
# 		assert reward.shape == ()
# 		return observation, reward, terminated, truncated, info
		

# 	def _get_obs(self):   
# 		return {
# 			'observation': np.array(self.state).reshape(-1),
# 			'achieved_goal': np.array(self.state).reshape(-1),
# 			'desired_goal': np.array(self.goal).reshape(-1),
# 		}
		

# 	def compute_reward(self, ag, dg, info):
# 		r = self.problem.goal.r
# 		# distance_func = lambda x, y: sum(((xi - yi)**2) for xi, yi in zip(x, y))**0.5
# 		distance_func = lambda x, y: np.linalg.norm(
# 			np.array(x)-np.array(y), axis=-1)
# 		return np.where(distance_func(ag, dg) <= r, 1, 0)
# 		# if distance_func(ag, dg) <= r:
# 		# 	return np.array(1.)
# 		# else:
# 		# 	return np.array(0.)

# 	def _reward(self, state):
# 		# return np.array(0.) if self.problem.goal.contains(state) else np.array(-1.)
# 		distance_func = lambda x, y: sum(((xi - yi)**2) for xi, yi in zip(x, y))**0.5
# 		# return -np.array(distance_func(self.goal, state))
# 		# return np.array(0.) if self.problem.goal.contains(state) else np.array(-1.)
# 		return np.array(1.) if self.problem.goal.contains(state) else np.array(0.)
# 		# pass


def action_normalize(s, upper_bound, lower_bound):
	return (s+lower_bound)*2/(upper_bound-lower_bound) - 1
	# return s*2/size - 1

def action_denormalize(s, upper_bound, lower_bound):
	return (s+1)*(upper_bound-lower_bound)/2 - lower_bound
	# return (s+1)*size/2


class ProblemGymGoalEnv(Env):
	def __init__(self, make_problem, goal_dim, new_api = False, goal_conditioned=False):
		self.make_problem = make_problem
		problem = make_problem()
		self.include_goal = True
		self.problem = problem
		state = problem.start
		self.goal_conditioned = goal_conditioned
		# self.geometric = (self.problem.controlSpace != None)
		self.goal_dim = 2

		if self.problem.controlSpace != None:
			space = problem.controlSpace
			self.space = space

			# u_lower = np.array(space.uspace.bounds()[0])
			# u_upper = np.array(space.uspace.bounds()[1])
			u_lower = np.array(space.controlSet(state).bounds()[0])
			u_upper = np.array(space.controlSet(state).bounds()[1])
			self.action_lower = u_lower
			self.action_upper = u_upper
			self.action_space = Box(low=-np.ones_like(u_lower), high=np.ones_like(u_upper))
			# self.action_space = Box(low=-u_lower, high=u_upper)
			# if self.include_goal:
			# 	c_lower = np.array(space.cspace.bounds()[0]*2)
			# 	c_upper = np.array(space.cspace.bounds()[1]*2)
			# else: 
			# 	c_lower = np.array(space.cspace.bounds()[0])
			# 	c_upper = np.array(space.cspace.bounds()[1])

			problem.configurationSpace
			if self.include_goal:
				full_dim = self.goal_dim + len(problem.configurationSpace.bounds()[0])
				# c_lower = np.array(problem.configurationSpace.bounds()[0]*2)[:full_dim]
				# c_upper = np.array(problem.configurationSpace.bounds()[1]*2)[:full_dim]
				c_lower = np.array(problem.configurationSpace.bounds()[0]*2)
				c_upper = np.array(problem.configurationSpace.bounds()[1]*2)
			else: 
				c_lower = np.array(problem.configurationSpace.bounds()[0])
				c_upper = np.array(problem.configurationSpace.bounds()[1])

			# self.observation_space = Box(low=c_lower, high=c_upper)
			if self.goal_conditioned:
				# c_lower = np.array(problem.configurationSpace.bounds()[0])[:self.goal_dim]
				# c_upper = np.array(problem.configurationSpace.bounds()[1])[:self.goal_dim]
				c_lower = np.array(problem.configurationSpace.bounds()[0])
				c_upper = np.array(problem.configurationSpace.bounds()[1])
				self.observation_space = Dict({
					"observation" : Box(low=c_lower, high=c_upper),
					"state" : Box(low=c_lower, high=c_upper),
					"achieved_goal" : Box(low=c_lower, high=c_upper),
					"desired_goal" : Box(low=c_lower, high=c_upper)
				})
			else:
				self.observation_space = Box(low=c_lower, high=c_upper)
				
			self._interpolator = space.interpolator
			self.max_dist = self.problem.goal.r*2
			self.geometric = False
		else: 
			space = problem.configurationSpace
			self.space = space
			if self.include_goal:
				c_lower = np.array(space.bounds()[0]*2)
				c_upper = np.array(space.bounds()[1]*2)
			else: 
				c_lower = np.array(space.bounds()[0])
				c_upper = np.array(space.bounds()[1])
			self.observation_space = Box(low=c_lower, high=c_upper)
			if self.goal_conditioned:
				c_lower = np.array(space.bounds()[0])
				c_upper = np.array(space.bounds()[1])
				self.observation_space = Dict({
					"observation" : Box(low=c_lower, high=c_upper),
					"state" : Box(low=c_lower, high=c_upper),
					"achieved_goal" : Box(low=c_lower, high=c_upper),
					"desired_goal" : Box(low=c_lower, high=c_upper)
				})
			else:
				self.observation_space = Box(low=c_lower, high=c_upper)

			u_lower = np.array(space.bounds()[0])*0 - 1
			u_upper = np.array(space.bounds()[0])*0 + 1
			self.action_lower = u_lower
			self.action_upper = u_upper
			self.action_space = Box(low=-np.ones_like(u_lower), high=np.ones_like(u_upper))
			# self.action_space = Box(low=-u_lower, high=u_upper)
			self.max_dist = self.problem.goal.r*2
			# self.max_dist = 0.05
			self.geometric = True
		
		# self.edgeChecker = EpsilonEdgeChecker(space,0.01)
		self.edgeChecker = EpsilonEdgeChecker(self.problem.configurationSpace, self.max_dist/10) #0.01)
		self.initialized = False
		self.new_api = new_api
		# import ipdb 
		# ipdb.set_trace()

	def unwrapped(self):
		return self

	def map_action(self, state, action):
		# scaled_action = action*self.max_dist#*0.8
		# return np.array(state) + scaled_action
		if self.geometric:
			scaled_action = action*self.max_dist#*0.8
			return np.array(state) + scaled_action
		else:
			return action_denormalize(action, self.action_upper, self.action_lower)

	def step(self, action):
		# action =action*0.5
		# import ipdb
		# ipdb.set_trace()
		if not self.initialized:
			print("*** gymnasium.error.ResetNeeded: Cannot call env.step() before calling env.reset()")
			assert False
		if self.geometric:
			#clip to l2 norm = 1 here?
			# ipdb.set_trace()
			u = self.map_action(self.state, action).tolist()
		else: 
			u = action.tolist()
			time = np

        
		edge = self.space.interpolator(self.state,u)
		if self.edgeChecker.feasible(edge):
			new_state = edge.end()
		else: 
			new_state = self.state
		self.state = new_state

		return self.get_obs()

	def reset(self, seed=0, options={}):
		# self.problem = self.make_problem()
		# ipdb.set_trace()
		self.goal = self.problem.goal.c
		self.state = self.problem.start
		obs = self.get_obs()
		self.initialized = True
		return (obs[0], obs[-1])
		# if self.new_api:
		# 	return (obs[0], obs[-1])
		# else:
		# 	return obs[0]

	def get_save_state(self):
		return (self.state, self.goal)

	def restore_save_state(self, data):
		self.state = data[0]
		self.goal = data[1]

	def set_state(self, env, state):
		# env.restore_save_state(state_vec, self.goal)
		state_vec = np.array(state).tolist()
		try:
			assert len(state_vec) == len(env.state)
		except: 
			import ipdb
			ipdb.set_trace()
		env.state = state_vec


	def get_obs(self):        
		observation = self._get_obs()

		# reward = self._reward(self.state)
		reward = self.compute_reward(self.state, self.goal, {})
		terminated = False
		truncated = False
		info = {'done': False, 'is_success': reward > 0.5}
		assert reward.shape == ()
		return observation, reward, terminated, truncated, info

	def _get_obs(self):   
		observation = {
			'observation': np.array(self.state).reshape(-1),
			'state': np.array(self.state).reshape(-1),
			# 'achieved_goal': np.array(self.state[:self.goal_dim]).reshape(-1),
			# 'desired_goal': np.array(self.goal[:self.goal_dim]).reshape(-1),
			'achieved_goal': np.array(self.state).reshape(-1),
			'desired_goal': np.array(self.goal).reshape(-1),
		}
		if self.goal_conditioned:
			return observation
		else:
			return np.concatenate([
				observation['observation'], 
				observation['desired_goal']
			])

	def compute_reward(self, ag, dg, info):
		# import ipdb
		# ipdb.set_trace()
		r = self.problem.goal.r
		# distance_func = lambda x, y: sum(((xi - yi)**2) for xi, yi in zip(x, y))**0.5
		distance_func = lambda x, y: np.linalg.norm(
			np.array(x)[...,:self.goal_dim]-np.array(y)[...,:self.goal_dim], axis=-1)
		return np.where(distance_func(ag, dg) <= r, 1, 0) #+ 100
		# return np.where(distance_func(ag, dg) <= r, 0, -1)
		# if distance_func(ag, dg) <= r:

	def _reward(self, state):
		# return np.array(0.) if self.problem.goal.contains(state) else np.array(-1.)
		distance_func = lambda x, y: sum(((xi - yi)**2) for xi, yi in zip(x, y))**0.5
		# return -np.array(distance_func(self.goal, state))
		# return np.array(0.) if self.problem.goal.contains(state) else np.array(-1.)
		return np.array(1.) if self.problem.goal.contains(state) else np.array(0.)
		# pass


class WaypointProblemGymGoalEnv(ProblemGymGoalEnv):
	def __init__(self, make_problem, goal_dim, new_api = False, goal_conditioned=False):
		self.make_problem = make_problem
		problem = make_problem()
		self.include_goal = True
		self.problem = problem
		state = problem.start
		self.goal_conditioned = goal_conditioned
		# self.geometric = (self.problem.controlSpace != None)
		self.goal_dim = 2

		self.n_waypoints = 1
		self.waypoint_flags = [0]*self.n_waypoints
		waypoint_lower = [0]*self.n_waypoints
		waypoint_upper = [1]*self.n_waypoints

		if self.problem.controlSpace != None:
			space = problem.controlSpace
			self.space = space

			u_lower = np.array(space.controlSet(state).bounds()[0])
			u_upper = np.array(space.controlSet(state).bounds()[1])
			self.action_lower = u_lower
			self.action_upper = u_upper
			self.action_space = Box(low=-np.ones_like(u_lower), high=np.ones_like(u_upper))

			problem.configurationSpace
			if self.include_goal:
				full_dim = self.goal_dim + len(problem.configurationSpace.bounds()[0])
				c_lower = np.array(problem.configurationSpace.bounds()[0]*2)
				c_upper = np.array(problem.configurationSpace.bounds()[1]*2)
				obs_c_lower = np.array(problem.configurationSpace.bounds()[0]*2 + waypoint_lower)
				obs_c_upper = np.array(problem.configurationSpace.bounds()[1]*2 + waypoint_upper)
			else: 
				c_lower = np.array(problem.configurationSpace.bounds()[0])
				c_upper = np.array(problem.configurationSpace.bounds()[1])
				obs_c_lower = np.array(problem.configurationSpace.bounds()[0] + waypoint_lower)
				obs_c_upper = np.array(problem.configurationSpace.bounds()[1] + waypoint_upper)

			# self.observation_space = Box(low=c_lower, high=c_upper)
			if self.goal_conditioned:
				c_lower = np.array(problem.configurationSpace.bounds()[0])
				c_upper = np.array(problem.configurationSpace.bounds()[1])
				obs_c_lower = np.array(problem.configurationSpace.bounds()[0] + waypoint_lower)
				obs_c_upper = np.array(problem.configurationSpace.bounds()[1] + waypoint_upper)
				self.observation_space = Dict({
					"observation" : Box(low=obs_c_lower, high=obs_c_upper),
					"state" : Box(low=obs_c_lower, high=obs_c_upper),
					# "observation" : Box(low=c_lower, high=c_upper),
					"achieved_goal" : Box(low=c_lower, high=c_upper),
					"desired_goal" : Box(low=c_lower, high=c_upper)
				})
			else:
				self.observation_space = Box(low=c_lower, high=c_upper)
				
			self._interpolator = space.interpolator
			self.max_dist = self.problem.goal.r*2
			self.geometric = False
		else: 
			space = problem.configurationSpace
			self.space = space
			if self.include_goal:
				c_lower = np.array(space.bounds()[0]*2)
				c_upper = np.array(space.bounds()[1]*2)
				obs_c_lower = np.array(space.bounds()[0] + waypoint_lower)
				obs_c_upper = np.array(space.bounds()[1] + waypoint_upper)
			else: 
				c_lower = np.array(space.bounds()[0])
				c_upper = np.array(space.bounds()[1])
				obs_c_lower = np.array(space.bounds()[0] + waypoint_lower)
				obs_c_upper = np.array(space.bounds()[1] + waypoint_upper)
			self.observation_space = Box(low=obs_c_lower, high=obs_c_upper)
			if self.goal_conditioned:
				c_lower = np.array(space.bounds()[0])
				c_upper = np.array(space.bounds()[1])
				obs_c_lower = np.array(space.bounds()[0] + waypoint_lower)
				obs_c_upper = np.array(space.bounds()[1] + waypoint_upper)
				self.observation_space = Dict({
					"observation" : Box(low=obs_c_lower, high=obs_c_upper),
					"achieved_goal" : Box(low=c_lower, high=c_upper),
					"desired_goal" : Box(low=c_lower, high=c_upper)
				})
			else:
				self.observation_space = Box(low=c_lower, high=c_upper)

			u_lower = np.array(space.bounds()[0])*0 - 1
			u_upper = np.array(space.bounds()[0])*0 + 1
			self.action_lower = u_lower
			self.action_upper = u_upper
			self.action_space = Box(low=-np.ones_like(u_lower), high=np.ones_like(u_upper))
			# self.action_space = Box(low=-u_lower, high=u_upper)
			self.max_dist = self.problem.goal.r*2
			# self.max_dist = 0.05
			self.geometric = True
		
		# self.edgeChecker = EpsilonEdgeChecker(space,0.01)
		self.edgeChecker = EpsilonEdgeChecker(self.problem.configurationSpace, self.max_dist/10) #0.01)
		self.initialized = False
		self.new_api = new_api

		# print(self.observation_space)
		# import ipdb 
		# ipdb.set_trace()

	def _get_obs(self):   
		observation = {
			'observation': np.array(self.state + self.waypoint_flags).reshape(-1),
			'state': np.array(self.state + self.waypoint_flags).reshape(-1),
			'achieved_goal': np.array(self.state).reshape(-1),
			'desired_goal': np.array(self.goal).reshape(-1),
		}
		if self.goal_conditioned:
			return observation
		else:
			return np.concatenate([
				observation['observation'], 
				observation['desired_goal']
			])

	def step(self, action):
		if not self.initialized:
			print("*** gymnasium.error.ResetNeeded: Cannot call env.step() before calling env.reset()")
			assert False
		if self.geometric:
			#clip to l2 norm = 1 here?
			# ipdb.set_trace()
			u = self.map_action(self.state, action).tolist()
		else: 
			u = action.tolist()
			time = np

        
		edge = self.space.interpolator(self.state,u)
		if self.edgeChecker.feasible(edge):
			new_state = edge.end()
		else: 
			new_state = self.state
		self.state = new_state

		return self.get_obs()

def maze_env(dynamics, n_divs = 10, new_api = False):
    assert dynamics in ["geometric", "dubins"]
    if dynamics == "geometric":
        goal_dim = 2
        make_problem = lambda : geometric.mazeTest(n_divs, goal_setting='fixed')
        # make_problem = lambda : geometric.mazeTest(n_divs, goal_setting='random')
    elif dynamics == "dubins":
        goal_dim = 2
        make_problem = lambda : dubins.dubinsMazeTest(n_divs, goal_setting='fixed')

    # problem = geometric.mazeTest(n_divs, goal_setting='random')
    # problem = geometric.mazeTest(n_divs)
    # problem = pendulum.pendulumTest()
    # make_problem = lambda : geometric.mazeTest(n_divs, goal_setting='random')
    # env = ProblemGymEnv(problem, new_api = new_api)
    # make_problem = lambda : dubins.dubinsCarTest()
    # make_problem = lambda : dubins.dubinsTest2()
    # make_problem = lambda : dubins.dubinsMazeTest(n_divs, goal_setting='fixed')

    goal = False
    from rl.wrappers import (
        NormalizeWrapper,
        ReparametrizeWrapper,
        PILCOWrapper,
        ScaleRewardWrapper,
        ClipRewardWrapper,
        ScaledObservationWrapper,
        GoalWrapper,
        PositiveRewardWrapper,
    )
    from rl.SaveStateWrapper import wrap_with_save_state_wrapper
    from gymnasium.wrappers import StepAPICompatibility
    env = ProblemGymGoalEnv(make_problem, goal_dim, new_api = new_api, goal_conditioned=True)
    # env = WaypointProblemGymGoalEnv(make_problem, goal_dim, new_api = new_api, goal_conditioned=True)
    env = GoalWrapper(env)
    env = StepAPICompatibility(env, output_truncation_bool=False)
        # env = GoalWrapper(env)

    env.reset()
    # import ipdb
    # ipdb.set_trace()
    return env

if __name__=="__main__":
    problem = geometric.mazeTest()
    # problem = dubins.dubinsTest2()
    # problem = dubins.dubinsTest()
    # problem = pendulum.pendulumTest()
    env = ProblemGymEnv(problem)
    env.reset()
    env.step(env.action_space.sample())
    # ipdb.set_trace()