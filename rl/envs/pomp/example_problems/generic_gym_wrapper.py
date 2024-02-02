from OpenGL.GL import *
# from .geometric import Set
from ..spaces.objectives import TimeObjectiveFunction
# from ..spaces.controlspace import LambdaKinodynamicSpace, GymWrapperControlSpace
# from ..spaces.configurationspace import MultiConfigurationSpace, BoxConfigurationSpace
from ..spaces.gymwrappers import GymWrapperGoalConditionedControlSpace, RLAgentControlSelector, DDPGAgentWrapper
from ..spaces.sets import *#
# from ..spaces.configurationspace import *
# from ..spaces.so2space import SO2Space, so2
from ..spaces.biassets import BoxBiasSet
from ..planners.problem import PlanningProblem

# from ..HER_mod.rl_modules.velocity_env import *
# from HER_mod.rl_modules.velocity_env import CarEnvironment
import math


use_agent = True
# p_goal = .5
p_random = .5
p_goal = 1
# p_random = 0
agent_loc = "saved_models/her_mod_"

def set_state(self, state: np.ndarray):#, goal_length: int):
    # env_state = np.concatenate([np.array([0]),state[:-goal_length]])
    assert type(state) == np.ndarray
    env_state = np.concatenate([np.array([0]),state])
    self.sim.set_state_from_flattened(np.array(env_state))
    self.sim.forward()

def state_to_goal(self, state: np.ndarray):
    assert type(state) == np.ndarray
    env_state = np.concatenate([np.array([0]),state])
    self.sim.set_state_from_flattened(np.array(env_state))
    self.sim.forward()
    obs = self._get_obs()
    return obs['achieved_goal']


class GymEnvWrapper: 
    def __init__(self, Env):
        self.env = Env
        self.setup()


    def setup(self):
        obs = self.env.reset()
        try: 
            self.start_state = obs['observation']#.tolist()
            self.goal = obs['desired_goal'].tolist()
        except: 
            import ipdb
            ipdb.set_trace()
            
        # setattr(self.env, 'set_state', set_state)
        self.control_space = GymWrapperGoalConditionedControlSpace(self.env, self.goal)

    def controlSpace(self):
        return self.control_space

    def controlSet(self):
        return self.control_space.action_set

    def startState(self):
        return self.start_state.tolist()

    def configurationSpace(self):
        return self.control_space.configuration_space
    
    def goalSet(self):
        return self.control_space.goal_set