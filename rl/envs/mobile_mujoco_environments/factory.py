from rl.envs.mobile_mujoco_environments.envs.mushr_env import MushrReachEnv
from rl.envs.mobile_mujoco_environments.envs.quadrotor_env import QuadrotorReachEnv
from rl.envs.mobile_mujoco_environments.envs.trailer_car_env import TrailerCarEnv
from rl.envs.mobile_mujoco_environments.envs.bicycle_env import BicycleReachEnv
from rl.envs.mobile_mujoco_environments.envs.x2_env import X2ReachEnv
from gym.envs.registration import register
import numpy as np

class MushrEnvironmentFactory:
    def __init__(self, return_full_trajectory=False, max_speed=1.0, max_steering_angle=1.0, prop_steps=5, max_steps=250, goal_limits=None, with_obstacles=True):
        self.noise_levels = [0.01, 0.02, 0.04]
        self.noise_level_names = ["Low", "Med", "High"]
        self.return_full_trajectory = return_full_trajectory
        self.max_speed = max_speed
        self.max_steering_angle = max_steering_angle
        self.prop_steps = prop_steps
        self.max_steps = max_steps
        self.goal_limits = goal_limits
        self.with_obstacles = with_obstacles
        assert self.max_speed <= 1.0
        assert self.max_steering_angle <= 1.0
    
    def register_environments_with_position_goals(self):
        register(id="MushrObsEnv-v0",entry_point=MushrReachEnv, kwargs={'noisy': False, 'use_obs': True, 'use_orientation': False, 'return_full_trajectory': self.return_full_trajectory, 'prop_steps': self.prop_steps, 'max_steps': self.max_steps})

        register(id="QuadrotorObsEnv-v0", entry_point=QuadrotorReachEnv,
                 kwargs={'noisy': False, 'use_obs': True, 'use_orientation': False,
                         'return_full_trajectory': self.return_full_trajectory, 'prop_steps': self.prop_steps, 'max_steps': self.max_steps})
        
        register(id="BicycleObsEnv-v0", entry_point=BicycleReachEnv,
                 kwargs={'noisy': False, 'use_obs': True, 'use_orientation': False,
                         'return_full_trajectory': self.return_full_trajectory, 'prop_steps': self.prop_steps, 'max_steps': self.max_steps, 'goal_limits': self.goal_limits})

        register(id="X2ObsEnv-v0", entry_point=X2ReachEnv,
                 kwargs={'noisy': False, 'use_obs': True, 'use_orientation': False,
                         'return_full_trajectory': self.return_full_trajectory, 'prop_steps': self.prop_steps,
                         'max_steps': self.max_steps, 'goal_limits': self.goal_limits, 'with_obstacles': self.with_obstacles})

        for i, noise_level in enumerate(self.noise_levels):
            register(id="MushrObs"+self.noise_level_names[i]+"NoisyEnv-v0",entry_point=MushrReachEnv, kwargs={'noisy': True, 'use_obs': True, 'use_orientation': False, 'noise_scale': noise_level, 
            'return_full_trajectory': self.return_full_trajectory, 'prop_steps': self.prop_steps, 'max_steps': self.max_steps})

            register(id="QuadrotorObs" + self.noise_level_names[i] + "NoisyEnv-v0", entry_point=QuadrotorReachEnv,
                     kwargs={'noisy': True, 'use_obs': True, 'use_orientation': False, 'noise_scale': noise_level,
                             'return_full_trajectory': self.return_full_trajectory, 'prop_steps': self.prop_steps, 'max_steps': self.max_steps})
        
    def register_environments_with_position_and_orientation_goals(self):
        register(id="MushrObsEnv-v0",entry_point=MushrReachEnv, kwargs={'noisy': False, 'use_obs': True, 'use_orientation': True, 'return_full_trajectory': self.return_full_trajectory, 'prop_steps': self.prop_steps, 'max_steps': self.max_steps})
        register(id="TrailerCarEnv-v0",entry_point=TrailerCarEnv, kwargs={'return_full_trajectory': self.return_full_trajectory, 'prop_steps': self.prop_steps, 'max_steps': self.max_steps})
        for i, noise_level in enumerate(self.noise_levels):
            register(id="MushrObs"+self.noise_level_names[i]+"NoisyEnv-v0",entry_point=MushrReachEnv, kwargs={'noisy': True, 'use_obs': True, 'use_orientation': True, 'noise_scale': noise_level, 'return_full_trajectory': self.return_full_trajectory,  'prop_steps': self.prop_steps, 'max_steps': self.max_steps})

    def get_applied_action(self, action, env_name='MushrObs'):
        if env_name == 'QuadrotorObs':
            return np.array([
                (action[0] + 1.0) * 3.5,
                action[1],
                action[2],
                action[3],
            ])
        elif env_name == 'X2Obs':
            return np.array([
                (action[0] + 1.0) * 5,
                (action[1] + 1.0) * 5,
                (action[2] + 1.0) * 5,
                (action[3] + 1.0) * 5,
            ])
        else:
            return np.array([self.max_steering_angle * action[0], self.max_speed * action[1]])
