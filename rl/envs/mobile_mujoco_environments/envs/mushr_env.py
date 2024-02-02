import numpy as np 
import os

import gym
import mujoco

from gym import spaces

import rl.envs.mobile_mujoco_environments.env_utils


class Mushr:
    def __init__(self,xml_path,env_limit=10):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)

        self.range = np.array([env_limit,env_limit,2*np.pi,2.0,2*np.pi/3])

    def reset(self):
        self.data.qpos = np.copy(self.initial_qpos)
        self.data.qvel = np.copy(self.initial_qvel)
        self.data.ctrl = np.zeros(self.model.nu)
        self.data.time = 0
    
        mujoco.mj_forward(self.model,self.data)
    
    def get_obs(self,noisy=False,use_obs=False,noise_scale=0.01):
        if use_obs:
            # Obs: [x, y, theta, v, steering_angle]
            vel = np.sqrt(self.data.qvel[0]**2+self.data.qvel[1]**2)
            s = np.array([self.data.qpos[0], self.data.qpos[1], env_utils.quat2euler(self.data.qpos[3:7])[2], vel, self.data.qpos[7]])
            if noisy:
                s += np.random.normal(0,noise_scale*self.range)
        else:
            s = np.concatenate([self.data.qpos,self.data.qvel])
            if noisy:
                raise NotImplementedError
        return s
    
    def apply_action(self, action, noisy=False):
        self.data.ctrl[0] = action[0] + np.random.normal(0,0.1) if noisy else action[0]
        self.data.ctrl[1] = action[1] + np.random.normal(0,0.1) if noisy else action[1]

class MushrReachEnv(gym.Env):
    env_limit = 10
    distance_threshold = 0.5
    def __init__(self,max_steps=30,noisy=False,use_obs=False,
                use_orientation=False,noise_scale=0.01,
                return_full_trajectory=False, max_speed=1.0, max_steering_angle=1.0,prop_steps=100):
        self.max_steps = max_steps
        self.mushr = Mushr(os.path.join(os.path.dirname(__file__), "assets/mushr_friction_floor.xml"), self.env_limit)
        self.mushr.reset()
        self.action_space = spaces.Box(low=-1,high=1,shape=(2,))

        self.obs_dims = 5 if use_obs else self.mushr.model.nq+self.mushr.model.nv
        self.goal_dims = 3 if use_orientation else 2

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-np.inf,high=np.inf,shape=(self.obs_dims,)),
            "achieved_goal": spaces.Box(low=-np.inf,high=np.inf,shape=(self.goal_dims,)),
            "desired_goal": spaces.Box(low=-np.inf,high=np.inf,shape=(self.goal_dims,))
        })

        self.noisy = noisy
        self.noise_scale = noise_scale
        self.use_obs = use_obs
        self.use_orientation = use_orientation
        self.return_full_trajectory = return_full_trajectory
        
        self.max_speed = max_speed
        self.max_steering_angle = max_steering_angle

        self.prop_steps = prop_steps

    def reset(self,goal=None):
        self.mushr.reset()
        self.steps = 0
        if goal is  None:
            self.goal = np.random.uniform(-self.env_limit,self.env_limit,size=(self.goal_dims,))
            if self.use_orientation:
                self.goal[2] = np.random.uniform(-np.pi,np.pi)
        else:
            self.goal = goal
        return self._get_obs()

    def _get_obs(self):
        obs = self.mushr.get_obs(noisy=self.noisy,use_obs=self.use_obs,noise_scale=self.noise_scale)
        if self.use_obs:
            if self.use_orientation:
                achieved_goal = np.array([obs[0],obs[1],obs[2]])
            else:
                achieved_goal = np.array([obs[0],obs[1]])
        else:
            if self.use_orientation:
                achieved_goal = np.array([obs[0], obs[1], env_utils.quat2euler(obs[3:7])[2]])
            else:
                achieved_goal = np.array([obs[0],obs[1]])  
        return {
            "observation": np.float32(obs),
            "achieved_goal": np.float32(achieved_goal),
            "desired_goal": np.float32(self.goal)
        } 

    def _terminal(self,s,g):
        return env_utils.goal_distance(s, g) < self.distance_threshold

    def compute_reward(self,ag,dg,info):
        return -(env_utils.goal_distance(ag, dg) >= self.distance_threshold).astype(np.float32)

    def step(self,action):
        self.steps += 1
        
        applied_action = np.zeros_like(action)
        applied_action[0] = action[0]*self.max_steering_angle
        applied_action[1] = action[1]*self.max_speed
        self.mushr.apply_action(action,self.noisy)
        
        current_traj = []

        for _ in range(self.prop_steps):
            for i in range(self.mushr.model.nv): self.mushr.data.qacc_warmstart[i] = 0 
            mujoco.mj_step(self.mushr.model,self.mushr.data)
            if self.return_full_trajectory:
                current_traj.append(self._get_obs()["achieved_goal"])

        obs = self._get_obs()
        info = {
            "is_success": self._terminal(obs["achieved_goal"],obs["desired_goal"]),
            "traj": np.array(current_traj)
        }
        done = self._terminal(obs["achieved_goal"],obs["desired_goal"]) or self.steps >= self.max_steps
        reward = self.compute_reward(obs["achieved_goal"],obs["desired_goal"],{})
        return obs,reward,done,info

if __name__ == "__main__":     
    env = MushrReachEnv()
    obs = env.reset()
    traj = [np.copy(obs["observation"])]
    for _ in range(300):
        action = env.action_space.sample()
        action = np.array([0.0,1.0])
        obs, reward, done, _ = env.step(action)
        traj.append(np.copy(obs["observation"]))
        print("Achieved: ",obs["achieved_goal"])
        print("Desired: ",obs["desired_goal"])
        print("Reward: ",reward)
        print("==========================================")
        if done: 
            print("Done")
            break
    
    traj = np.array(traj)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,8))
    plt.xlim(-env.env_limit,env.env_limit)
    plt.ylim(-env.env_limit,env.env_limit)
    traj = np.vstack(traj)
    plt.plot(traj[:,0],traj[:,1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("env_test.png")