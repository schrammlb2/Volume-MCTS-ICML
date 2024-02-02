import numpy as np
import os


# import gym
# from gym import spaces
import gymnasium as gym
from gym import spaces


import mujoco

import rl.envs.mobile_mujoco_environments.env_utils as env_utils

np.set_printoptions(suppress=True)

class Quadrotor:
    def __init__(self, xml_path, env_limit=10):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)

        self.range = np.array([env_limit, env_limit, 2 * np.pi, 2.0, 2 * np.pi / 3])

    def reset(self):
        self.data.qpos = np.copy(self.initial_qpos)
        self.data.qvel = np.copy(self.initial_qvel)
        self.data.ctrl = np.zeros(self.model.nu)
        self.data.time = 0

        mujoco.mj_forward(self.model, self.data)

    def get_obs(self):
        # return np.concatenate([self.data.qpos, self.data.qvel])
        return np.concatenate([
            self.data.qpos[:-2],#/self.range, 
            self.data.qpos[-2:],
            self.data.qvel[:-1],#/self.range,
            self.data.qvel[-1:]])

    def apply_action(self, action):
        self.data.ctrl[0] = action[0]
        self.data.ctrl[1] = action[1]
        self.data.ctrl[2] = action[2]
        self.data.ctrl[3] = action[3]

class QuadrotorReachEnv(gym.Env):
    env_limit = 10
    goal_limit = 1
    distance_threshold = 0.5

    def __init__(self, max_steps=30, noisy=False, use_obs=False,
                 use_orientation=False, noise_scale=0.01,
                 return_full_trajectory=False, max_speed=1.0, prop_steps=100):

        print('Environment Configuration: ')
        print('Max Steps: ', max_steps)
        print('Prop Steps: ', prop_steps)

        self._max_episode_steps = max_steps
        self.quadrotor = Quadrotor(os.path.join(os.path.dirname(__file__), "assets/quadrotor_quat.xml"), self.env_limit)
        self.quadrotor.reset()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))

        self.obs_dims = self.quadrotor.model.nq + self.quadrotor.model.nv
        self.goal_dims = 3

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dims,)),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dims,)),
            "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(self.goal_dims,)),
            "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(self.goal_dims,))
        })

        self.return_full_trajectory = return_full_trajectory

        self.max_speed = max_speed

        self.prop_steps = prop_steps

    def reset(self, seed=None, options={}, goal=None):
        self.quadrotor.reset()
        self.steps = 0

        if goal is None:
            goal_min = [-self.goal_limit, -self.goal_limit, 0]
            goal_max = [self.goal_limit]*3
            self.goal = np.random.uniform(goal_min, goal_max, size=(self.goal_dims, ))
        else:
            self.goal = goal
        return (self._get_obs(), {})

    def _get_obs(self):
        return self._duplicate_get_obs()

    def _duplicate_get_obs(self):
        obs = self.quadrotor.get_obs()

        achieved_goal = np.array([obs[0], obs[1], obs[2]])

        return {
            "observation": np.float32(obs),
            "state": np.float32(obs),
            "achieved_goal": np.float32(achieved_goal),
            "desired_goal": np.float32(self.goal)
        }

    def _terminal(self, s, g):
        return env_utils.goal_distance(s, g) < self.distance_threshold

    def compute_reward(self, ag, dg, info):
        return -(env_utils.goal_distance(ag, dg) >= self.distance_threshold).astype(np.float32)

    def step(self, action):
        self.steps += 1

        applied_action = np.zeros_like(action)
        applied_action[0] = (action[0] + 1.0) * 3.5
        applied_action[1] = action[1]
        applied_action[2] = action[2]
        applied_action[3] = action[3]
        self.quadrotor.apply_action(applied_action)

        current_traj = []
        for _ in range(self.prop_steps):
            for i in range(self.quadrotor.model.nv): self.quadrotor.data.qacc_warmstart[i] = 0
            mujoco.mj_step(self.quadrotor.model, self.quadrotor.data)
            if self.return_full_trajectory:
                current_traj.append(self._get_obs()["achieved_goal"])
        obs = self._get_obs()
        info = {
            "is_success": self._terminal(obs["achieved_goal"], obs["desired_goal"]),
            "traj": np.array(current_traj)
        }
        done = self._terminal(obs["achieved_goal"], obs["desired_goal"]) or self.steps >= self._max_episode_steps
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {})
        # return obs, reward, done, info
        # import ipdb
        # ipdb.set_trace()
        terminated=done
        truncated=False
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    env = QuadrotorReachEnv(max_steps=10, prop_steps=2)
    obs = env.reset()
    traj = [np.copy(obs["observation"])]
    for _ in range(300):
        next_action = env.action_space.sample()
        next_action = np.array([1.0, 0, 0, 0])
        obs, reward, done, _ = env.step(next_action)
        traj.append(np.copy(obs["observation"]))
        print("Achieved: ", obs["achieved_goal"])
        print("Desired: ", obs["desired_goal"])
        print("Reward: ", reward)
        print("==========================================")
        if done:
            print("Done")
            break

    traj = np.array(traj)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection="3d")
    # ax.axes.set_xlim3d(left=-env.env_limit, right=env.env_limit)
    # ax.axes.set_ylim3d(bottom=-env.env_limit, top=env.env_limit)
    # ax.axes.set_zlim3d(bottom=-env.env_limit, top=env.env_limit)
    traj = np.vstack(traj)
    ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.savefig("env_test.png")
