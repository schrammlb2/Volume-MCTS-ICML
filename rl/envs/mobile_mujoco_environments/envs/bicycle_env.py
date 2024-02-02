import numpy as np
import os


import gym
import mujoco

from gym import spaces

import rl.envs.mobile_mujoco_environments.env_utils

np.set_printoptions(suppress=True)

class Bicycle:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)

    def reset(self):
        self.data.qpos = np.copy(self.initial_qpos)
        self.data.qvel = np.copy(self.initial_qvel)
        self.data.ctrl = np.zeros(self.model.nu)
        self.data.time = 0

        mujoco.mj_forward(self.model, self.data)

    def get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def apply_action(self, action):
        self.data.ctrl[0] = action[0]
        self.data.ctrl[1] = action[1]

class BicycleReachEnv(gym.Env):
    env_limit = 10
    distance_threshold = 0.5
    upright_threshold = 0.175

    def __init__(self, max_steps=30, noisy=False, use_obs=False,
                 use_orientation=False, noise_scale=0.01,
                 return_full_trajectory=False, max_speed=1.0, prop_steps=100, goal_limits=None):

        print('Environment Configuration: ')
        print('Max Steps: ', max_steps)
        print('Prop Steps: ', prop_steps)

        if goal_limits is not None:
            low_goal_limit, high_goal_limit, goal_angle_limit = goal_limits
        else:
            high_goal_limit=10
            low_goal_limit=0.8
            goal_angle_limit=np.pi/3

        self.max_steps = max_steps
        self.bicycle = Bicycle(os.path.join(os.path.dirname(__file__), "assets/bicycle.xml"))
        self.bicycle.reset()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        self.obs_dims = self.bicycle.model.nq + self.bicycle.model.nv
        self.goal_dims = 3
        self.high_goal_limit = high_goal_limit
        self.low_goal_limit = low_goal_limit
        self.goal_angle_limit = goal_angle_limit

        low_limits, high_limits = self.get_space_limits()

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dims,)),
            "achieved_goal": spaces.Box(low=low_limits, high=high_limits, shape=(self.goal_dims,)),
            "desired_goal": spaces.Box(low=low_limits, high=high_limits, shape=(self.goal_dims,))
        })

        self.return_full_trajectory = return_full_trajectory

        self.max_speed = max_speed

        self.prop_steps = prop_steps

    def get_space_limits(self):
        low = []
        high = []

        unit_vec = np.array([1, 0])
        left_rot_vec = env_utils.rot_vector(-self.goal_angle_limit, unit_vec)
        right_rot_vec = env_utils.rot_vector(self.goal_angle_limit, unit_vec)

        # For X
        low.append(0)
        high.append(self.high_goal_limit)

        # For Y
        low.append(left_rot_vec[1])
        high.append(right_rot_vec[1])

        # For Z
        low.append(0)
        high.append(0)

        return np.array(low, dtype=np.float32), np.array(high, dtype=np.float32)

    def reset(self, goal=None):
        self.bicycle.reset()
        self.steps = 0

        if goal is None:
            mag = np.random.uniform(low=self.low_goal_limit, high=self.high_goal_limit)
            ang = np.random.uniform(low=-self.goal_angle_limit, high=self.goal_angle_limit)
            self.goal = np.concatenate([env_utils.rot_vector(ang, np.array([mag, 0])), [0]])
        else:
            self.goal = goal
        return self._get_obs()

    def _get_obs(self):
        obs = self.bicycle.get_obs()

        achieved_goal = np.array([obs[0], obs[1], obs[2]])

        return {
            "observation": np.float32(obs),
            "achieved_goal": np.float32(achieved_goal),
            "desired_goal": np.float32(self.goal)
        }

    def _terminal(self, s, g):
        return env_utils.goal_distance(s[:2], g[:2]) < self.distance_threshold

    def is_upright(self, z):
        return z > self.upright_threshold

    def compute_reward(self, ag, dg, info):
        if len(ag.shape) == 1:
            ag = ag.reshape(1, -1)
            dg = dg.reshape(1, -1)

        rewards = -np.ones(shape=(ag.shape[0], 1))
        rewards[ag[:, 2] < self.upright_threshold] = -10
        rewards[np.logical_and(
            ag[:, 2] > self.upright_threshold,
            env_utils.goal_distance(ag[:, :2], dg[:, :2]) < self.distance_threshold
        )] = 0
        # breakpoint()
        return rewards.astype(np.float32)[0]

    def is_done(self, obs):
        return (self._terminal(obs["achieved_goal"], obs["desired_goal"]) or
                # obs['observation'][2] < 0.18 or
                self.steps >= self.max_steps
        )

    def step(self, action):
        self.steps += 1

        applied_action = np.zeros_like(action)
        applied_action[0] = action[0]
        applied_action[1] = action[1]
        self.bicycle.apply_action(applied_action)

        current_traj = []
        for _ in range(self.prop_steps):
            for i in range(self.bicycle.model.nv): self.bicycle.data.qacc_warmstart[i] = 0
            mujoco.mj_step(self.bicycle.model, self.bicycle.data)
            if self.return_full_trajectory:
                current_traj.append(self._get_obs()["achieved_goal"])
        obs = self._get_obs()
        info = {
            "is_success": self._terminal(obs["achieved_goal"], obs["desired_goal"]),
            "traj": np.array(current_traj)
        }
        done = self.is_done(obs)
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {})[0]
        return obs, reward, done, info


if __name__ == "__main__":
    env = BicycleReachEnv(max_steps=250, prop_steps=5)
    obs = env.reset()
    traj = [np.copy(obs["observation"])]
    for _ in range(2500):
        next_action = env.action_space.sample()
        next_action = np.array([1.0, 1])
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
