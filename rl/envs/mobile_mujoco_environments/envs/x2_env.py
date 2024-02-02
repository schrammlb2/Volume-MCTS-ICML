import numpy as np
import os


import gym
import mujoco

from gym import spaces
import rl.envs.mobile_mujoco_environments.env_utils as env_utils

np.set_printoptions(suppress=True)

max_thrust_limit = 5

class X2:
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
        self.data.ctrl[2] = action[2]
        self.data.ctrl[3] = action[3]

class X2ReachEnv(gym.Env):
    env_limit = 10
    distance_threshold = 0.5
    upright_threshold = 0.175

    def __init__(self, max_steps=30, noisy=False, use_obs=False,
                 use_orientation=False, noise_scale=0.01,
                 return_full_trajectory=False, max_speed=1.0, prop_steps=100, goal_limits=None, with_obstacles=False):

        print('Environment Configuration: ')
        print('Max Steps: ', max_steps)
        print('Prop Steps: ', prop_steps)
        print('Goal Limits: ', goal_limits)
        print('With Obstacles: ', with_obstacles)

        if goal_limits is not None:
            low_goal_limit, high_goal_limit = goal_limits
        else:
            high_goal_limit=10
            low_goal_limit=0.15

        self.max_steps = max_steps
        asset_path = "assets/skydio_x2/x2_obstacles.xml" if with_obstacles else "assets/skydio_x2/x2_open.xml"
        self.x2 = X2(os.path.join(os.path.dirname(__file__), asset_path))
        self.x2.reset()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))

        self.obs_dims = self.x2.model.nq + self.x2.model.nv
        self.goal_dims = 3
        self.high_goal_limit = high_goal_limit
        self.low_goal_limit = low_goal_limit

        low_limits, high_limits = self.get_space_limits()

        self.low_goal_limits = low_limits
        self.high_goal_limits = high_limits

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dims,)),
            "achieved_goal": spaces.Box(low=low_limits, high=high_limits, shape=(self.goal_dims,)),
            "desired_goal": spaces.Box(low=low_limits, high=high_limits, shape=(self.goal_dims,))
        })

        self.return_full_trajectory = return_full_trajectory

        self.prop_steps = prop_steps

    def get_space_limits(self):
        low = []
        high = []

        # For X
        low.append(-self.high_goal_limit)
        high.append(self.high_goal_limit)

        # For Y
        low.append(-self.high_goal_limit)
        high.append(self.high_goal_limit)

        # For Z
        low.append(self.low_goal_limit)
        high.append(self.high_goal_limit)

        return np.array(low, dtype=np.float32), np.array(high, dtype=np.float32)

    def reset(self, goal=None):
        self.x2.reset()
        self.steps = 0

        if goal is None:
            self.goal = np.random.uniform(self.low_goal_limits, self.high_goal_limits,
                                          size=(self.goal_dims,))
        else:
            self.goal = goal
        return self._get_obs()

    def _get_obs(self):
        obs = self.x2.get_obs()

        achieved_goal = np.array([obs[0], obs[1], obs[2]])

        return {
            "observation": np.float32(obs),
            "achieved_goal": np.float32(achieved_goal),
            "desired_goal": np.float32(self.goal)
        }

    def _terminal(self, s, g):
        return env_utils.goal_distance(s, g) < self.distance_threshold

    def compute_reward(self, ag, dg, info):
        return -(env_utils.goal_distance(ag, dg) >= self.distance_threshold).astype(np.float32)

    def is_done(self, obs):
        return (self._terminal(obs["achieved_goal"], obs["desired_goal"]) or
                self.steps >= self.max_steps
        )

    def step(self, action):
        self.steps += 1

        applied_action = np.zeros_like(action)
        applied_action[0] = (action[0] + 1) * (max_thrust_limit/2)
        applied_action[1] = (action[1] + 1) * (max_thrust_limit/2)
        applied_action[2] = (action[2] + 1) * (max_thrust_limit/2)
        applied_action[3] = (action[3] + 1) * (max_thrust_limit/2)
        self.x2.apply_action(applied_action)

        current_traj = []
        for _ in range(self.prop_steps):
            for i in range(self.x2.model.nv): self.x2.data.qacc_warmstart[i] = 0
            mujoco.mj_step(self.x2.model, self.x2.data)
            if self.return_full_trajectory:
                current_traj.append(self._get_obs()["achieved_goal"])
        obs = self._get_obs()
        info = {
            "is_success": self._terminal(obs["achieved_goal"], obs["desired_goal"]),
            "traj": np.array(current_traj)
        }
        done = self.is_done(obs)
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {})
        return obs, reward, done, info


if __name__ == "__main__":
    env = X2ReachEnv(max_steps=100, prop_steps=10, goal_limits=[1.7,2])
    obs = env.reset()
    traj = [np.copy(obs["observation"])]
    for _ in range(2500):
        next_action = env.action_space.sample()
        next_action = np.array([1.0, 1.0, 1.0, 1.0])
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
