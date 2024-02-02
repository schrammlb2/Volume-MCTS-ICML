import numpy as np 
import gym

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class TrailerCarEnv(gym.Env):
    env_limit = 10.0
    distance_threshold = 0.5
    dt = 0.1
    L = 0.25
    d1 = 0.5
    def __init__(self, max_steps=60, return_full_trajectory=False, prop_steps=10):
        self.max_steps = max_steps
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))

        self.obs_dims = 4
        self.goal_dims = 3

        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dims,)),
            "achieved_goal": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.goal_dims,)),
            "desired_goal": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.goal_dims,))
        })

        self.return_full_trajectory = return_full_trajectory
        self.prop_steps = prop_steps
    
    def reset(self, goal=None):
        self.steps = 0
        self.state = np.zeros((self.obs_dims,))
        if goal is None:
            self.state[:2] = np.random.uniform(-self.env_limit, self.env_limit, size=(2,))
            self.state[2] = np.random.uniform(-np.pi, np.pi)
            self.state[3] = self.state[2] + np.random.uniform(-np.pi/4, np.pi/4)
            self.goal = np.random.uniform(-self.env_limit, self.env_limit, size=(self.goal_dims,))
            self.goal[2] = np.random.uniform(-np.pi, np.pi)
        else:
            self.goal = goal
        return self._get_obs()
    
    def _get_obs(self):
        return {
            "observation": np.float32(self.state),
            "achieved_goal": np.float32(self.state[:3]),
            "desired_goal": np.float32(self.goal)
        }
    
    def _norm_angle_pi(self, angle):
        while angle > np.pi:
            angle -= 2*np.pi
        while angle < -np.pi:
            angle += 2*np.pi
        return np.clip(angle, -np.pi, np.pi)
    
    def _propagate_dynamics(self, state, action):
        # state: [x, y ,theta0, theta1]
        # action: [v, phi]
        derivatives = np.zeros((4,))
        derivatives[0] = action[0] * np.cos(state[2])
        derivatives[1] = action[0] * np.sin(state[2])
        derivatives[2] = action[0] * np.tan(action[1]) / self.L
        derivatives[3] = action[0] * np.sin(state[2] - state[3]) / self.d1
        state += derivatives * self.dt
        # Clip the angle
        state[2] = self._norm_angle_pi(state[2])
        state[3] = self._norm_angle_pi(state[3])
        # Enforce that angle between theta0 and theta1 is less than ppi/4
        if self._norm_angle_pi(state[2] - state[3]) > np.pi/4:
            state[3] = self._norm_angle_pi(state[2] - np.pi/4)
        return np.copy(state)

    def _terminal(self,s,g):
        return goal_distance(s,g) < self.distance_threshold

    def compute_reward(self,ag,dg,info):
        return -(goal_distance(ag,dg) >= self.distance_threshold).astype(np.float32)
    
    def step(self, action):
        self.steps += 1

        action = np.clip(action, -1, 1)
        # Scale action[0] to [-0.1,0.5]
        action[0] = (action[0] + 1) * 0.3 + 0.1
        action[1] *= np.pi/3

        current_traj = []
        for _ in range(self.prop_steps):
            self.state = self._propagate_dynamics(self.state, action)
            if self.return_full_trajectory:
                current_traj.append(self.state)
        obs = self._get_obs()
        info = {
            "is_success": self._terminal(obs["achieved_goal"], obs["desired_goal"]),
            "traj": np.array(current_traj)
        }
        done = self._terminal(obs["achieved_goal"], obs["desired_goal"]) or self.steps >= self.max_steps
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {})
        return obs, reward, done, info
