import threading
import numpy as np

"""
the replay buffer here is basically from the openai baselines code

"""
class replay_buffer:
    def __init__(self, env_params, buffer_size, sample_func):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                        'g': np.empty([self.size, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T, self.env_params['action']]),
                        }
        # thread lock
        self.lock = threading.Lock()
    
    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['obs'][idxs] = mb_obs
            self.buffers['ag'][idxs] = mb_ag
            self.buffers['g'][idxs] = mb_g
            self.buffers['actions'][idxs] = mb_actions
            self.n_transitions_stored += self.T * batch_size
    
    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx


class nonepisodic_replay_buffer:
    def __init__(self, env_params, buffer_size, replay_k, reward_func):
        assert type(replay_k)==int
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        self.replay_k = replay_k
        self.reward_func = reward_func
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        # create the buffer to store info
        self.buffers = {
                        'initial_obs': np.empty([self.size, self.env_params['obs']]),
                        'obs': np.empty([self.size, self.env_params['obs']]),
                        'obs_next': np.empty([self.size, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.env_params['goal']]),
                        'ag_next': np.empty([self.size, self.env_params['goal']]),
                        'g': np.empty([self.size, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.env_params['action']]),
                        'her_goals': np.empty([self.size, self.replay_k, self.env_params['goal']]),
                        't_remaining': np.empty([self.size, 1]),
                        'col': np.empty([self.size]),
                        'vol': np.empty([self.size, 1]),
                        }
        # thread lock
        self.lock = threading.Lock()
    
    # store the episode
    def store_episode(self, episode_batch):
        # mb_obs, mb_ag, mb_g, mb_actions, mb_col = episode_batch
        # batch_size = mb_obs.shape[0]
        batch_size = episode_batch['obs'].shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            for key in episode_batch.keys():
                self.buffers[key][idxs] = episode_batch[key]
            # self.buffers['obs'][idxs] = mb_obs
            # self.buffers['ag'][idxs] = mb_ag
            # self.buffers['g'][idxs] = mb_g
            # self.buffers['actions'][idxs] = mb_actions
            # self.buffers['col'][idxs] = mb_col
            self.n_transitions_stored += batch_size
    
    # sample the data from the replay buffer
    def sample(self, batch_size, off_goal=True):
        temp_buffers = {}
        idxs = np.random.randint(0, self.current_size, batch_size)
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][idxs]
        assert len(temp_buffers['her_goals'].shape) == 3
        # sample transitions

        # new_goals = self.buffers['ag_next'][idxs]
        # new_goals = temp_buffers['g'].copy()
        new_goals = self.buffers['her_goals'][idxs, np.random.randint(0, self.replay_k, size=(batch_size,))]
        temp = np.random.uniform(size=batch_size) > 1/(1+self.replay_k)
        her_indexes = np.where(temp)
        temp_buffers['her_used'] = np.where(temp, 1, 0)
        temp_buffers['policy_g'] = temp_buffers['g'].copy()
        temp_buffers['g'][her_indexes] = new_goals[her_indexes]
        temp_buffers['exact_goal'] = np.where(temp_buffers['t_remaining'] > 1, 0, 1)
        temp_buffers['sampled_g'] = new_goals
        temp_buffers['r'] = np.expand_dims(self.reward_func(temp_buffers['ag_next'], temp_buffers['g'], None), 1)

        return temp_buffers

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx
