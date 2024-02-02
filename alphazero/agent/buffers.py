from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict


class ReplayBuffer:
    """Replay buffer class.

    The buffer holds all training experiences. It is implemented as a FIFO queue
    using a list and an insertion index. The __next__ implemention allows iteration
    over the class in the training process.

    Attributes
    -------
    max_size: int
        Maximum number of experiences in the buffer.
    batch_size: int
        Training batch size.
    sample_array: np.ndarray
        Array holding the indices of all samples.
    sample_index: int
        Index of the current experience.
    insert_index: int
        Index of where the next experience is inserted.
    size: int
        Current size of the buffer.
    experience: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        List of experiences. An experience consists of a state, actions, visitation counts,
        action values and a value target.
    """

    max_size: int
    batch_size: int
    sample_array: np.ndarray
    sample_index: int
    insert_index: int
    size: int
    experience: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]

    def __init__(self, max_size: int, batch_size: int) -> None:
        """Constructor.

        Parameters
        ----------
        max_size: int
            Maximum size of this instance.
        batch_size: int
            Batch size of this instance.
        """
        self.max_size = max_size
        self.batch_size = batch_size
        self.clear()
        self.sample_index = 0

    def clear(self) -> None:
        """Empties the experience list and resets the insertion index."""
        self.experience = []
        self.insert_index = 0
        self.size = 0

    def store(
        self,
        experience: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Store a single experience in the buffer.

        An experience consists of an environment state, its corresponding actions, visitation counts and
        action values as well as a value target

        Parameters
        ----------
        experience: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple containing the state, actions, visitation counts, action values and the value target.
        """
        if self.size < self.max_size:
            self.experience.append(experience)
            self.size += 1
        else:
            self.experience[self.insert_index] = experience
            self.insert_index += 1
            if self.insert_index >= self.size:
                self.insert_index = 0

    def reshuffle(self) -> None:
        """Reshuffle the buffer and reset its sample index."""
        self.sample_array = np.arange(self.size)
        np.random.shuffle(self.sample_array)
        self.sample_index = 0

    def __iter__(self) -> ReplayBuffer:
        """Return itself as an Iterable."""
        return self

    def __len__(self) -> int:
        """Return the number of experiences in this class."""
        return len(self.experience)

    def __next__(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the next batch of training data.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Batch of collated training experiences.
        """
        if (self.sample_index + self.batch_size > self.size) and (
            not self.sample_index == 0
        ):
            self.reshuffle()  # Reset for the next epoch
            raise (StopIteration)

        assert self.sample_array is not None
        if self.sample_index + 2 * self.batch_size > self.size:
            indices = self.sample_array[self.sample_index :]
            batch = [self.experience[i] for i in indices]
        else:
            indices = self.sample_array[
                self.sample_index : self.sample_index + self.batch_size
            ]
            batch = [self.experience[i] for i in indices]
        self.sample_index += self.batch_size

        # reshape experience into batches
        # action_list = [exp[1] for exp in batch]
        # max_num_actions = max((a.shape[0] for a in action_list))
        # # if 
        try: 
            states, actions, counts, Qs, values = map(np.stack, zip(*batch))
        except: 
            # print("Stack failed, trying alt")
            # states = np.stack([exp[0] for exp in batch])
            # actions = np.stack([exp[1] for exp in batch])
            # counts = np.stack([exp[2] for exp in batch])
            # Qs = np.stack([exp[3] for exp in batch])
            # values = np.stack([exp[4] for exp in batch])
            import ipdb
            ipdb.set_trace()
        
        # import ipdb
        # ipdb.set_trace()
        return states, actions, counts, Qs, values

    next = __next__



class NonConstantActionReplayBuffer:
    """Replay buffer class.

    The buffer holds all training experiences. It is implemented as a FIFO queue
    using a list and an insertion index. The __next__ implemention allows iteration
    over the class in the training process.

    Attributes
    -------
    max_size: int
        Maximum number of experiences in the buffer.
    batch_size: int
        Training batch size.
    sample_array: np.ndarray
        Array holding the indices of all samples.
    sample_index: int
        Index of the current experience.
    insert_index: int
        Index of where the next experience is inserted.
    size: int
        Current size of the buffer.
    experience: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        List of experiences. An experience consists of a state, actions, visitation counts,
        action values and a value target.
    """

    max_size: int
    batch_size: int
    sample_array: np.ndarray
    sample_index: int
    insert_index: int
    size: int
    experience: List[Dict]

    def __init__(self, max_size: int, batch_size: int) -> None:
        """Constructor.

        Parameters
        ----------
        max_size: int
            Maximum size of this instance.
        batch_size: int
            Batch size of this instance.
        """
        self.max_size = max_size
        self.batch_size = batch_size
        self.clear()
        self.sample_index = 0

        self.randomization = "index"
        self.randomization = "array"
        

    def clear(self) -> None:
        """Empties the experience list and resets the insertion index."""
        self.experience = []
        self.insert_index = 0
        self.size = 0

    def store(
        self,
        experience: Dict,
    ) -> None:
        """Store a single experience in the buffer.

        An experience consists of an environment state, its corresponding actions, visitation counts and
        action values as well as a value target

        Parameters
        ----------
        experience: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple containing the state, actions, visitation counts, action values and the value target.
        """
        if self.size < self.max_size:
            self.experience.append(experience)
            self.size += 1
        else:
            self.experience[self.insert_index] = experience
            self.insert_index += 1
            if self.insert_index >= self.size:
                self.insert_index = 0

    def reshuffle(self) -> None:
        """Reshuffle the buffer and reset its sample index."""
        self.sample_array = np.arange(self.size)
        if self.randomization == "array":
            np.random.shuffle(self.sample_array)
        else: 
            # pass
            if np.random.rand() < 0.2:
                np.random.shuffle(self.sample_array)
        self.sample_index = 0

    def __iter__(self) -> ReplayBuffer:
        """Return itself as an Iterable."""
        return self

    def __len__(self) -> int:
        """Return the number of experiences in this class."""
        return len(self.experience)

    def __next__(
        self,
    ):
        # self.sample_index = np.random.randint(0, max(0, self.size-self.batch_size))
        """Returns the next batch of training data.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Batch of collated training experiences.
        """
        if (self.sample_index + self.batch_size > self.size) and (
            not self.sample_index == 0
        ):
            self.reshuffle()  # Reset for the next epoch
            raise (StopIteration)

        assert self.sample_array is not None
        # if self.randomization == "index":
        #     indices = self.sample_array[self.sample_index :]
        #     batch = [self.experience[np.random.randint(0, max(1, self.size))] for i in indices]
        # elif self.randomization == "array": 
        if self.sample_index + 2 * self.batch_size > self.size:
            indices = self.sample_array[self.sample_index :]
            batch = [self.experience[i] for i in indices]
        else:
            indices = self.sample_array[
                self.sample_index : self.sample_index + self.batch_size
            ]
            batch = [self.experience[i] for i in indices]
        # else: 
        #     raise Exception
        self.sample_index += self.batch_size

        # reshape experience into batches
        # action_list = [exp[1] for exp in batch]
        # max_num_actions = max((a.shape[0] for a in action_list))
        # states = np.stack([exp['state'] for exp in batch])
        # # actions = np.stack([exp[1] for exp in batch])
        # actions = [exp['actions'] for exp in batch]
        # counts  = [exp['counts'] for exp in batch]
        # # counts = np.stack([exp[2] for exp in batch])
        # Qs = [exp['Qs'] for exp in batch]
        # try: 
        #     values = np.stack([exp['V_target'] for exp in batch])
        # except: 
        #     import ipdb
        #     ipdb.set_trace()

        # return_dict = {
        #     'states': states, 
        #     'actions': actions,
        #     'counts': counts, 
        #     'Qs': Qs, 
        #     'values': values
        # }
        try: 
            return_dict = {key: [exp[key] for exp in batch] for key, value in batch[0].items()}
            return_dict['states'] = np.stack(return_dict['state'])
            return_dict['values'] = np.stack(return_dict['V_target'])
            return_dict['n'] = np.stack(return_dict['n'])
            return_dict['mu'] = np.stack(return_dict['mu'])
            return_dict['sigma'] = np.stack(return_dict['sigma'])
        except: 
            import ipdb
            ipdb.set_trace()
        # return_dict['values'] = np.stack([exp['V_target'] for exp in batch])
        return return_dict

    next = __next__

from HER.rl_modules.replay_buffer import nonepisodic_replay_buffer
class WrapperReplayBuffer:
    #Wrapper around Stable Baselines buffer
    #Supposed to be more efficient
    max_size: int
    batch_size: int
    sample_array: np.ndarray
    sample_index: int
    insert_index: int
    size: int
    experience: List[Dict]

    def __init__(self, max_size: int, batch_size: int) -> None:
        self.max_size = max_size
        self.batch_size = batch_size
        self.clear()
        self.sample_index = 0

        self.randomization = "index"
        self.randomization = "array"
        

    def clear(self) -> None:
        pass

    def store(
        self,
        experience: Dict,
    ) -> None:
        pass

    def reshuffle(self) -> None:
        pass

    def __iter__(self) -> ReplayBuffer:
        """Return itself as an Iterable."""
        return self

    def __len__(self) -> int:
        """Return the number of experiences in this class."""
        return len(self.experience)

    def __next__(
        self,
    ):
        pass


class StableBaselinesBuffer:
    # Buffer adapted from stable baselines. 
    # Faster than existing buffer
    def __init__(self, env_params, buffer_size, batch_size, replay_k):
        assert type(replay_k)==int
        self.env_params = env_params
        self.size = buffer_size 
        self.replay_k = replay_k
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        # create the buffer to store info
        self.init_buffers()
        # thread lock
        # self.lock = threading.Lock()
        self.batch_size = batch_size

    def init_buffers(self):
        arr = lambda x: np.empty([self.size] + x)
        self.buffers = {
                        'mu': arr([self.env_params['action']]), 
                        'sigma': arr([self.env_params['action']]), 
                        'V_hat': arr([1]),
                        'state': arr([self.env_params['obs']]),
                        'actions': arr([self.env_params['action']]), 
                        'lambda': arr([1]),
                        'counts': arr([1]), 
                        'Qs': arr([1]), 
                        'V_target': arr([1]), 
                        'n': arr([1]), 
                        'total_volume': arr([1]),
                        'hi': arr([self.env_params['obs']]), 
                        'lo': arr([self.env_params['obs']]), 
                        'prob': arr([1]), 
                        'base_prob': arr([1]), 
                        'volume': arr([1]), 
                        'unweighted_volume': arr([1]), 
                        'inv_density': arr([1]), 
                        'local_volume': arr([1]), 
                        'root_state': arr([self.env_params['obs']]),
                        'policy': arr([1]), 
                        'traj_value': arr([1]), 
                        'children_unweighted_density': arr([1]),
        }
    
    # store the episode
    def store_episode(self, episode_batch):
        # batch_size = mb_obs.shape[0]
        try: 
            batch_size = episode_batch['mu'].shape[0]
        except: 
            import ipdb
            ipdb.set_trace()
        # with self.lock:
        idxs = self._get_storage_idx(inc=batch_size)
        # store the informations
        for key in episode_batch.keys():
            self.buffers[key][idxs] = episode_batch[key]
        self.n_transitions_stored += batch_size
        #end lock

    def store(self, batch):
        self.store_episode(batch)
    
    # sample the data from the replay buffer
    def sample(self, batch_size, off_goal=True):
        temp_buffers = {}
        idxs = np.random.randint(0, self.current_size, batch_size)
        # with self.lock:
        for key in self.buffers.keys():
            if key == "state":
                out_key = "states"
            elif key == "V_target":
                out_key = "values"
            else: 
                out_key = key

            temp_buffers[out_key] = self.buffers[key][idxs]
        #end lock

        # assert len(temp_buffers['her_goals'].shape) == 3
        # sample transitions

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

    def clear(self):
        self.init_buffers()

    def reshuffle(self):
        pass

    def __iter__(self) -> ReplayBuffer:
        """Return itself as an Iterable."""
        return self

    def __len__(self) -> int:
        """Return the number of experiences in this class."""
        return len(self.experience)

    def __next__(self,):
        return self.sample(self.batch_size)