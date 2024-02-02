import random
from collections import defaultdict
from typing import Any, Dict, Tuple, Union
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
import numpy as np
import gym
import hydra
from omegaconf.dictconfig import DictConfig

import copy

from alphazero.helpers import stable_normalizer, argmax_key
from alphazero.agent.buffers import ReplayBuffer
from alphazero.agent.losses import A0CLoss
from alphazero.search.mcts import MCTSDiscrete
from alphazero.policy_calculator import calculate_policy, calculate_policy_with_volume 
from alphazero.policy_calculator import calculate_policy_with_volume_2


class Agent(ABC):
    """Abstract base class for the AlphaZero agent.

    Defines the interface and some common methods for the discrete and continuous agent.

    Attributes
    ----------
    device: torch.device
        Torch device. Can be either CPU or cuda.
    nn: Union[DiscretePolicy, DiagonalNormalPolicy, DiagonalGMMPolicy, GeneralizedBetaPolicy]
        Neural network policy used by this agent.
    mcts: Union[MCTSDiscrete, MCTSContinuous]
        Tree search algorithm. Continuous MCTS used progressive widening.
    loss: Union[AlphaZeroLoss, A0CLoss, A0CLossTuned]
        Loss object to train the policy.
    optimizer: torch.optim.Optimizer
        Pytorch optimizer object for performing gradient descent.
    final_selection: str
        String indicating how the final action should be chosen. Can be either "max_visit"
        or "max_value".
    train_epochs: int
        Number of training epochs per episode.
    clip: float
        Value for gradient clipping.
    """

    def __init__(
        self,
        policy_cfg: DictConfig,
        loss_cfg: DictConfig,
        mcts_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        final_selection: str,
        train_epochs: int,
        grad_clip: float,
        device: str,
    ) -> None:
        """Initializer for common attributes of all agent instances.

        Parameters
        ----------
        policy_cfg: DictConfig
            Hydra configuration object for the policy.
        loss_cfg: DictConfig
            Hydra configuration object for the loss.
        mcts_cfg: DictConfig
            Hydra configuration object for the MCTS.
        optimizer_cfg: DictConfig
            Hydra configuration object for the SGD optimizer.
        final_selection: str
            String identifier for the final selection policy. Can be either "max_visit"
            or "max_value".
        train_epochs: int
            Number of training epochs per episode step.
        grad_clip: float
            Gradient clipping value.
        device: str
            Device used to train the network. Can be either "cpu" or "cuda".
        """

        # instantiate network
        self.device = torch.device(device)
        self.nn = hydra.utils.call(policy_cfg).to(torch.device(device))
        self.mcts = hydra.utils.instantiate(mcts_cfg, model=self.nn)
        self.loss = hydra.utils.instantiate(loss_cfg).to(self.device)
        self.optimizer_cfg = optimizer_cfg
        self.optimizer = hydra.utils.instantiate(
            optimizer_cfg, params=self.nn.parameters()
        )

        self.final_selection = final_selection
        self.train_epochs = train_epochs
        self.clip = grad_clip

    @abstractmethod
    def act(
        self,
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Interface for the act method (interaction with the environment)."""
        ...

    @abstractmethod
    def update(
        self, obs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        """Interface for a single gradient descent update step."""
        ...

    @property
    def action_dim(self) -> int:
        """Returns the dimensionality of the action space as int."""
        return self.nn.action_dim

    @property
    def state_dim(self) -> int:
        """Returns the dimensionality of the state space as int."""
        return self.nn.state_dim

    @property
    def n_hidden_layers(self) -> int:
        """Returns the number of hidden layers in the policy network as int."""
        return self.nn.n_hidden_layers

    @property
    def n_hidden_units(self) -> int:
        """Computes the total number of hidden units and returns them as int."""
        return self.nn.n_hidden_units

    @property
    def n_rollouts(self) -> int:
        """Returns the number of MCTS search iterations per environment step."""
        return self.mcts.n_rollouts

    @property
    def learning_rate(self) -> float:
        """Float learning rate of the optimizer."""
        return self.optimizer.lr

    @property
    def c_uct(self) -> float:
        """Constant (float) in the MCTS selection policy weighing the exploration term (UCTS constant)."""
        return self.mcts.c_uct

    @property
    def gamma(self) -> float:
        """Returns the MCTS discount factor as float."""
        return self.mcts.gamma

    def reset_mcts(self, root_state: np.ndarray) -> None:
        """Reset the MCTS by setting the root node to a target environment state.

        Parameters
        ----------
        root_state: np.ndarray
            Environment state defining the new root node.
        """
        self.mcts.root_node = None
        self.mcts.root_state = root_state

    def train(self, buffer: ReplayBuffer) -> Dict[str, Any]:
        """Implementation of a training loop for the neural network.

        The training loop is executed after each environment episode. It is the same
        for both continuous and discrete agents. Differences are in the update method
        which must be implemented for each agent individually.

        Parameters
        ----------
        buffer: ReplayBuffer
            Instance of the replay buffer class containing the training experiences.

        Returns
        -------
        Dict[str, Any]
            Dictionary holding the values of all loss components as float. Keys are the names
            of the loss components.
        """
        buffer.reshuffle()
        running_loss: Dict[str, Any] = defaultdict(float)
        for epoch in range(self.train_epochs):
            for batches, obs in enumerate(buffer):
                loss = self.update(obs)
                for key in loss.keys():
                    running_loss[key] += loss[key]
        for val in running_loss.values():
            val = val / (batches + 1)
        return running_loss


class DiscreteAgent(Agent):
    """Implementation of an AlphaZero agent for discrete action spaces.

    The Discrete agent handles execution of the MCTS as well as network training.
    It interacts with the environment through the act method which executes the search
    and returns the training data.
    Implements an update step for the discrete algorithm is in the update method.

    Attributes
    ----------
    temperature : float
        Temperature parameter for the normalization procedure in the action selection.
    """

    def __init__(
        self,
        policy_cfg: DictConfig,
        mcts_cfg: DictConfig,
        loss_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        final_selection: str,
        train_epochs: int,
        grad_clip: float,
        temperature: float,
        device: str,
    ) -> None:
        """Constructor for the discrete agent.

        Delegates the initialization of components to the ABC constructor.

        Parameters
        ----------
        policy_cfg: DictConfig
            Hydra configuration object for the policy.
        loss_cfg: DictConfig
            Hydra configuration object for the loss.
        mcts_cfg: DictConfig
            Hydra configuration object for the MCTS.
        optimizer_cfg: DictConfig
            Hydra configuration object for the SGD optimizer.
        final_selection: str
            String identifier for the final selection policy. Can be either "max_visit"
            or "max_value".
        train_epochs: int
            Number of training epochs per episode step.
        grad_clip: float
            Gradient clipping value.
        temperature: float
            Temperature parameter for normalizing the visit counts in the final
            selection policy.
        device: str
            Device used to train the network. Can be either "cpu" or "cuda".
        """

        super().__init__(
            policy_cfg=policy_cfg,
            loss_cfg=loss_cfg,
            mcts_cfg=mcts_cfg,
            optimizer_cfg=optimizer_cfg,
            final_selection=final_selection,
            train_epochs=train_epochs,
            grad_clip=grad_clip,
            device=device,
        )

        assert isinstance(self.mcts, MCTSDiscrete)

        # initialize values
        self.temperature = temperature

    def act(  # type: ignore[override]
        self,
        Env: gym.Env,
        deterministic: bool = False,
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Main interface method for the agent to interact with the environment.

        The act method wraps execution of the MCTS search and final action selection.
        It also returns the statistics at the root node for network training.
        The choice of the action to be executed can be either based on visitation counts
        or on action values. Through the deterministic flag it can be specified if this
        choice is samples from the visitation/action value distribution.

        Parameters
        ----------
        Env: gym.Env
            Gym environment from which the MCTS should be executed.
        deterministic: bool = False
            If True, the action with the highest visitation count/action value is executed
            in the environment. If false, the final action is samples from the visitation count
            or action value distribution.

        Returns
        -------
        Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing the action to be executed in the environment and root node
            training information. Elements are:
                - action: MCTS-improved action to be executed in the environment.
                - state: Root node state vector.
                - actions: Root node child actions.
                - counts: Visitation counts at the root node.
                - Qs: Action values at the root node.
                - V: Value target returned from the MCTS.
        """
        self.mcts.search(Env=Env)
        state, actions, counts, Qs, V = self.mcts.return_results(self.final_selection)

        if self.final_selection == "max_value":
            # select final action based on max Q value
            pi = stable_normalizer(Qs, self.temperature)
            action = pi.argmax() if deterministic else np.random.choice(len(pi), p=pi)
        else:
            # select the final action based on visit counts
            pi = stable_normalizer(counts, self.temperature)
            action = pi.argmax() if deterministic else np.random.choice(len(pi), p=pi)

        return action, state, actions, counts, Qs, V

    def mcts_forward(self, action: int, node: np.ndarray) -> None:
        """Moves the MCTS root node to the actually selected node.

        Using the selected node as future root node implements tree reuse.

        Parameters
        ----------
        action: int
            Action that has been selected in the environment.
        node: np.ndarray
            Environment state for the new root node.
        """
        self.mcts.forward(action, node)

    def update(
        self, obs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        """Performs a gradient descent update step.

        This is the main training method for the neural network. Given a batch of observations
        from the replay buffer, it uses the network, optimizer and loss attributes of
        this instance to perform a single update step within the train method.

        Parameters
        ----------
        obs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Batch of observations. Contains:
                - states: Root node states.
                - actions: Selected actions at each root node state.
                - counts: Visitation counts for the actions at each root state.
                - Qs: Action values at the root node (currently unused).
                - V_target: Improved MCTS value targets.

        Returns
        -------
        Dict[str, float]
            A dictionary where the keys are the name of a loss component (full loss, policy loss, value loss, entropy loss)
            and the values are the scalar loss values.
        """

        # zero out gradients
        for param in self.nn.parameters():
            param.grad = None

        # Qs are currently unused in update setp
        states: np.ndarray
        actions: np.ndarray
        counts: np.ndarray
        V_target: np.ndarray
        states, actions, counts, _, V_target = obs
        states_tensor = torch.from_numpy(states).float().to(self.device)
        values_tensor = (
            torch.from_numpy(V_target).unsqueeze(dim=1).float().to(self.device)
        )

        if isinstance(self.loss, A0CLoss):
            actions_tensor = torch.from_numpy(actions).float().to(self.device)
            # regularize the counts to always be greater than 0
            # this prevents the logarithm from producing nans in the next step
            counts += 1
            counts_tensor = torch.from_numpy(counts).float().to(self.device)

            log_probs, entropy, V_hat = self.nn.get_train_data(
                states_tensor, actions_tensor
            )
            loss_dict = self.loss(
                log_probs=log_probs,
                counts=counts_tensor,
                entropy=entropy,
                V=values_tensor,
                V_hat=V_hat,
            )
        else:
            action_probs_tensor = F.softmax(
                torch.from_numpy(counts).float(), dim=-1
            ).to(self.device)
            pi_logits, V_hat = self.nn(states_tensor)
            loss_dict = self.loss(pi_logits, action_probs_tensor, V_hat, values_tensor)

        loss_dict["loss"].backward()

        if self.clip:
            clip_grad_norm(self.nn.parameters(), self.clip)

        self.optimizer.step()

        info_dict = {key: float(value) for key, value in loss_dict.items()}
        return info_dict


class ContinuousAgent(Agent):
    """Implementation of an A0C agent for continuous action spaces.

    The Continuous agent handles execution of the MCTS as well as network training.
    It interacts with the environment through the act method which executes the search
    and returns the training data.
    Implements an update step for the A0C loss in the update method.
    The differences between the continuous agent and the discrete agent are:
        - The continuous agent uses an MCTS with progressive widening.
        - Only the A0C loss and the tuned A0C loss work for this agent.
        - The policy network must use either a normal distribution, a GMM or a Beta distribution.

    Attributes
    ----------
    temperature : float
        Temperature parameter for the normalization procedure in the action selection.
    """

    def __init__(
        self,
        policy_cfg: DictConfig,
        mcts_cfg: DictConfig,
        loss_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        final_selection: str,
        epsilon: float,
        train_epochs: int,
        grad_clip: float,
        device: str,
    ) -> None:
        """Constructor for the discrete agent.

        Delegates the initialization of components to the ABC constructor.

        Parameters
        ----------
        policy_cfg: DictConfig
            Hydra configuration object for the policy.
        loss_cfg: DictConfig
            Hydra configuration object for the loss.
        mcts_cfg: DictConfig
            Hydra configuration object for the MCTS.
        optimizer_cfg: DictConfig
            Hydra configuration object for the SGD optimizer.
        final_selection: str
            String identifier for the final selection policy. Can be either "max_visit"
            or "max_value".
        epsilon: float
            Epsilon value for epsilon-greedy action selection. Epsilon-greedy is disabled
            when this value is set to 0.
        train_epochs: int
            Number of training epochs per episode step.
        grad_clip: float
            Gradient clipping value.
        device: str
            Device used to train the network. Can be either "cpu" or "cuda".
        """

        super().__init__(
            policy_cfg=policy_cfg,
            loss_cfg=loss_cfg,
            mcts_cfg=mcts_cfg,
            optimizer_cfg=optimizer_cfg,
            final_selection=final_selection,
            train_epochs=train_epochs,
            grad_clip=grad_clip,
            device=device,
        )

        self.epsilon = epsilon

    @property
    def action_limit(self) -> float:
        """Returns the action bound for this agent as float."""
        return self.nn.act_limit

    def epsilon_greedy(self, actions: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Epsilon-greedy implementation for the final action selection.

        Parameters
        ----------
        actions: np.ndarray
            Actions to choose from.
        values: np.ndarray
            Values according which the best action is selected. Can be either visitation
            counts or action values.

        Returns
        -------
        np.ndarray
            Action chosen according to epsilon-greedy.
        """
        if random.random() < self.epsilon:
            return np.random.choice(actions)[np.newaxis]
        else:
            return actions[values.argmax()][np.newaxis]

    def act(  # type: ignore[override]
        self,
        Env: gym.Env,
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Main interface method for the agent to interact with the environment.

        The act method wraps execution of the MCTS search and final action selection.
        It also returns the statistics at the root node for network training.
        The choice of the action to be executed can be either the most visited action or
        the action with the highest action value. If the epsilon > 0 is specified when
        instantiating this agent, actions are selected using the epsilon-greedy algorithm.

        Parameters
        ----------
        Env: gym.Env
            Gym environment from which the MCTS should be executed.

        Returns
        -------
        Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing the action to be executed in the environment and root node
            training information. Elements are:
                - action: MCTS-improved action to be executed in the environment.
                - state: Root node state vector.
                - actions: Root node child actions.
                - counts: Visitation counts at the root node.
                - Qs: Action values at the root node.
                - V: Value target returned from the MCTS.
        """
        self.mcts.search(Env=Env)
        state, actions, counts, Qs, V = self.mcts.return_results(self.final_selection)

        if self.final_selection == "max_value":
            if self.epsilon == 0:
                # select the action with the best action value
                action = actions[Qs.argmax()][np.newaxis]
            else:
                action = self.epsilon_greedy(actions=actions, values=Qs)
        else:
            if self.epsilon == 0:
                # select the action that was visited most
                action = actions[counts.argmax()][np.newaxis]
            else:
                action = self.epsilon_greedy(actions=actions, values=counts)


        # if len(action.shape) != 1:
        #     import ipdb
        #     ipdb.set_trace()
        action = action.reshape(-1)

        return action, state, actions, counts, Qs, V

    def update(
        self, obs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        """Performs a gradient descent update step.

        This is the main training method for the neural network. Given a batch of observations
        from the replay buffer, it uses the network, optimizer and loss attributes of
        this instance to perform a single update step within the train method.

        Parameters
        ----------
        obs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Batch of observations. Contains:
                - states: Root node states.
                - actions: Selected actions at each root node state.
                - counts: Visitation counts for the actions at each root state.
                - Qs: Action values at the root node (currently unused).
                - V_target: Improved MCTS value targets.

        Returns
        -------
        Dict[str, float]
            A dictionary where the keys are the name of a loss component (full loss, policy loss, value loss, entropy loss)
            and the values are the scalar loss values.
        """

        # zero out gradients
        for param in self.nn.parameters():
            param.grad = None

        # Qs are currently unused in update
        states: np.ndarray
        actions: np.ndarray
        counts: np.ndarray
        V_target: np.ndarray
        states, actions, counts, Qs, V_target = obs

        actions_tensor = torch.from_numpy(actions).float().to(self.device)
        states_tensor = torch.from_numpy(states).float().to(self.device)
        counts_tensor = torch.from_numpy(counts).float().to(self.device)
        values_tensor = (
            torch.from_numpy(V_target).unsqueeze(dim=1).float().to(self.device)
        )

        log_probs, entropy, V_hat = self.nn.get_train_data(
            states_tensor, actions_tensor
        )

        loss_dict = self.loss(
            log_probs=log_probs,
            counts=counts_tensor,
            entropy=entropy,
            V=values_tensor,
            V_hat=V_hat,
        )

        loss_dict["loss"].backward()

        if self.clip:
            clip_grad_norm(self.nn.parameters(), self.clip)

        self.optimizer.step()

        info_dict = {key: float(value) for key, value in loss_dict.items()}
        return info_dict


class NGUAgent(ContinuousAgent):
    def set_kd_tree(self, kd_tree):
        self.mcts.epoch=self.epoch
        self.mcts.kd_tree = kd_tree


class RPOAgent(ContinuousAgent):
    pass

class RPOAdvantageAgent(RPOAgent):
    def get_tensor_obs(self, obs):        
        # Qs are currently unused in update
        states = obs['states']
        actions = obs['actions']
        counts = obs['counts']
        V_target = obs['values']

        # tensor_obs = {
        #     key: (
        #             [ 
        #                 torch.from_numpy(x).float().to(self.device)
        #                 for x in item
        #             ]
        #             if 
        #                 type(item) is list 
        #             else
        #                 torch.from_numpy(item).float().to(self.device)
        #         )
        #     for key, item in obs.items()
        # }
        # tensor_obs['advantage'] = [
        #     torch.from_numpy(Q-V).float().to(self.device)
        #     for Q, V in zip(obs['Qs'], obs['values'])
        # ]
        tensor_obs = {}
        tensor_obs['states'] = torch.from_numpy(states).float().to(self.device)
        tensor_obs['actions'] = [
            torch.from_numpy(action).float().to(self.device)
            for action in obs['actions']
        ]
        # for action in tensor_obs['actions']:
        #     assert len(action.shape) == 3
        tensor_obs['counts'] = [
            torch.from_numpy(count).float().to(self.device)
            for count in obs['counts']
        ]
        tensor_obs['Qs'] = [
            torch.from_numpy(Q).float().to(self.device)
            for Q in obs['Qs']
        ]
        tensor_obs['values'] = (
            torch.from_numpy(V_target).unsqueeze(dim=1).float().to(self.device)
        )
        tensor_obs['advantage'] = [
            torch.from_numpy(Q-V).float().to(self.device)
            for Q, V in zip(obs['Qs'], obs['values'])
        ]
        tensor_obs['n'] = torch.from_numpy(obs['n']).float().to(self.device)
        tensor_obs['mu']= torch.from_numpy(obs['mu']).float().to(self.device)
        tensor_obs['sigma']= torch.from_numpy(obs['sigma']).float().to(self.device)
        return tensor_obs

    def update(
        self, obs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        # zero out gradients
        for param in self.nn.parameters():
            param.grad = None

        tensor_obs = self.get_tensor_obs(obs)
        train_data_dict = self.nn.get_train_data_generalized(tensor_obs)

        loss_dict = self.loss(tensor_obs=tensor_obs, train_data_dict=train_data_dict)

        loss_dict["loss"].backward()

        if self.clip:
            clip_grad_norm(self.nn.parameters(), self.clip)

        self.optimizer.step()

        try: 
            info_dict = {key: float(value) for key, value in loss_dict.items()}
        except: 
            import ipdb
            ipdb.set_trace()
        return info_dict
        
    def act(  # type: ignore[override]
        self, Env: gym.Env,
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        self.mcts.search(Env=Env)
        result_dict = self.mcts.return_results(self.final_selection)

        self.final_selection = 'rpo'
        if self.final_selection == "max_value":
            if self.epsilon == 0:
                action = result_dict['actions'][result_dict['Qs'].argmax()][np.newaxis]
            else:
                action = self.epsilon_greedy(actions=result_dict['actions'], values=result_dict['Qs'])
        elif self.final_selection == "rpo":
            node = self.mcts.root_node
            lam = self.mcts.c_uct*(node.n)**(-1/2)
            policy = calculate_policy(node.child_actions, lam)
            try:
                action = np.random.choice(node.child_actions, p=policy).action.reshape(-1)
            except:
                import ipdb
                ipdb.set_trace()
        else:
            if self.epsilon == 0:
                action = result_dict['actions'][result_dict['counts'].argmax()][np.newaxis]
            else:
                action = self.epsilon_greedy(actions=result_dict['actions'], values=result_dict['counts'])

        return action, result_dict


class RPO_Whole_Tree_Agent(RPOAdvantageAgent):
    def act(  # type: ignore[override]
        self, Env: gym.Env,
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        self.mcts.search(Env=Env)
        n_cutoff = 25
        def walk_tree(node):
            dicts = []
            if node.n >= n_cutoff:
                dicts.append(self.mcts.return_results(self.final_selection, node))
                for next_action in node.child_actions:
                    dicts += walk_tree(next_action.child_node)
            return dicts


        dicts = walk_tree(self.mcts.root_node)
        result_dict = dicts[0]

        self.final_selection = 'rpo'
        if self.final_selection == "max_value":
            if self.epsilon == 0:
                action = result_dict['actions'][result_dict['Qs'].argmax()][np.newaxis]
            else:
                action = self.epsilon_greedy(actions=result_dict['actions'], values=result_dict['Qs'])
        elif self.final_selection == "rpo":
            node = self.mcts.root_node
            lam = self.mcts.c_uct*(node.n)**(-1/2)
            policy = calculate_policy(node.child_actions, lam)
            action = np.random.choice(node.child_actions, p=policy).action.reshape(-1)
        else:
            if self.epsilon == 0:
                try: 
                    action = result_dict['actions'][result_dict['counts'].argmax()][np.newaxis]
                except:
                    import ipdb
                    ipdb.set_trace()
            else:
                action = self.epsilon_greedy(actions=result_dict['actions'], values=result_dict['counts'])

        # return action, result_dict
        return action, dicts

class Volume_Agent_2(RPO_Whole_Tree_Agent):
    def get_tensor_obs(self, obs): 
        tensor_obs = super().get_tensor_obs(obs)
        # for name in ['lambda', 'total_volume', 'hi', 'lo', 'prob', 'base_prob', 'volume', 'local_volume']:
        for name in ['lambda', 'total_volume', 'hi', 'lo', 'prob', 'base_prob', 
                    'volume', 'local_volume', 'root_state']:
            tensor_obs[name] = torch.tensor(np.array(obs[name])).float().to(self.device)
        return tensor_obs
    
    def act(  # type: ignore[override]
        self, Env: gym.Env,
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        self.mcts.search(Env=Env)
        n_cutoff = 2
        def walk_tree(node):
            dicts = []
            if node.n >= n_cutoff:
                dicts.append(self.mcts.return_results(self.final_selection, node))
                for next_action in node.child_actions:
                    dicts += walk_tree(next_action.child_node)
            return dicts


        dicts = walk_tree(self.mcts.root_node)
        result_dict = dicts[0]

        self.final_selection = 'rpo'
        if self.final_selection == "max_value":
            if self.epsilon == 0:
                action = result_dict['actions'][result_dict['Qs'].argmax()][np.newaxis]
            else:
                action = self.epsilon_greedy(actions=result_dict['actions'], values=result_dict['Qs'])
        elif self.final_selection == "rpo":
            node = self.mcts.root_node
            lam = self.mcts.c_uct*(node.n)**(-1/2)
            # lam = result_dict['lambda']
            # policy = calculate_policy(node.child_actions, lam).reshape(-1)
            policy = calculate_policy_with_volume_2(node, 
                result_dict['n'], 1, with_pw=False, 
                global_total_volume=result_dict['total_volume'])#, 
            #     # local=True)
            if len(policy.shape) != 1:
                import ipdb
                ipdb.set_trace()
            action = np.random.choice(node.child_actions, p=policy).action.reshape(-1)
        else:
            if self.epsilon == 0:
                try: 
                    action = result_dict['actions'][result_dict['counts'].argmax()][np.newaxis]
                except:
                    import ipdb
                    ipdb.set_trace()
            else:
                action = self.epsilon_greedy(actions=result_dict['actions'], values=result_dict['counts'])

        # return action, result_dict
        return action, dicts

class Volume_Agent_3(RPO_Whole_Tree_Agent):
    # pass
    def act(  # type: ignore[override]
        self, Env: gym.Env,
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        self.mcts.search(Env=Env)#, epoch=self.epoch)
        n_cutoff = 25
        def walk_tree(node):
            dicts = []
            if node.n >= n_cutoff:
                dicts.append(self.mcts.return_results(self.final_selection, node))
                for next_action in node.child_actions:
                    dicts += walk_tree(next_action.child_node)
            return dicts


        dicts = walk_tree(self.mcts.root_node)
        result_dict = dicts[0]
        action = result_dict['actions'][result_dict['counts'].argmax()][np.newaxis]
        # return action, result_dict
        return action, dicts

class VolumeAgent(RPOAdvantageAgent):
    pass

class Explorer_Agent(Volume_Agent_2):    
    def get_tensor_obs(self, obs): 
        tensor_obs = super().get_tensor_obs(obs)
        tensor_obs['epoch'] = torch.tensor(obs['epoch']).float().to(self.device)
        return tensor_obs
    def act(  # type: ignore[override]
        self, Env: gym.Env,
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        self.mcts.search(Env=Env, epoch=self.epoch)
        n_cutoff = 25
        def walk_tree(node):
            dicts = []
            if node.n >= n_cutoff:
                dicts.append(self.mcts.return_results(self.final_selection, node))
                for next_action in node.child_actions:
                    dicts += walk_tree(next_action.child_node)
            return dicts


        dicts = walk_tree(self.mcts.root_node)
        result_dict = dicts[0]
        action = result_dict['actions'][result_dict['counts'].argmax()][np.newaxis]
        # return action, result_dict
        return action, dicts

class MultistepAgent(Volume_Agent_2, NGUAgent):
    def action_selection_criteria(self, x):
        return x.n

    def act(  # type: ignore[override]
        self, Env: gym.Env,
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # self.mcts.search(Env=Env, epoch=self.epoch)
        self.mcts.search(Env=Env)
        n_cutoff = 100
        def walk_tree(node):
            dicts = []
            if node.n >= n_cutoff:
                dicts.append(self.mcts.return_results(self.final_selection, node))
                for next_action in node.child_actions:
                    dicts += walk_tree(next_action.child_node)
            return dicts


        dicts = walk_tree(self.mcts.root_node)
        result_dict = dicts[0]
        step_cutoff = 10
        def get_action_sequence(node):
            if node.n >= step_cutoff:
                # action = argmax_key(node.child_actions, lambda x: x.n)
                action = max(node.child_actions, key=self.action_selection_criteria)                
                return [action.action] + get_action_sequence(action.child_node)
            else: 
                return []

        action_list = get_action_sequence(self.mcts.root_node)
        return action_list, dicts
        # action = result_dict['actions'][result_dict['counts'].argmax()][np.newaxis]
        # return action, dicts

class OneShotAgent(MultistepAgent):
    def train(self, buffer: ReplayBuffer) -> Dict[str, Any]:
        buffer.reshuffle()
        running_loss: Dict[str, Any] = defaultdict(float)
        for epoch in range(self.train_epochs*100):
            try: 
                obs = buffer.next()
            except: 
                buffer.reshuffle()
                obs = buffer.next()
            loss = self.update(obs)
            for key in loss.keys():
                running_loss[key] += loss[key]

        for val in running_loss.values():
            val = val / (self.train_epochs)
        return running_loss

    def get_tensor_obs(self, obs): 
        tensor_obs  = super().get_tensor_obs(obs)
        # tensor_obs['epoch'] = torch.tensor(obs['epoch']).float().to(self.device)
        tensor_obs['epoch']     = self.epoch
        tensor_obs['gamma']     = self.gamma
        tensor_obs['traj_value']= torch.tensor(obs['traj_value']).unsqueeze(-1)
        tensor_obs['children_unweighted_density'] = [
            torch.tensor(v)
            for v in obs['children_unweighted_density']
        ]
        tensor_obs['base_prob'] = torch.tensor(obs['base_prob'])

        return tensor_obs

    def action_selection_criteria(self, x):
        return x.Q

    def act(  # type: ignore[override]
        self, Env: gym.Env,
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # self.mcts.search(Env=Env, epoch=self.epoch)
        self.mcts.epoch=self.epoch
        self.mcts.search(Env=Env)
        n_cutoff = 2
        def walk_tree(node):
            dicts = []
            if node.n >= n_cutoff:
                dicts.append(self.mcts.return_results(self.final_selection, node))
                for next_action in node.child_actions:
                    dicts += walk_tree(next_action.child_node)
            return dicts

        self.mcts.preprocess_return_results()

        dicts = walk_tree(self.mcts.root_node)
        result_dict = dicts[0]
        step_cutoff = 2
        def get_action_sequence(node, i=False):
            if i:
                import ipdb
                ipdb.set_trace()
            if node.n >= step_cutoff:
                # action = argmax_key(node.child_actions, lambda x: x.Q)
                action = max(node.child_actions, key=self.action_selection_criteria)    
                # if node.V > action.Q:
                #     return [] #If current state is better, just stay in that state                    
                if node.r >= 1.:
                    return []

                return [action.action] + get_action_sequence(action.child_node, i)
            else: 
                return []


        action_list = get_action_sequence(self.mcts.root_node)
        # import ipdb
        # ipdb.set_trace()
        # action_list = get_action_sequence(self.mcts.root_node, True)
        return action_list, dicts


class OneShotAgent2(OneShotAgent):
    def action_selection_criteria(self, x):
        return x.pessimistic_value


class HERAgent(OneShotAgent2):
    def act(  # type: ignore[override]
        self, Env: gym.Env,
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # self.mcts.search(Env=Env, epoch=self.epoch)
        self.mcts.epoch=self.epoch
        self.mcts.search(Env=Env)

        def random_choice(x): 
            n_sum = sum([i['N'] for i in x])
            indices = np.arange(len(x))
            p = np.array([i['N'] for i in x])/n_sum
            return np.random.choice(x, p=p)

        n_cutoff = 2
        n_goals = 4
        def walk_tree(node):
            dicts = []
            goals = []
            goal = {
                "achieved_goal": None,
                "accumulated_reward": 0, #Accumulated reward between goal being sampled and current state
                "N":1 #Number of times most recent ancestor was expanded
            }
            if node.n >= n_cutoff:
                for next_action in node.child_actions:
                    child_dict, child_goals = walk_tree(next_action.child_node)
                    dicts += child_dict
                    goals += child_goals

                new_dict = self.mcts.return_results(self.final_selection, node)
                new_goal = new_dict['raw_obs']
                new_goal["N"] = 1
                new_goal['accumulated_reward'] = 0
                new_goal['depth'] = 0
                for _ in range(n_goals):
                    goals.append(new_goal)

                # selected_goals = [copy.deepcopy(random_choice(goals)) for _ in range(n_goals)]
                selected_goals = [copy.deepcopy(random_choice(goals)) for _ in range(n_goals)]
                for goal in selected_goals:
                    goal_for_parsing = {
                        "observation": new_dict['raw_obs']['observation'],
                        "desired_goal": goal['achieved_goal']
                    }
                    goal['state'] = Env.observation(goal_for_parsing)
                    goal['accumulated_reward'] = Env.compute_reward(
                        new_dict['raw_obs']['achieved_goal'],
                        goal['achieved_goal'], None
                    ) + self.mcts.gamma*goal['accumulated_reward']
                    goal["N"] = node.n
                    goal["depth"] += 1
                new_dict['HER_goals'] = selected_goals

                dicts.append(new_dict)
            else: 
                selected_goals = []
            return dicts, selected_goals


        self.mcts.preprocess_return_results()

        dicts, _ = walk_tree(self.mcts.root_node)

        result_dict = dicts[0]
        step_cutoff = 2
        def get_action_sequence(node, i=False):
            if i:
                import ipdb
                ipdb.set_trace()
            if node.n >= step_cutoff:
                action = max(node.child_actions, key=self.action_selection_criteria)                   
                if node.r >= 1.:
                    return []

                return [action.action] + get_action_sequence(action.child_node, i)
            else: 
                return []


        action_list = get_action_sequence(self.mcts.root_node)
        return action_list, dicts


    def get_tensor_obs(self, obs): 
        tensor_obs  = super().get_tensor_obs(obs)
        # tensor_obs['epoch'] = torch.tensor(obs['epoch']).float().to(self.device)
        tensor_obs['epoch']     = self.epoch
        tensor_obs['gamma']     = self.gamma
        tensor_obs['traj_value']= torch.tensor(obs['traj_value']).unsqueeze(-1)
        tensor_obs['children_unweighted_density'] = [
            torch.tensor(v)
            for v in obs['children_unweighted_density']
        ]
        tensor_obs['base_prob'] = torch.tensor(obs['base_prob'])
        HER_observations = []
        HER_targets = []
        for goal in obs['HER_goals']:
            for o in goal:
                # HER_observations.append(torch.tensor(o['observation']))
                HER_observations.append(torch.tensor(o['state']))
                HER_targets.append(torch.tensor(o['accumulated_reward']))

        tensor_obs['HER_observations'] = torch.stack(HER_observations, dim=0).float()
        tensor_obs['HER_targets'] = torch.stack(HER_targets, dim=0).unsqueeze(dim=-1).float()

        return tensor_obs




class ExternalTrainingHERAgent(OneShotAgent2):
    def set_network(self, new_network):
        # del self.mcts.model
        # self.HER_model = new_network
        self.nn = new_network
        self.mcts.model = new_network
        self.optimizer = hydra.utils.instantiate(
            self.optimizer_cfg, params=self.nn.parameters()
        )
        # self.HER_model = self.nn
# class HERAgent(OneShotAgent2):
    def act(  # type: ignore[override]
        self, Env: gym.Env,
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # self.mcts.search(Env=Env, epoch=self.epoch)
        self.mcts.epoch=self.epoch
        self.mcts.search(Env=Env)

        def random_choice(x): 
            n_sum = sum([i['N'] for i in x])
            indices = np.arange(len(x))
            p = np.array([i['N'] for i in x])/n_sum
            return np.random.choice(x, p=p)

        n_cutoff = 2
        n_goals = 4
        def walk_tree(node):
            accumulated_her_dicts = []
            dicts = []
            goals = [[] for _ in range(n_goals)]
            goal = {
                "achieved_goal": None,
                "accumulated_reward": 0, #Accumulated reward between goal being sampled and current state
                "N":1 #Number of times most recent ancestor was expanded
            }
            new_dict = self.mcts.return_results(self.final_selection, node)
            if node.n >= n_cutoff:
                her_dicts = []
                for next_action in node.child_actions:
                    child_dict, next_her_dicts, next_s, child_goals = walk_tree(next_action.child_node)
                    dicts += child_dict
                    accumulated_her_dicts += next_her_dicts

                    if len(child_goals) < len(goals):
                        import ipdb
                        ipdb.set_trace()

                    for i in range(n_goals):
                        goals[i].append(child_goals[i])

                    her_dicts.append({
                        'obs': new_dict['raw_obs']['observation'],
                        'obs_next': next_s['observation'],
                        'actions': next_action.action,
                        'ag': new_dict['raw_obs']['achieved_goal'],
                        'ag_next': next_s['achieved_goal'],
                        'g': new_dict['raw_obs']['achieved_goal'],
                        'child_goals': child_goals,
                        't_remaining': [0],
                        'col': False,
                        # 'vol': new_dict['local_volume']/len(node.child_actions),
                        'vol': [next_action.child_node.local_unweighted_volume],
                        # 'all_future_goals':[]
                    })

                new_goal = new_dict['raw_obs']
                new_goal["N"] = 1
                new_goal['accumulated_reward'] = 0
                new_goal['depth'] = 0

                selectable_goals = [[] for _ in range(n_goals)]
                for hd in her_dicts:
                    new_goal_list = [[copy.deepcopy(hd['child_goals'][j])] + [new_goal] for j in range(n_goals)]
                    action_conditioned_her_goals = [random_choice(new_goal_list[j]) for j in range(n_goals)]
                    for j in range(n_goals):
                        action_conditioned_her_goals[j]['N'] = hd['child_goals'][j]['N'] + 1
                        selectable_goals[j].append(action_conditioned_her_goals[j])
                    hd['her_goals'] = action_conditioned_her_goals

                accumulated_her_dicts += her_dicts
                selected_goals = [random_choice(selectable_goals[j]) for j in range(n_goals)]

                for goal in selected_goals:
                    goal['state'] = Env.observation(goal)
                    goal['accumulated_reward'] = Env.compute_reward(
                        new_dict['raw_obs']['achieved_goal'],
                        goal['achieved_goal'], None
                    ) + self.mcts.gamma*goal['accumulated_reward']
                    goal["N"] = node.n
                    goal["depth"] += 1
                new_dict['HER_goals'] = selected_goals

                dicts.append(new_dict)
            else: 
                selected_goals = []
                for _ in range(n_goals):
                    appending_goal = copy.deepcopy(new_dict['raw_obs'])
                    appending_goal['accumulated_reward'] = 0
                    appending_goal['depth'] = 0
                    appending_goal["N"] = 1
                    selected_goals.append(appending_goal)

            s = new_dict['raw_obs']
            for g in selected_goals:
                assert type(g) == dict
                assert 'achieved_goal' in g.keys()
                assert 'N' in g.keys()
            return dicts, accumulated_her_dicts, s, selected_goals

        self.mcts.preprocess_return_results()

        dicts, her_dicts, _, _ = walk_tree(self.mcts.root_node)
        for hd in her_dicts:
            # hd['old_her_goals'] = hd['her_goals']
            hd['her_goals'] = np.stack([g['achieved_goal'] for g in hd['her_goals']], axis=0)
        # her_dicts['her_goals'] = np.stack()

        final_her_dict = {
            'obs': [],
            'obs_next': [],
            'actions': [],
            'ag': [],
            'ag_next': [],
            'g': [],
            't_remaining': [],
            'col': [],
            'her_goals': [],
            'vol': [],
        }
        for i in range(len(her_dicts)):
            for key in final_her_dict.keys():
                final_her_dict[key].append(her_dicts[i][key])
        
        for key in final_her_dict.keys():
            final_her_dict[key] = np.array(final_her_dict[key])

        result_dict = dicts[0]
        step_cutoff = 2
        def get_action_sequence(node, i=False):
            if i:
                import ipdb
                ipdb.set_trace()
            if node.n >= step_cutoff:
                action = max(node.child_actions, key=self.action_selection_criteria)                   
                if node.r >= 1.:
                    return []

                return [action.action] + get_action_sequence(action.child_node, i)
            else: 
                return []


        action_list = get_action_sequence(self.mcts.root_node)
        return action_list, dicts, final_her_dict

    # def act(  # type: ignore[override]
    #     self, Env: gym.Env,
    # ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #     #Correct but slow version
    #     self.mcts.epoch=self.epoch
    #     self.mcts.search(Env=Env)

    #     def random_choice(x): 
    #         n_sum = sum([i['N'] for i in x])
    #         indices = np.arange(len(x))
    #         p = np.array([i['N'] for i in x])/n_sum
    #         return np.random.choice(x, p=p)

    #     n_cutoff = 2
    #     n_goals = 4
    #     def walk_tree(node):
    #         accumulated_her_dicts = []
    #         dicts = []
    #         goals = [[] for _ in range(n_goals)]
    #         goal = {
    #             "achieved_goal": None,
    #             "accumulated_reward": 0, #Accumulated reward between goal being sampled and current state
    #             "N":1 #Number of times most recent ancestor was expanded
    #         }
    #         new_dict = self.mcts.return_results(self.final_selection, node)
    #         if node.n >= n_cutoff:
    #             her_dicts = []
    #             all_child_goals = []
    #             child_goal_list = []
    #             for next_action in node.child_actions:
    #                 child_dict, next_her_dicts, next_s, child_goals = walk_tree(next_action.child_node)
    #                 child_goal_list.append((next_action.child_node, child_goals))
    #                 dicts += child_dict
    #                 accumulated_her_dicts += next_her_dicts
    #                 all_child_goals += child_goals

    #                 her_dicts.append({
    #                     'obs': new_dict['raw_obs']['observation'],
    #                     'obs_next': next_s['observation'],
    #                     'actions': next_action.action,
    #                     'ag': new_dict['raw_obs']['achieved_goal'],
    #                     'ag_next': next_s['achieved_goal'],
    #                     'g': new_dict['raw_obs']['achieved_goal'],
    #                     'child_goals': child_goals,
    #                     't_remaining': [0],
    #                     'col': False,
    #                     'node_depth': node.depth,
    #                     'her_goals': []
    #                     # 'all_future_goals':[]
    #                 })

    #             new_goal = new_dict['raw_obs']
    #             new_goal["N"] = 1
    #             new_goal['accumulated_reward'] = 0
    #             new_goal['depth'] = 0

    #             goal_num = len(child_goals)
    #             selectable_goals = all_child_goals + [new_goal]

    #             selected_goals = []
    #             for i in range(n_goals):
    #                 goal = np.random.choice(selectable_goals)
    #                 selected_goals.append(goal)
    #                 for hd in her_dicts:
    #                     hd['her_goals'].append(goal)

    #             accumulated_her_dicts += her_dicts

    #             for goal in selected_goals:
    #                 goal['state'] = Env.observation(goal)
    #                 goal['accumulated_reward'] = Env.compute_reward(
    #                     new_dict['raw_obs']['achieved_goal'],
    #                     goal['achieved_goal'], None
    #                 ) + self.mcts.gamma*goal['accumulated_reward']
    #                 goal["N"] = node.n
    #                 goal["depth"] += 1
    #             new_dict['HER_goals'] = selected_goals

    #             dicts.append(new_dict)
    #         else: 
    #             selected_goals = []
    #             for _ in range(n_goals):
    #                 appending_goal = copy.deepcopy(new_dict['raw_obs'])
    #                 appending_goal['accumulated_reward'] = 0
    #                 appending_goal['depth'] = 1
    #                 appending_goal["N"] = 1
    #                 selected_goals.append(appending_goal)
    #             selectable_goals = [appending_goal]

    #         s = new_dict['raw_obs']
    #         for g in selected_goals:
    #             assert type(g) == dict
    #             assert 'achieved_goal' in g.keys()
    #             assert 'N' in g.keys()

    #         return dicts, accumulated_her_dicts, s, selectable_goals

    #     self.mcts.preprocess_return_results()

    #     dicts, her_dicts, _, _ = walk_tree(self.mcts.root_node)
    #     for hd in her_dicts:
    #         # hd['old_her_goals'] = hd['her_goals']
    #         hd['her_goals'] = np.stack([g['achieved_goal'] for g in hd['her_goals']], axis=0)

    #     final_her_dict = {
    #         'obs': [],
    #         'obs_next': [],
    #         'actions': [],
    #         'ag': [],
    #         'ag_next': [],
    #         'g': [],
    #         # 'child_goals': [],
    #         't_remaining': [],
    #         'col': [],
    #         # 'node_depth': [],
    #         'her_goals': []
    #         # 'all_future_goals':[]
    #     }
    #     for i in range(len(her_dicts)):
    #         for key in final_her_dict.keys():
    #             final_her_dict[key].append(her_dicts[i][key])

        
    #     for key in final_her_dict.keys():
    #         final_her_dict[key] = np.array(final_her_dict[key])

    #     result_dict = dicts[0]
    #     step_cutoff = 2
    #     def get_action_sequence(node, i=False):
    #         if i:
    #             import ipdb
    #             ipdb.set_trace()
    #         if node.n >= step_cutoff:
    #             action = max(node.child_actions, key=self.action_selection_criteria)                   
    #             if node.r >= 1.:
    #                 return []

    #             return [action.action] + get_action_sequence(action.child_node, i)
    #         else: 
    #             return []

    #     action_list = get_action_sequence(self.mcts.root_node)
    #     return action_list, dicts, final_her_dict


    def get_tensor_obs(self, obs): 
        tensor_obs  = super().get_tensor_obs(obs)
        # tensor_obs['epoch'] = torch.tensor(obs['epoch']).float().to(self.device)
        tensor_obs['epoch']     = self.epoch
        tensor_obs['gamma']     = self.gamma
        tensor_obs['traj_value']= torch.tensor(obs['traj_value']).unsqueeze(-1)
        tensor_obs['children_unweighted_density'] = [
            torch.tensor(v)
            for v in obs['children_unweighted_density']
        ]
        tensor_obs['base_prob'] = torch.tensor(obs['base_prob'])
        HER_observations = []
        HER_targets = []
        for goal in obs['HER_goals']:
            for o in goal:
                # HER_observations.append(torch.tensor(o['observation']))
                HER_observations.append(torch.tensor(o['state']))
                HER_targets.append(torch.tensor(o['accumulated_reward']))

        tensor_obs['HER_observations'] = torch.stack(HER_observations, dim=0).float()
        tensor_obs['HER_targets'] = torch.stack(HER_targets, dim=0).unsqueeze(dim=-1).float()

        return tensor_obs



class OffPolicyHERAgent(OneShotAgent2):
    # pass
    def __init__(
        self,
        policy_cfg: DictConfig,
        mcts_cfg: DictConfig,
        loss_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        final_selection: str,
        epsilon: float,
        train_epochs: int,
        grad_clip: float,
        device: str,
    ) -> None:
        super().__init__(
            policy_cfg=policy_cfg,
            loss_cfg=loss_cfg,
            mcts_cfg=mcts_cfg,
            optimizer_cfg=optimizer_cfg,
            final_selection=final_selection,
            epsilon=epsilon,
            train_epochs=train_epochs,
            grad_clip=grad_clip,
            device=device,
        )
        self.mcts_cfg = mcts_cfg
        self.epsilon = epsilon

    def set_network(self, env_name):
        import pickle
        import os
        directory = os.getcwd().split("outputs")[0]
        filename = directory + f"transfer folder/{env_name}.pkl"
        with open(filename, 'rb') as file:
            self.HER_model = pickle.load(file)
        # self.HER_model = self.nn
        
        print(f"we are using the HER agent at {filename}")
        self.mcts = hydra.utils.instantiate(self.mcts_cfg, model=self.HER_model)


class ContinuousOffPolicyHERAgent(ContinuousAgent):
    def __init__(
        self,
        policy_cfg: DictConfig,
        mcts_cfg: DictConfig,
        loss_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        final_selection: str,
        epsilon: float,
        train_epochs: int,
        grad_clip: float,
        device: str,
    ) -> None:
        super().__init__(
            policy_cfg=policy_cfg,
            loss_cfg=loss_cfg,
            mcts_cfg=mcts_cfg,
            optimizer_cfg=optimizer_cfg,
            final_selection=final_selection,
            epsilon=epsilon,
            train_epochs=train_epochs,
            grad_clip=grad_clip,
            device=device,
        )
        self.mcts_cfg = mcts_cfg

    def set_network(self, env_name):
        import pickle
        import os
        directory = os.getcwd().split("outputs")[0]
        filename = directory + f"transfer folder/{env_name}.pkl"
        with open(filename, 'rb') as file:
            self.HER_model = pickle.load(file)
            
        print(f"we are using the HER agent at {filename}")
        self.mcts = hydra.utils.instantiate(self.mcts_cfg, model=self.HER_model)

class ContinuousExternalTrainingHERAgent(ContinuousAgent):
    # def __init__(
    #     self,
    #     policy_cfg: DictConfig,
    #     mcts_cfg: DictConfig,
    #     loss_cfg: DictConfig,
    #     optimizer_cfg: DictConfig,
    #     final_selection: str,
    #     epsilon: float,
    #     train_epochs: int,
    #     grad_clip: float,
    #     device: str,
    # ) -> None:
    #     super().__init__(
    #         policy_cfg=policy_cfg,
    #         loss_cfg=loss_cfg,
    #         mcts_cfg=mcts_cfg,
    #         optimizer_cfg=optimizer_cfg,
    #         final_selection=final_selection,
    #         epsilon=epsilon,
    #         train_epochs=train_epochs,
    #         grad_clip=grad_clip,
    #         device=device,
    #     )

    #     self.device = torch.device(device)
    #     self.nn = hydra.utils.call(policy_cfg).to(torch.device(device))
    #     self.mcts = hydra.utils.instantiate(mcts_cfg, model=self.nn)
    #     self.loss = hydra.utils.instantiate(loss_cfg).to(self.device)
    #     self.optimizer_cfg = optimizer_cfg
    #     self.optimizer = hydra.utils.instantiate(
    #         optimizer_cfg, params=self.nn.parameters()
    #     )
    #     self.epsilon = epsilon

    #     self.final_selection = final_selection
    #     self.train_epochs = train_epochs
    #     self.clip = grad_clip


    #     del self.nn
    def set_network(self, new_network):
        # del self.mcts.model
        # self.HER_model = new_network
        self.nn = new_network
        self.mcts.model = new_network
        self.optimizer = hydra.utils.instantiate(
            self.optimizer_cfg, params=self.nn.parameters()
        )

class BetterBufferAgent(OneShotAgent2):    
    def train(self, buffer: ReplayBuffer) -> Dict[str, Any]:
        buffer.reshuffle()
        running_loss: Dict[str, Any] = defaultdict(float)
        for epoch in range(self.train_epochs*100):
            obs = buffer.sample(buffer.batch_size)
            loss = self.update(obs)
            for key in loss.keys():
                running_loss[key] += loss[key]

        for val in running_loss.values():
            val = val / (self.train_epochs)
        return running_loss
    def get_tensor_obs(self, obs):        
        tensor_obs = {}
        for key in obs.keys():
            tensor_obs[key] = torch.tensor(obs[key])
        tensor_obs['epoch']     = self.epoch
        tensor_obs['gamma']     = self.gamma
        tensor_obs['traj_value']= torch.tensor(obs['traj_value']).unsqueeze(-1)
        tensor_obs['advantage'] = tensor_obs['Qs'] - tensor_obs['values']
        
        return tensor_obs

    def update(
        self, obs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        # zero out gradients
        for param in self.nn.parameters():
            param.grad = None

        tensor_obs = self.get_tensor_obs(obs)
        train_data_dict = self.nn.get_train_data_generalized(tensor_obs)

        loss_dict = self.loss(tensor_obs=tensor_obs, train_data_dict=train_data_dict)

        loss_dict["loss"].backward()

        if self.clip:
            clip_grad_norm(self.nn.parameters(), self.clip)

        self.optimizer.step()

        try: 
            info_dict = {key: float(value) for key, value in loss_dict.items()}
        except: 
            import ipdb
            ipdb.set_trace()
        return info_dict