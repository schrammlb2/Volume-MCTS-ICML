import copy
import random
from typing import Any, Optional, Tuple, Union
import gym
import torch
import numpy as np
from abc import ABC, abstractmethod

from alphazero.search.states import (
    Action,
    Node,
    ActionContinuous,
    ActionDiscrete,
    NodeContinuous,
    NodeDiscrete,
)
from alphazero.search.kd_states import (
    KDNodeContinuous, 
    KDActionContinuous, 
    PWAction
)
from alphazero.helpers import argmax, Object

from alphazero.search.kd_tree import KDTreePolicyAlt, KDTreeValue
from alphazero.search.hoot_tree_2 import (
    HOOTNode, 
    HOOTAction,
    # HOOTTree
)
# from alphazero.policy_calculator import calculate_policy, calculate_policy_with_volume
from alphazero.policy_calculator import (
    calculate_policy, 
    calculate_policy_with_volume_2, 
    calculate_policy_with_volume_without_volume, 
    calculate_one_shot_policy
)
import ipdb
# scales the step reward between -1 and 0
PENDULUM_R_SCALE = 16.2736044
PENDULUM_R_SCALE = 1
# from copy import *
check_goal = False

def check_transition_diversity(node):
    if len(node.child_actions) <= 1:
        return False
    else: 
        for action_1 in node.child_actions:
            for action_2 in node.child_actions:
                if action_1 == action_2: 
                    continue
                action_distance = ((action_1.action - action_2.action)**2).sum()**0.5
                space_distance = ((action_1.child_node.state - action_2.child_node.state)**2).sum()**0.5
                diff = 1000
                if space_distance < action_distance/diff:
                    import ipdb
                    ipdb.set_trace()
                    return (action_1, action_2)
        return False



class MCTS(ABC):
    """Base MCTS class.

    The base MCTS class implements functionality that is common for either discrete or
    continuous action spaces. This is specifically:
        - The calculation of the value target.
        - Epsilon-greedy action selection.
        - Selection phase of the MCTS.
        - Expansion phase of the MCTS.
        - Backup phase of the MCTS.
        - Return of the final search results.

    The search itself, adding of the value targets and the selection phase must be implemented
    differently for discrete and continuous action spaces due to progressive widening.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_rollouts: int,
        c_uct: float,
        gamma: float,
        epsilon: float,
        device: str,
        V_target_policy: str,
        root_state: np.ndarray,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        model: torch.nn.Module
            Point to the network model used to guide the search.
        n_rollouts: int
            Number of search traces per execution of the MCTS search.
        c_uct: float
            UCT exploration term constant.
        gamma: float
            Discount factor for the backup phase of the MCTS.
        epsilon: float
            Epsilon constant for epsilon greedy action selection.
        device: str
            Device for the search execution. Can be either "cpu" or "gpu".
        V_target_policy: str
            Method for calculating the value targets. Can be either "on_policy", "off_policy" or "greedy".
        root_state: np.ndarray
            Environment state associated with the root node.
        """
        self.device = torch.device(device)
        self.root_node = None
        self.root_state = root_state
        self.model = model
        self.n_rollouts = n_rollouts
        self.c_uct = c_uct
        self.gamma = gamma
        self.epsilon = epsilon
        self.V_target_policy = V_target_policy

        self.stats = Object()
        self.stats.pretty_print = lambda : None
        self.backprop_mode = "standard"

    @abstractmethod
    def selectionUCT(self, node: Node) -> Action:
        """Interface for the selection method. Must be implemented differently for both action spaces"""
        ...

    @abstractmethod
    def search(self, Env: gym.Env) -> None:
        """Interface method for executing the search."""
        ...

    @staticmethod
    def get_on_policy_value_target(Q: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """Calculate the on-policy value target.

        The on-policy value target is the sum of the root action counts weighted by
        the visitation count distribution. The visitation count distribution is obtained
        by dividing the visitation counts with the total sum of counts at the root.

        Parameters
        ----------
        Q: np.ndarray
            Q values of the root node actions
        counts: np.ndarray
            Visitation counts of the root node actions.

        Returns
        -------
        np.ndarray
            Scalar numpy array containing the value target.
        """
        return np.sum((counts / np.sum(counts)) * Q)

    @staticmethod
    def get_off_policy_value_target(Q: np.ndarray) -> Any:
        """Calculate the off-policy value target.

        The off policy value target is the maximum action value at the root node.
        This is the value target proposed in the A0C paper:
        https://arxiv.org/pdf/1805.09613.pdf.

        Parameters
        ----------
        Q: np.ndarray
            Action values at the root node.

        Returns
        -------
        Any
            Scalar numpy array containing the value target.
        """
        return Q.max()

    def get_greedy_value_target(self, final_selection: str) -> Any:
        """Calculate the greedy value target.

        The greedy value target descends down the tree selecting the action with the
        highest action value/highest visitation count until a leaf node is reached.
        This action's Q-value is then returned as value target.
        More information about this value target can be found here:
        https://ala2020.vub.ac.be/papers/ALA2020_paper_18.pdf.

        Parameters
        ----------
        final_selection: str
            Final selection policy used in the search. Can be "max_value" or "max_visit".

        Returns
        -------
        Any
            Scalar value target.
        """
        assert self.root_node is not None
        node = self.root_node

        while node.terminal and node.has_children:
            if final_selection == "max_value":
                Q = np.array(
                    [child_action.Q for child_action in self.root_node.child_actions]
                )
                child = node.child_actions[Q.argmax()].child_node
            else:
                counts = np.array(
                    [child_action.n for child_action in node.child_actions]
                )
                child = node.child_actions[counts.argmax()].child_node

            if not child:
                break
            else:
                node = child

        Q = np.array([child_action.Q for child_action in node.child_actions])
        return Q.max()

    def epsilon_greedy(self, node: Node, UCT: np.ndarray) -> Action:
        """Implementation of epsilon greedy action selection.

        Parameters
        ----------
        node: Node
            Node from which a child action should be selected.
        UCT: np.ndarray
            UCT values of the child actions in the passed node.

        Returns
        -------
        Action
            Action selected by this algorithm.
        """
        if random.random() < self.epsilon:
            # return a random child if the epsilon greedy conditions are met
            return node.child_actions[random.randint(0, len(node.child_actions) - 1)]
        else:
            winner = argmax(UCT)
            return node.child_actions[winner]

    @staticmethod
    def selection(action: Action) -> Optional[Node]:
        """MCTS selection phase. Select the child node of the chosen action.

        When this method returns None, the search proceeds with the expansion stage.

        Parameters
        ----------
        action: Action
            Action whose child node should be selected.

        Returns
        -------
        Optional[Node]
            Child node of the action.
        """
        return action.child_node

    @staticmethod
    def expansion(
        action: Action, state: np.ndarray, reward: float, terminal: bool
    ) -> Node:
        """Expand the tree by adding a new child node given an action.

        Parameters
        ----------
        action: Action
            Action that leads to the newly created node.
        state: np.ndarray
            Environment state of the new node.
        reward: float
            Reward obtained from the environment transition that leads to the new node.
        terminal: bool
            Flag indicating whether the new state is terminal or not.

        Returns
        -------
        Node
            Node wrapping the passed environment state.
        """
        node = action.add_child_node(state, reward, terminal)
        # problematic = check_transition_diversity(action.parent_node)
        # if problematic != False:
        #     import ipdb 
        #     ipdb.set_trace()
        return node

    @staticmethod
    def backprop(node: Node, gamma: float) -> None:
        """Implementation of the MCTS backup phase.

        Starting from the passed node, the backup loop goes back up the tree until the
        root node is reached. For each node along its path it updates:
            - The cumulative discounted reward of the node.
            - The visitation count node.
        For each action along the path:
            - Update of the cumulative discounted reward.
            - Update of the visitation count.
            - Update of the action value (cumulative reward/ visitation count)

        Parameters
        ----------
        node: Node
            Leaf node of the algorithm that has been evaluated by the neural network.
        gamma: float
            Discount factor.
        """
        R = node.V
        # loop back-up until root is reached
        while node.parent_action is not None:
            R = node.r + gamma * R
            action = node.parent_action
            action.update(R)
            if action.Q.shape != ():
                import ipdb
                ipdb.set_trace()
            assert action.Q.shape == ()
            node = action.parent_node
            node.update_visit_counts()



    def backprop(self, node: Node, gamma: float) -> None:
        if "kernel" in self.backprop_mode and False:
            R = node.V
            # loop back-up until root is reached
            while node.parent_action is not None:
                if node.has_children and "greedy" in self.backprop_mode:
                    optimistic_R = max([
                        action.Q if not hasattr(action, "base_Q") else action.base_Q
                        for action in node.child_actions
                    ])
                else: 
                    optimistic_R = R
                R = node.r + gamma * R
                if "very_greedy" in self.backprop_mode:
                    R = max(1/(1-gamma)*node.r, R)
                action = node.parent_action
                action.update(R)
                if "very_greedy" in self.backprop_mode:
                    node.V = R
                if node.has_children and "greedy" in self.backprop_mode:
                    action.base_Q = R
                    # value = self.kd_tree.kernel_regression(node.state.tolist(), lambda x: x.V)
                    value = self.kd_tree.best_nearby(node.state.tolist(), lambda x: x.V)
                    # value = self.kd_tree.knn_regression(node.state.tolist(), lambda x: x.V)
                    action.Q = max(value, R)
                    # action.Q = R
                node = action.parent_node    

                node.update_visit_counts()
        else:
            R = node.V
            # loop back-up until root is reached
            while node.parent_action is not None:
                if node.has_children and "greedy" in self.backprop_mode:
                    optimistic_R = max([action.Q for action in node.child_actions])
                else: 
                    optimistic_R = R
                    
                R = node.r + gamma * R
                if "very_greedy" in self.backprop_mode:
                    R = max(1/(1-gamma)*node.r, R)
                action = node.parent_action
                action.update(R)
                if node.has_children and "greedy" in self.backprop_mode:
                    action.Q = R
                # if "very_greedy" in self.backprop_mode:
                node.V = action.Q
                node = action.parent_node    

                node.update_visit_counts()

    def return_results(
        self, final_selection: str, node = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if node == None: 
            node = self.root_node
        assert node is not None
        """Returns the root node statistics once an MCTS stage has been completed.

        Parameters
        ----------
        final_selection: str
            Policy according to which the final action should be chosen. Needed for the
            calculation of the greedy value targets.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple containing the:
                - Root node state.
                - Actions selected at the root node.
                - Root node visitation counts.
                - Root node action values.
                - Root node value targets.
        """
        assert node is not None
        actions = np.array(
            [child_action.action for child_action in node.child_actions]
        )
        counts = np.array(
            [child_action.n for child_action in node.child_actions]
        )

        Q = np.array([child_action.Q for child_action in node.child_actions])

        if self.V_target_policy == "greedy":
            V_target = self.get_greedy_value_target(final_selection)
        elif self.V_target_policy == "on_policy":
            V_target = self.get_on_policy_value_target(Q, counts)
        else:
            V_target = self.get_off_policy_value_target(Q)

        return node.state, actions.squeeze(), counts, Q, V_target

    def select_action(self, node):
        actions = node.child_actions
        Qs = np.array([child_action.Q for child_action in node.child_actions])
        action_node = actions[Qs.argmax()]
        return action_node

    def getPath(self, node = None):
        return None
        if node == None:
            node = self.root_node
        assert node is not None

        if len(node.child_actions) == 0:
            return [node.state], []
        else:
            action_node = self.select_action(node)
            if not hasattr(action_node, 'child_node') or action_node.child_node == None:
                return [], []
            else: 
                state_sequence, action_sequence = self.getPath(node = action_node.child_node)
                new_state_sequence  = [node.state.tolist()] + state_sequence
                if self.Env.geometric: 
                    new_action_sequence = [self.Env.map_action(node.state, action_node.action).tolist()] + action_sequence
                else:
                    new_action_sequence = [action_node.action.tolist()] + action_sequence
                return new_state_sequence, new_action_sequence

    def getBestPath(self, whatever):
        return self.getPath(node = self.root_node)

    def getRoadmap(self, use_objects=False):
        """Returns a graph (V,E) where V contains states and E contains
        triples (i,j,u) where control u connnects V[i] to V[j]"""
        V = []
        E = []
        n = self.root_node
        if n is None:
            return (V,E)
        if use_objects:
            V.append(n)
        else: 
            V.append(n.state.tolist())
        q = [(n,0)]
        while len(q) > 0:
            n,i = q.pop()
            for action_node in n.child_actions:
                j = len(V)
                # E.append((i,j,action_node.action.tolist()))
                if hasattr(action_node, 'child_node') and action_node.child_node != None:
                    if self.Env.geometric: 
                        # action = self.Env.map_action(n.state, action_node.action)
                        # action = self.Env.map_action(n.state, action_node.action)
                        action = action_node.child_node.state
                    else:
                        action = action_node.action
                    E.append((i,j,action.tolist()))
                    if use_objects:
                        V.append(action_node.child_node)
                    else: 
                        V.append(action_node.child_node.state)
                    q.append((action_node.child_node,j))
        return V, E


class MCTSDiscrete(MCTS):
    """Implementation of the MCTS algorithm for discrete action spaces.

    Assumes that the number of actions in each state is finite and does not change over time.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_actions: int,
        n_rollouts: int,
        c_uct: float,
        gamma: float,
        epsilon: float,
        V_target_policy: str,
        device: str,
        root_state: np.ndarray,
    ):
        """Constructor.

        Parameters
        ----------
        model: torch.nn.Module
            Point to the network model used to guide the search.
        num_actions: int
            Number of available actions in each environment state.
        n_rollouts: int
            Number of search traces per execution of the MCTS search.
        c_uct: float
            UCT exploration term constant.
        gamma: float
            Discount factor for the backup phase of the MCTS.
        epsilon: float
            Epsilon constant for epsilon greedy action selection.
        V_target_policy: str
            Method for calculating the value targets. Can be either "on_policy", "off_policy" or "greedy".
        device: str
            Device for the search execution. Can be either "cpu" or "gpu".
        root_state: np.ndarray
            Environment state associated with the root node.
        """
        super().__init__(
            model=model,
            n_rollouts=n_rollouts,
            c_uct=c_uct,
            gamma=gamma,
            epsilon=epsilon,
            V_target_policy=V_target_policy,
            device=device,
            root_state=root_state,
        )

        self.num_actions = num_actions

    def initialize_search(self) -> None:
        """Initialize the search at the root node. This includes concretely:
            - Construction of a Node object from the environment state passed to this class.
            - Check whether the root node is terminal or not.

        In case the tree is reused, no new root is constructed.
        """
        if self.root_node is None:
            self.root_node = NodeDiscrete(  # type: ignore[assignment]
                self.root_state,
                r=0.0,
                terminal=False,
                parent_action=None,
                num_actions=self.num_actions,
            )
        else:
            # continue from current root
            self.root_node.parent_action = None
        if self.root_node.terminal:  # type: ignore[attr-defined]
            raise ValueError("Can't do tree search from a terminal node")

    def evaluation(self, node: NodeDiscrete) -> None:
        """Use the neural network to evaluate a node.

        Evaluation of a node consists of adding a value estimate and prior probabilities
        for all available actions at the node.

        Parameters
        ----------
        node: NodeDiscrete
            Node to be evaluated.
        """
        state = (
            torch.from_numpy(
                node.state[
                    None,
                ]
            )
            .float()
            .to(self.device)
        )

        node.V = (
            # (self.model.predict_V(state)).item()  # type:ignore[operator]
            np.squeeze(self.model.predict_V(state))  # type: ignore[operator]
            # (self.model.predict_V(state)).reshape([1])  # type: ignore[operator]
            if not node.terminal
            else np.array(0.0) #0.0
        )
        if node.V.shape != ():
            import ipdb
            ipdb.set_trace()
        assert node.V.shape == ()

        node.child_actions = [
            ActionDiscrete(a, parent_node=node, Q_init=node.V)
            for a in range(node.num_actions)
        ]
        node.priors = self.model.predict_pi(state).flatten()  # type:ignore[operator]

    def search(self, Env: gym.Env) -> None:
        """Execute the MCTS search.

        The MCTS algorithm relies on knowing the environment's transition dynamics to
        perform rollouts. In the case of OpenAI gym this means that the environment has to
        be passed. It is then copied in this method before each search trace. A total
        of n_rollouts searches are executed.

        Parameters
        ----------
        Env: gym.Env
            OpenAI gym environment
        """

        self.initialize_search()

        assert self.root_node is not None

        # add network estimates to the root node
        self.evaluation(self.root_node)
        for i in range(self.n_rollouts):
            # reset to root for new trace
            node = self.root_node

            # copy original Env to rollout from
            mcts_env = copy.deepcopy(Env)
            # self.env = mcts_env

            while not node.terminal:
                action = self.selectionUCT(node)
                # action = np.squeeze(self.selectionUCT(node))

                # take step
                new_state, reward, terminal, _ = mcts_env.step(action.action)
                # self.env = mcts_env
                if hasattr(action, "child_node"):
                    # selection
                    node = self.selection(action)
                    continue
                else:
                    # expansion
                    node = self.expansion(action, new_state, reward, terminal)

                    # Evaluate node -> Add distribution and value estimate
                    self.evaluation(node)
                    break

            self.backprop(node, self.gamma)

    def selectionUCT(self, node: NodeDiscrete) -> Action:  # type: ignore[override]
        """UCT selection method for discrete action spaces.

        Calculates the UCT value for all actions of a node. If enabled, epsilon-greedy
        action selection is performed.

        Parameters
        ----------
        node: NodeDiscrete
            Node for which a child action should be selected.

        Returns
        -------
        Action
            Best action according to UCT or epsilon-greedy.
        """
        assert node.priors is not None
        UCT = np.array(
            [
                child_action.Q
                + prior * self.c_uct * (np.sqrt(node.n + 1) / (child_action.n + 1))
                for child_action, prior in zip(node.child_actions, node.priors)
            ]
        )
        if self.epsilon == 0:
            # do standard UCT action selection if epsilon=0
            winner = argmax(UCT)
            return node.child_actions[winner]
        else:
            return self.epsilon_greedy(node=node, UCT=UCT)

    def forward(self, action: int, state: np.ndarray) -> None:
        """Moves the root node forward.

        This method implements tree reuse. The action selected in the environment leads to
        a state which is the new root node of the tree. Through this the search tree is
        preserved.

        Parameters
        ----------
        action: int
            Action selected in the environment.
        state: np.ndarray
            State obtained by selecting the passed action.
        """
        assert self.root_node is not None
        if not hasattr(self.root_node.child_actions[action], "child_node"):
            self.root_node = None
            self.root_state = state
        elif (
            np.linalg.norm(
                self.root_node.child_actions[action].child_node.state - state
            )
            > 0.01
        ):
            print(
                "Warning: this domain seems stochastic. Not re-using the subtree for next search. "
                + "To deal with stochastic environments, implement progressive widening."
            )
            self.root_node = None
            self.root_state = state
        else:
            self.root_node = self.root_node.child_actions[action].child_node


class MCTSContinuous(MCTS):
    """Continuous MCTS implementation.

    This class uses progressive widening to deal with continuous action spaces.
    More information about progressive widening can be found here:
    https://hal.archives-ouvertes.fr/hal-00542673v2/document
    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_rollouts: int,
        c_uct: float,
        c_pw: float,
        kappa: float,
        gamma: float,
        epsilon: float,
        V_target_policy: str,
        device: str,
        root_state: np.ndarray,
    ):
        """Constructor.

        Parameters
        ----------
        model: torch.nn.Module
            Point to the network model used to guide the search.
        n_rollouts: int
            Number of search traces per execution of the MCTS search.
        c_uct: float
            UCT exploration term constant.
        c_pw: float
            Progressive widening factor.
        kappa: float
            Progressive widening exponent.
        gamma: float
            Discount factor for the backup phase of the MCTS.
        epsilon: float
            Epsilon constant for epsilon greedy action selection.
        V_target_policy: str
            Method for calculating the value targets. Can be either "on_policy", "off_policy" or "greedy".
        device: str
            Device for the search execution. Can be either "cpu" or "gpu".
        root_state: np.ndarray
            Environment state associated with the root node.
        """
        super().__init__(
            model=model,
            n_rollouts=n_rollouts,
            c_uct=c_uct,
            gamma=gamma,
            epsilon=epsilon,
            V_target_policy=V_target_policy,
            device=device,
            root_state=root_state,
        )

        self.c_pw = c_pw
        self.kappa = kappa
        self.backprop_mode="greedy"

    def initialize_search(self) -> None:
        """Initialize the search at the root node. This includes concretely:
            - Construction of a Node object from the environment state passed to this class.
            - Check whether the root node is terminal or not.

        Note that tree reuse is not possible for continuous domains.
        """
        self.root_node = NodeContinuous(  # type: ignore[assignment]
            self.root_state, r=0.0, terminal=False, parent_action=None
        )
        if self.root_node.terminal:  # type: ignore[attr-defined]
            raise ValueError("Can't do tree search from a terminal node")
        self.global_n = 1

    def add_value_estimate(self, node: NodeContinuous) -> None:
        """Adds a neural network value estimate to the passed node.

        Parameters
        ----------
        node: NodeContinuous
            Node which should be evaluated.
        """
        state = (
            torch.from_numpy(
                node.state[None,]
            )
            .float()
            .to(self.device)
        )
        node.V = (
            # (self.model.predict_V(state)).item()
            np.squeeze(self.model.predict_V(state))  # type: ignore[operator]
            # (self.model.predict_V(state)).reshape([1])  # type: ignore[operator]
            if not node.terminal
            else np.array(1/(1-self.gamma)*node.r) #np.array(0.0)
        )
        # if "stationary" in self.backprop_mode:
        #     node.V = max(1/(1-self.gamma)*node.r + node.V*0, node.V)

        # if "kernel" in self.backprop_mode:                
        #     value = self.kd_tree.best_nearby(node.state.tolist(), lambda x: x.V)
        #     alpha = 0.5
        #     node.V = (1-alpha)*node.V + alpha*value
        if type(node.V) is torch.Tensor: 
            node.V = node.V.numpy()

        if node.V.shape != ():
            import ipdb
            ipdb.set_trace()

        assert node.V.shape == ()

    def add_pw_action(self, node: NodeContinuous) -> None:
        """Adds a new action to the passed node.

        This method uses the network the evaluate the node and produce a distribution
        over the action space. From this distribution a new action is sampled and appended
        to the node.
        Note 1: It is inefficient to evaluate the network each time a new action should be sampled.
            It is faster to add the distribution to the node and sample from it multiple times.
            This works as the generated distribution is deterministic given a state. The version
            implemented in this class is slower. Since the used neural networks are small it does
            not matter too much.
        Note 2: This method does not check whether the criterion for progressive widening is met.

        Parameters
        ----------
        node: NodeContinuous
            Node to which a new action should be added.
        """
        state = (
            torch.from_numpy(
                node.state[
                    None,
                ]
            )
            .float()
            .to(self.device)
        )
        action = self.model.sample_action(state)  # type: ignore[operator]
        action = action.reshape(-1)#.squeeze()
        new_child = ActionContinuous(action, parent_node=node, Q_init=node.V)
        node.child_actions.append(new_child)


    # def planMore(self, n):
    #     Env = self.Env
    #     # mcts_env = copy.deepcopy(Env)
    #     # save_state = hasattr(mcts_env, "get_state") and hasattr(mcts_env, "set_state")
    #     # if save_state:
    #     #     self.root_node.saved_state = mcts_env.get_state()
    #     save_state = True
    #     # save_state = False
    #     if save_state:
    #         # self.root_node.environment = copy.deepcopy(Env)
    #         self.root_node.saved_state = Env.get_save_state()
    #         mcts_env = Env

    #     for i in range(n):
    #         # reset to root for new trace
    #         node = self.root_node
    #         self.global_n += 1

    #         # copy original Env to rollout from
    #         # if save_state:
    #         #     pass
    #         # else: 
    #         #     mcts_env = copy.deepcopy(Env)
    #         if save_state:
    #             pass
    #         else:
    #             mcts_env = copy.deepcopy(Env)

    #         self.depth = 0
    #         while not node.terminal:
    #             action = self.selectionUCT(node)
    #             self.depth += 1

    #             # take step
    #             # ipdb.set_trace()
    #             if save_state:
    #                 pass
    #             else:
    #                 new_state, reward, terminal, _ = mcts_env.step(action.action)
    #                 reward /= PENDULUM_R_SCALE

    #             if hasattr(action, "child_node"):
    #                 # selection
    #                 node = self.selection(action)
    #                 continue
    #             else:
    #                 if save_state:
    #                     mcts_env.restore_save_state(node.saved_state)
    #                     new_state, reward, terminal, _ = mcts_env.step(action.action)
    #                     reward /= PENDULUM_R_SCALE
    #                 else:
    #                     pass

    #                 node = self.expansion(
    #                     action, np.squeeze(new_state), reward, terminal
    #                 )

    #                 if save_state:
    #                     node.saved_state = mcts_env.get_save_state()
    #                 else:
    #                     pass


    #                 # if save_state:
    #                 #     node.saved_state = mcts_env.get_state()

    #                 self.add_value_estimate(node)
    #                 break

    #         self.backprop(node, self.gamma)


    def planMore(self, n):
        Env = self.Env
        save_state = True
        # save_state = False
        if save_state:
            self.root_node.saved_state = Env.get_save_state()
            # mcts_env = Env


        for i in range(n):
            # reset to root for new trace
            node = self.root_node
            self.global_n += 1

            if save_state:
                mcts_env = copy.deepcopy(Env)
                if check_goal: assert mcts_env.goal is not None
            else:
                mcts_env = copy.deepcopy(Env)
                if check_goal: assert mcts_env.goal is not None

            self.depth = 0
            terminal = False
            while not node.terminal and not terminal:
                action = self.selectionUCT(node)
                self.depth += 1
                if save_state:
                    pass
                else:
                    new_state, reward, terminal, _ = mcts_env.step(action.action)
                    # reward /= PENDULUM_R_SCALE
                # else:
                #     new_state = node.state
                #     reward = node.r

                if hasattr(action, "child_node"):
                    node = self.selection(action)
                    continue
                elif type(action) == type(None):
                    continue
                else:
                    if save_state:
                        mcts_env.restore_save_state(node.saved_state)
                        new_state, reward, terminal, _ = mcts_env.step(action.action)
                        # reward /= PENDULUM_R_SCALE
                    else:
                        pass
                    node = self.expansion(
                        action, np.squeeze(new_state), reward, terminal
                    )
                    if save_state:
                        node.saved_state = mcts_env.get_save_state()
                    else:
                        pass
                    self.add_value_estimate(node)
                    break

            self.backprop(node, self.gamma)

        if save_state:
            mcts_env.restore_save_state(self.root_node.saved_state)

    def search(self, Env: gym.Env) -> None:
        """Execute the MCTS search.

        The MCTS algorithm relies on knowing the environment's transition dynamics to
        perform rollouts. In the case of OpenAI gym this means that the environment has to
        be passed. It is then copied in this method before each search trace. A total
        of n_rollouts searches are executed.

        Parameters
        ----------
        Env: gym.Env
            OpenAI gym environment
        """

        self.initialize_search()
        assert self.root_node is not None
        self.add_value_estimate(self.root_node)
        self.add_pw_action(self.root_node)
        self.Env = Env
        self.planMore(self.n_rollouts)


    def ucb(self, parent_node: NodeContinuous, action_node: ActionContinuous):
        # try: 
        #     action_node.child_node
        # except: 
        #     ipdb.set_trace()
        return action_node.Q + self.c_uct * (np.sqrt(parent_node.n + 1) / (action_node.n + 1))

    def selectionUCT(self, node: NodeContinuous) -> Action:  # type: ignore[return-value, override]
        """UCT selection method for continuous action spaces.

        The method first checks for progressive widening. If widening is performed, no
        UCT is needed and the newly generated action is returned.
        Otherwise we proceed with regular UCT selection and use epsilon-greedy if enabled

        Parameters
        ----------
        node: NodeContinuous
            Node for which an action should be selected.

        Returns
        -------
        Action
            Action that is either:
                - Newly generated through progressive widening.
                - The action with the highest UCT value.
                - A random action from epsilon-greedy if enabled.
        """
        # no epsilon greedy if we add a node with progressive widening
        if node.check_pw(self.c_pw, self.kappa):
            self.add_pw_action(node)
            return node.child_actions[-1]
        else:
            # UCT = np.array(
            #     [
            #         child_action.Q
            #         + self.c_uct * (np.sqrt(node.n + 1) / (child_action.n + 1))
            #         for child_action in node.child_actions
            #     ]
            # )
            UCT = np.array(
                [ 
                    self.ucb(node, child_action) 
                    for child_action in node.child_actions
                ]
            )
            if self.epsilon == 0:
                # do standard UCT action selection if epsilon=0
                winner = argmax(UCT)
                return node.child_actions[winner]
            else:
                return self.epsilon_greedy(node=node, UCT=UCT)



class HOOT(MCTSContinuous):
    """Continuous MCTS implementation.

    This class uses progressive widening to deal with continuous action spaces.
    More information about progressive widening can be found here:
    https://hal.archives-ouvertes.fr/hal-00542673v2/document
    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_rollouts: int,
        c_uct: float,
        c_pw: float,
        kappa: float,
        gamma: float,
        epsilon: float,
        V_target_policy: str,
        device: str,
        root_state: np.ndarray,
    ):
        super().__init__(
            model=model,
            n_rollouts=n_rollouts,
            c_uct=c_uct,
            c_pw=c_pw,
            kappa=kappa,
            gamma=gamma,
            epsilon=epsilon,
            V_target_policy=V_target_policy,
            device=device,
            root_state=root_state,
        )

        # lo = [-model.action_bound for _ in range(model.action_dim)]
        # hi =  [model.action_bound for _ in range(model.action_dim)]
        # self.hoot_tree = HOOTTree(self, lo, hi)

    # def initialize_search(self) -> None:
    #     raise NotImplemented
    # #     self.root_node = HOOTNode(  # type: ignore[assignment]
    # #         self.root_state, r=0.0, terminal=False, parent_action=None, 
    # #         lo = [-self.model.action_bound for _ in range(self.model.action_dim)],
    # #         hi =  [self.model.action_bound for _ in range(self.model.action_dim)],
    # #         device = self.device
    # #     )
    # #     if self.root_node.terminal:  # type: ignore[attr-defined]
    # #         raise ValueError("Can't do tree search from a terminal node")
    # #     self.global_n = 1
    
    def initialize_search(self) -> None:
        self.root_node = HOOTNode(
            state=self.root_state,  r=0,
            terminal=False, parent_action=None,
            dim=self.model.action_dim, 
            device = self.device,
        )
        if self.root_node.terminal:  # type: ignore[attr-defined]
            raise ValueError("Can't do tree search from a terminal node")
        self.global_n = 1


    def search(self, Env: gym.Env) -> None:
        # raise NotImplemented
        self.initialize_search()
        assert self.root_node is not None
        self.add_value_estimate(self.root_node)
        self.selectionUCT(self.root_node)
        # action = self.root_node.hoot_tree.select(allow_expansion=True)
        self.Env = Env
        self.planMore(self.n_rollouts)

    # def selection(self, node, depth):
    #     if self.is_done or depth >= MAX_MCTS_DEPTH:
    #         return 0
    #     raw_action = node.hoo.select_action().tolist()
    #     action = [round(a, KEY_DECIMAL) for a in raw_action]
    #     if tuple(action) in node.children:
    #         child = node.children[tuple(action)]
    #         immediate_reward = child.immediate_reward
    #         value = child.selection(depth + 1)
    #         node.hoo.update(value + immediate_reward)
    #         return immediate_reward + value
    #     else:
    #         # create action here

    #         # snapshot, obs, immediate_reward, is_done, _ = env.get_result(self.snapshot, action)
    #         # child = HOOTNode(snapshot, obs, is_done, self, depth + 1, self.dim)
    #         # child.immediate_reward = immediate_reward
    #         # self.children[tuple(action)] = child 
    #         # value = child.selection(depth + 1)
    #         # self.hoo.update(value + immediate_reward)
    #         # return immediate_reward + value

    #         snapshot, obs, immediate_reward, is_done, _ = env.get_result(self.snapshot, action)
    #         child_action = HOOTAction(action, parent_node=self, Q_init=???)
    #         child = HOOTNode(snapshot, obs, is_done, self, depth + 1, self.dim)
    #         child.immediate_reward = immediate_reward
    #         self.children[tuple(action)] = child 
    #         value = child.selection(depth + 1)
    #         self.hoo.update(value + immediate_reward)
    #         return immediate_reward + value

    def selectionUCT(self, node: HOOTNode) -> Action:
        KEY_DECIMAL = 4
        raw_action = node.hoo.select_action().tolist()
        action = [round(a, KEY_DECIMAL) for a in raw_action]
        if tuple(action) in node.children:
            return node.children[tuple(action)]
        else:
            # create action here
            child_action = HOOTAction(np.array(action), parent_node=node, Q_init=node.V)
            node.children[tuple(action)] = child_action
            node.child_actions = list(node.children.values())
            return child_action
    #     return node.select(allow_expansion=True)

    @staticmethod
    def expansion(
        action: Action, state: np.ndarray, reward: float, terminal: bool
    ) -> Node:
        # raise NotImplemented
        node = action.add_child_node(state, reward, terminal)
        return node

    @staticmethod
    def backprop(node: Node, gamma: float) -> None:
        # raise NotImplemented
        original_node = node
        R = node.V
        node.hoo.update(R)
        # loop back-up until root is reached
        while node.parent_action is not None:
            try: 
                R = node.r + gamma * R
            except: 
                import ipdb
                ipdb.set_trace()
            action = node.parent_action
            action.update(R)
            # node.hoo.update(R)
            if action.Q.shape != ():
                import ipdb
                ipdb.set_trace()
            assert action.Q.shape == ()
            node = action.parent_node
            node.hoo.update(R)
            node.update_visit_counts()

    def return_results(
        self, final_selection: str, node = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # raise NotImplemented
        if node == None: 
            node = self.root_node
        assert node is not None
        actions = np.array(
            [child_action.action for child_action in node.child_actions]
        )
        counts = np.array(
            [child_action.n for child_action in node.child_actions]
        )

        Q = np.array([child_action.Q for child_action in node.child_actions])

        if self.V_target_policy == "greedy":
            V_target = self.get_greedy_value_target(final_selection)
        elif self.V_target_policy == "on_policy":
            V_target = self.get_on_policy_value_target(Q, counts)
        else:
            V_target = self.get_off_policy_value_target(Q)

        return node.state, actions.squeeze(), counts, Q, V_target

    def select_action(self, node):
        actions = node.child_actions
        Qs = np.array([child_action.Q for child_action in node.child_actions])
        action_node = actions[Qs.argmax()]
        return action_node


    def add_value_estimate(self, node: NodeContinuous) -> None:
        state = (torch.from_numpy(node.state[None,]).float().to(self.device))

        node.V = (
            np.squeeze(self.model.predict_V(state))  
            if not node.terminal
            else np.array(0.0)
        )

        if type(node.V) is torch.Tensor: 
            node.V = node.V.numpy()

        if node.V.shape != ():
            import ipdb
            ipdb.set_trace()

        assert node.V.shape == ()

class MCTSNGU(MCTSContinuous):   

    def expansion(self,
        action: Action, state: np.ndarray, reward: float, terminal: bool
    ) -> Node:
        node = action.add_child_node(state, reward, terminal)
        node.exploration_bonus = self.kd_tree.exploration_bonus(state.tolist())
        action.exploration_bonus = node.exploration_bonus
        return node


    def ucb(self, parent_node: NodeContinuous, action_node: ActionContinuous):
        if not hasattr(action_node, "child_node"):
            exploration_bonus = 10
        else: 
            exploration_bonus = action_node.exploration_bonus

        return action_node.Q + self.c_uct * exploration_bonus/self.epoch + self.c_uct * (np.sqrt(parent_node.n + 1) / (action_node.n + 1))
        # return action_node.Q + 0.1 * self.c_uct * exploration_bonus + self.c_uct * (np.sqrt(parent_node.n + 1) / (action_node.n + 1))

class MCTSRPO(MCTSContinuous):
    # def selectionUCT(self, node: NodeContinuous) -> Action:  # type: ignore[return-value, override]
    #     if node.check_pw(self.c_pw, self.kappa):
    #         self.add_pw_action(node)
    #         return node.child_actions[-1]
    #     else:
    #         lam = self.c_uct*(node.n+1)**(-1/2)
    #         policy = calculate_policy(node.child_actions, lam)
    #         try:
    #             action = np.random.choice(node.child_actions, p=policy)
    #         except:
    #             import ipdb
    #             ipdb.set_trace()
    #         return action


    def return_results(
        self, final_selection: str, node=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if node == None: 
            node = self.root_node
        assert node is not None
        # assert self.root_node is not None
        actions = np.array(
            [child_action.action for child_action in node.child_actions]
        )
        lam = self.c_uct*(node.n+1)**(-1/2)
        policy = calculate_policy(self.root_node.child_actions, lam)
        counts = policy*(node.n+1)
        # counts = np.array(
        #     [child_action.n for child_action in self.root_node.child_actions]
        # )

        Q = np.array([child_action.Q for child_action in node.child_actions])
        # V_target = np.sum(Q*policy)

        if self.V_target_policy == "greedy":
            V_target = self.get_greedy_value_target(final_selection)
        elif self.V_target_policy == "on_policy":
            V_target = self.get_on_policy_value_target(Q, counts)
        elif self.V_target_policy == "off_policy":
            V_target = self.get_off_policy_value_target(Q)
        elif self.V_target_policy == "rpo":
            V_target = np.sum(Q*policy)
        else: 
            import ipdb
            ipdb.set_trace()

        return node.state, actions.squeeze(), counts, Q, V_target

    # def backprop(self, node: Node, gamma: float) -> None:
    #     R = node.V
    #     # loop back-up until root is reached
    #     while node.parent_action is not None:
    #         R = node.r + gamma * R
    #         action = node.parent_action
    #         action.update(R)
    #         if len(node.child_actions) > 1:
    #             policy = calculate_policy(node.child_actions, node.n).reshape(-1)
    #             new_V = sum((
    #                 policy[i]*(node.child_actions[i].Q)
    #                 for i in range(len(node.child_actions))
    #             ))
    #             node.policy_V = new_V
    #             action.Q = node.r + gamma*new_V
    #         else: 
    #             node.policy_V = node.V
    #         node = action.parent_node    

    #         node.update_visit_counts()


class MCTSRPOAdvantage(MCTSRPO):
    pass
    def return_results(
        self, final_selection: str, node=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if node == None: 
            node = self.root_node
        assert node is not None

        # actions = np.array(
        #     [child_action.action for child_action in node.child_actions]
        # )
        actions = np.stack(
            [child_action.action for child_action in node.child_actions]
        )

        # if not len(actions.shape) == 3:
        #     import ipdb
        #     ipdb.set_trace()

        counts = np.array(
            [child_action.n for child_action in node.child_actions]
        )

        Q = np.array([child_action.Q for child_action in node.child_actions])

        if self.V_target_policy == "greedy":
            V_target = self.get_greedy_value_target(final_selection)
        elif self.V_target_policy == "on_policy":
            V_target = self.get_on_policy_value_target(Q, counts)
        else:
            V_target = self.get_off_policy_value_target(Q)

        state = (
            torch.from_numpy(
                node.state
            )
            .float()
            .to(self.device)
        )
        policy_output = self.model(state)
        # ipdb.set_trace()
        policy_dict = {
            'mu': policy_output[0].detach(),
            'sigma': policy_output[1].detach(),
            # 'log_coeff': policy_output[2].detach(),
            'V_hat': policy_output[-1].detach(),
        }
        policy_dict['state'] = node.state
        # policy_dict['actions'] = actions.squeeze()
        # policy_dict['actions'] = actions.squeeze()
        # policy_dict['actions'] = np.reshape(actions, (-1,))
        policy_dict['actions'] = np.reshape(actions, (-1,self.model.action_dim))
        # if len(policy_dict['actions'].shape) == 0:
        #     policy_dict['actions'] = policy_dict['actions'].unsqueeze()
        policy_dict['lambda'] = node.n**(-0.5)#lam
        policy_dict['counts'] = counts
        policy_dict['Qs'] = Q
        policy_dict['V_target'] = V_target
        policy_dict['n'] = node.n
        return policy_dict



    # pass

class MCTS_Volume_Intermediate(MCTSRPOAdvantage):
    #Implements all the requirements for Volume- and State-based planning, but does not change behavior
    def __init__(
        self,
        model: torch.nn.Module,
        n_rollouts: int,
        c_uct: float,
        c_pw: float,
        kappa: float,
        gamma: float,
        epsilon: float,
        V_target_policy: str,
        device: str,
        root_state: np.ndarray,
        # bounds: list,#[np.ndarray],
        observation_bounds: list,
    ):
        super().__init__(
            model = model,
            n_rollouts = n_rollouts,
            c_uct = c_uct,
            c_pw = c_pw,
            kappa = kappa,
            gamma = gamma,
            epsilon = epsilon,
            V_target_policy = V_target_policy,
            device = device,
            root_state = root_state,
        )
        # self.initialized_before = False
        # add bounds variable to call
        # assert False

        # self.lo = [np.max((i, -15)) for i in observation_bounds[0]]
        # self.hi = [np.min((i,  15)) for i in observation_bounds[1]]
        self.lo = [i for i in observation_bounds[0]]
        self.hi = [i for i in observation_bounds[1]]
        self.action_weight = 0.01
        self.backprop_mode = "very_greedy"
        # self.backprop_mode = "kernel_very_greedy"
        self.backprop_mode = "kernel_greedy_stationary"
        self.KDTree_constructor = KDTreePolicyAlt

    def return_results(
        self, final_selection: str, node=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        policy_dict = super().return_results(final_selection, node)
        lam = self.c_uct*self.global_n**(-0.5)

        counts = np.array(
            [child_action.n for i, child_action in enumerate(node.child_actions)]
        )
        policy_dict['counts'] = counts

        policy_dict['lambda'] = node.n**(-0.5)#lam
        policy_dict['total_volume'] = self.total_volume
        policy_dict['hi'] = node.hi
        policy_dict['lo'] = node.lo
        policy_dict['prob'] = node.prob
        policy_dict['base_prob'] = node.base_prob
        policy_dict['volume'] = node.children_density()/self.total_volume
        policy_dict['unweighted_volume'] = node.children_unweighted_density()/self.total_volume
        policy_dict['inv_density'] = node.children_inv_density()/self.total_volume
        policy_dict['local_volume'] = node.local_unweighted_density()/self.total_volume

        policy_dict['root_state'] = self.root_state
        return policy_dict


    def initialize_search(self) -> None:
        self.kd_tree = self.KDTree_constructor(lo=self.lo, hi=self.hi, density_model=self.model)
        self.root_node = KDNodeContinuous(  # type: ignore[assignment]
            self.root_state, r=0.0, terminal=False, parent_action=None
        )
        self.kd_tree.add(self.root_node.state, self.root_node)
        self.root_node.prob = 1
        self.root_node.base_prob = 1
        self.root_node.upstream_reward = 0
        self.root_node.depth = 0
        # ipdb.set_trace()
        self.total_volume = self.root_node.children_density()
        try: 
            assert self.root_node.local_volume is not None
        except: 
            ipdb.set_trace()
        if self.root_node.terminal:  # type: ignore[attr-defined]
            raise ValueError("Can't do tree search from a terminal node")
        self.global_n = 1


    def add_pw_action(self, node: KDNodeContinuous) -> None:
        state = (
            torch.from_numpy(
                node.state[
                    None,
                ]
            )
            .float()
            .to(self.device)
        )
        action, log_prob = self.model.sample_action(state, log_prob=True)  # type: ignore[operator]
        action = action.reshape(-1)


        new_child = KDActionContinuous(action, parent_node=node, Q_init=node.V, 
            log_prob=log_prob.item())
        new_child.upstream_reward = node.upstream_reward
        new_child.depth = node.depth
        node.child_actions.append(new_child)

    def expansion(self,
        action: Action, state: np.ndarray, reward: float, terminal: bool
    ) -> Node:
        # import ipdb
        # ipdb.set_trace()

        node = action.add_child_node(state, reward, terminal)
        node.prob = action.prob
        node.base_prob = action.base_prob
        node.upstream_reward = action.upstream_reward+reward
        node.depth = action.depth+1
        self.check_local_volume(action.parent_node)
        kd_node = self.kd_tree.add(state, node)
        self.total_volume == self.root_node.children_density()
        # assert kd_node.points[0][-1] == node
        try: 
            node.children_density()
            assert node.children_volume is not None
            assert node.local_volume is not None
        except: 
            node.set_new_local_volume(node.state, node.state, 0.00000001, 
                self.model.density(
                    # torch.tensor(node.state), torch.tensor(node.state)
                    # torch.tensor(node.state)
                    torch.tensor(self.root_node.state), torch.tensor(node.state)
                ).sum().detach().item()
            )
        assert node.local_volume is not None

        return node

    # def expansion(self,
    #     action: Action, state: np.ndarray, reward: float, terminal: bool
    # ) -> Node:
    #     node = super().expansion(action, state, reward, terminal)
    #     node.prob = action.prob
    #     node.base_prob = action.base_prob
    #     node.children_volume = lambda *x: 1
    #     return node

    def check_local_volume(self, node):
        node.local_volume
        if node.parent_action:
            self.check_local_volume(node.parent_action.parent_node)

    def selection(self, action):
        node = action.child_node
        node.prob = action.prob
        node.base_prob = action.base_prob
        return node

class MCTS_Volume_2(MCTS_Volume_Intermediate):
    def return_results(
        self, final_selection: str, node=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        policy_dict = super().return_results(final_selection, node)
        lam = self.c_uct*self.global_n**(-0.5)
        policy = calculate_policy_with_volume_2(node, 
            self.global_n, self.gamma**self.depth,
            with_pw=False, global_total_volume=self.total_volume)
        policy_dict['policy'] = [policy[i] for i, child_action in enumerate(node.child_actions)]
        return policy_dict






    # def selectionUCT(self, node: NodeContinuous) -> Action:  # type: ignore[return-value, override]
        
    #     if sum([not hasattr(act, "child_node") for act in node.child_actions]) > 1:
    #         print(f"Multiple child actions with no next state. This should never happen")
    #         ipdb.set_trace()


    #     if len(node.child_actions) == 0:
    #         self.add_pw_action(node)
    #         action = node.child_actions[-1]
    #         action.prob = node.prob
    #         action.base_prob = node.base_prob
    #         return action

    #     assert node.V.shape == ()

    #     for action in node.child_actions:
    #         if not hasattr(action, "child_node"):
    #             action.prob = node.prob
    #             action.base_prob = node.base_prob
    #             return action

    #     policy = calculate_policy_with_volume_2(node, 
    #         self.global_n, self.gamma**self.depth,
    #         with_pw=True, global_total_volume=self.total_volume)
        
    #     try:
    #         N = len(node.child_actions)+1
    #         action_index = np.random.choice(list(range(N)), p=policy)
    #         action = (node.child_actions + [PWAction(node)])[action_index]
    #         action.prob = policy[action_index]*node.prob
    #         action.base_prob = 1/N*node.base_prob
    #         # action = np.random.choice(node.child_actions + [PWAction(node)], p=policy)
    #         #issue -- volume of children not being measured correctly?
    #     except:
    #         import ipdb
    #         ipdb.set_trace()

    #     assert hasattr(action, "prob")
    #     assert hasattr(action, "base_prob")
    #     if type(action) is PWAction:
    #         self.add_pw_action(node)
    #         new_action = node.child_actions[-1]
    #         new_action.prob = action.prob
    #         new_action.base_prob = action.base_prob
    #         return new_action
    #     else: 
    #         return action



    def selectionUCT(self, node: NodeContinuous) -> Action:  # type: ignore[return-value, override]
        # if node.check_pw(self.c_pw, self.kappa) or len(node.child_actions) == 0:
        if len(node.child_actions) == 0:
            self.add_pw_action(node)
            action =  node.child_actions[-1]
            action.prob = node.prob
            action.base_prob = node.base_prob
            return action
        else:
            lam = self.c_uct*(node.n+1)**(-1/2)
            assert node.V.shape == ()

            for action in node.child_actions:
                if not hasattr(action, "child_node"):
                    action.prob = node.prob
                    action.base_prob = node.base_prob
                    return action

            if node.r > 0: 
                # ipdb.set_trace()
                return None

            if np.random.rand() < (1-self.gamma):
                self.add_pw_action(node)
                new_action = node.child_actions[-1]
                new_action.prob = action.prob
                new_action.base_prob = action.base_prob
                return new_action

            policy = calculate_policy_with_volume_2(node, 
                self.global_n, self.gamma**self.depth,
                # self.global_n, self.epoch**0.5*self.gamma**self.depth,
                # with_pw=False, global_total_volume=self.total_volume)                
                with_pw=True, global_total_volume=self.total_volume)
            
            N = len(node.child_actions)+1
            action_index = np.random.choice(list(range(N)), p=policy)
            # action = (node.child_actions[action_index]
            action = (node.child_actions + [PWAction(node)])[action_index]
            action.prob = policy[action_index]*node.prob
            action.base_prob = 1/N*node.base_prob
            if type(action) is PWAction:
                self.add_pw_action(node)
                new_action = node.child_actions[-1]
                new_action.prob = action.prob
                new_action.base_prob = action.base_prob
                return new_action
            else: 
                return action
            # return action


    # def selectionUCT(self, node: NodeContinuous) -> Action:  # type: ignore[return-value, override]
    #     action = super().selectionUCT(node)
    #     action.prob = 1
    #     action.base_prob = 1
    #     return action

    # # def ucb(self, parent_node: NodeContinuous, action_node: ActionContinuous):
    # #     parent_volume = parent_node.children_density()
    # #     action_volume = action_node.children_density()

    # #     try: 
    # #         assert parent_volume > 0
    # #         assert action_volume > 0
    # #         assert parent_volume >= action_volume
    # #     except: 
    # #         import ipdb
    # #         ipdb.set_trace()
    # #     return action_node.Q + self.c_uct * (( 
    # #         action_volume + (1-self.gamma)) 
    # #         * (parent_node.n / (action_node.n*self.global_n**0.5)))
    # #     # *self.global_n**0.5

    # def backprop(self, node: Node, gamma: float) -> None:
    #     R = node.V
    #     # loop back-up until root is reached
    #     while node.parent_action is not None:
    #         R = node.r + gamma * R
    #         action = node.parent_action
    #         action.update(R)
    #         if len(node.child_actions) > 1:
    #             # policy = calculate_policy(node.child_actions, node.n).reshape(-1)
    #             policy = calculate_policy_with_volume_2(node, 
    #                 self.global_n, self.gamma**self.depth,
    #                 with_pw=False, global_total_volume=self.total_volume)
    #             new_V = sum((
    #                 policy[i]*(node.child_actions[i].Q)
    #                 for i in range(len(node.child_actions))
    #             ))
    #             node.policy_V = new_V
    #             action.Q = node.r + gamma*new_V
    #         else: 
    #             node.policy_V = node.V
    #         node = action.parent_node    

    #         node.update_visit_counts()


# class MCTS_Volume_3(MCTSRPOAdvantage):
class MCTS_Volume_3(MCTS_Volume_Intermediate):
    def return_results(
        self, final_selection: str, node=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        policy_dict = super().return_results(final_selection, node)
        policy_dict['epoch'] = self.epoch
        return policy_dict


    def ucb(self, node, child_action, pseudocount):
        return super().ucb(node, child_action)
        offset = 1.0
        try:
            ucb_base = self.c_uct*(offset + node.n + pseudocount)**0.5/(offset + child_action.n + pseudocount/(len(node.child_actions)+1))
        except:
            ipdb.set_trace()
        ucb_value = child_action.Q + (ucb_base#*(1/(len(node.child_actions)+1)) #)
            # + self.c_uct*child_action.children_inv_density()
            # + 1/np.sqrt(1+self.epoch)*1/np.sqrt(1 + node.n)*child_action.children_inv_density()/self.total_volume)
            # + (1-1/np.sqrt(1+self.epoch))*child_action.children_density()/(node.children_density()+0.01)
            # + self.c_uct*1/np.sqrt(1+self.epoch)*np.log(1 + node.n)*np.sqrt(child_action.children_inv_density()+0.0001)/self.total_volume
        )
        try:
            assert not np.any(np.isnan(ucb_value))
        except:
            ipdb.set_trace()
        return ucb_value


    def selectionUCT(self, node: NodeContinuous) -> Action:  # type: ignore[return-value, override]
        # if node.check_pw(self.c_pw, self.kappa) or len(node.child_actions) == 0:
        with torch.no_grad():
            N = len(node.child_actions)+1
            if N == 1:
                self.add_pw_action(node)
                action =  node.child_actions[-1]
                action.prob = node.prob
                action.base_prob = node.base_prob
                return action
            else:
                lam = self.c_uct*(node.n+1)**(-1/2)
                assert node.V.shape == ()

                for action in node.child_actions:
                    if not hasattr(action, "child_node"):
                        action.prob = node.prob
                        action.base_prob = node.base_prob
                        return action

                pseudocount = 0

                UCT = np.array(
                    [ 
                        self.ucb(node, child_action, pseudocount) 
                        for child_action in (node.child_actions + [PWAction(node, n=N)])
                    ]
                )

                if self.epsilon == 0:
                    # do standard UCT action selection if epsilon=0
                    try: 
                        action_index = argmax(UCT)
                        action = (node.child_actions + [PWAction(node)])[action_index]
                    except: 
                        ipdb.set_trace()
                    # action = node.child_actions[winner]
                else:
                    action = self.epsilon_greedy(node=node, UCT=UCT)

                action.prob = action.n/self.global_n
                action.base_prob = 1/N*node.base_prob
                if type(action) is PWAction:
                    self.add_pw_action(node)
                    new_action = node.child_actions[-1]
                    new_action.prob = action.prob
                    new_action.base_prob = action.base_prob
                    return new_action
                else: 
                    return action

    def search(self, Env: gym.Env, epoch=0) -> None:
        self.epoch = epoch
        return super().search(Env)


class MCTS_Explorer(MCTS_Volume_2):
    pass
    def return_results(
        self, final_selection: str, node=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        policy_dict = super().return_results(final_selection, node)
        policy_dict['epoch'] = self.epoch
        return policy_dict

    def initialize_search(self) -> None:
        from alphazero.helpers import Object
        # density_model = Object()
        # Object.density = lambda x, y, z: 1
        self.kd_tree = self.KDTree_constructor(lo=self.lo, hi=self.hi, density_model=self.model)
        self.root_node = KDNodeContinuous(  # type: ignore[assignment]
            self.root_state, r=0.0, terminal=False, parent_action=None
        )
        self.kd_tree.add(self.root_node.state, self.root_node)
        self.root_node.prob = 1
        self.root_node.base_prob = 1
        # ipdb.set_trace()
        self.total_volume = self.root_node.children_density()
        try: 
            assert self.root_node.local_volume is not None
        except: 
            ipdb.set_trace()
        if self.root_node.terminal:  # type: ignore[attr-defined]
            raise ValueError("Can't do tree search from a terminal node")
        self.global_n = 1

    def ucb(self, node, child_action, pseudocount):
        # return child_action.Q + self.c_uct * (np.sqrt(node.n + 1) / (child_action.n + 1))
        # return action.Q + lam*(node.n/self.global_n) 
        #     *node.local_volume()/pi_a + lam*action.children_volume()/pi_a
        # return action.Q + (lam*node.local_volume()/len(node.child_actions)*1/(pi_a)
        #         # + lam*action.children_volume()/self.total_volume*1/pi_a)
        #         + lam*action.children_volume()/self.total_volume*1/pi_a)
        offset = 1.0
        try:
            ucb_base = self.c_uct*(offset + node.n + pseudocount)**0.5/(offset + child_action.n + pseudocount/(len(node.child_actions)+1))
        except:
            ipdb.set_trace()
            #len(node.child_actions)+1 because we have N arms to pull, plus 1 option to make a new arm
        # return action.Q + ucb_base*np.exp(-action.log_prob)*node.local_volume()/node.children_volume() + ucb_base*action.children_volume()/node.children_volume()
        # return child_action.Q + ucb_base*((np.exp(-child_action.log_prob)+0.001)/(len(node.child_actions)+1))
        # return child_action.Q + ucb_base*(np.exp(-child_action.log_prob)/(len(node.child_actions)+1) + 
        #     child_action.children_density()/node.children_density())
        
        # ucb_value = child_action.Q + ucb_base*(node.local_density()/node.children_density() + 
        #     child_action.children_density()/node.children_density())
        # ucb_value = child_action.Q + ucb_base*((np.exp(-child_action.log_prob)+0.001)/(len(node.child_actions)+1) + 
        #     child_action.children_density()/node.children_density())
        # ucb_value = child_action.Q + (ucb_base*((np.exp(-child_action.log_prob)+0.001)/(len(node.child_actions)+1)) 
        #     + 1/np.sqrt(offset + node.n + pseudocount)*child_action.children_inv_density()/self.total_volume)
        ucb_value = child_action.Q + (ucb_base#*(1/(len(node.child_actions)+1)) #)
            # + 1/np.sqrt(1+self.epoch)*1/np.sqrt(1 + node.n)*child_action.children_inv_density()/self.total_volume)
            # + (1-1/np.sqrt(1+self.epoch))*child_action.children_density()/(node.children_density()+0.01)
            + self.c_uct*1/np.sqrt(1+self.epoch)*np.log(1 + node.n)*np.sqrt(child_action.children_inv_density()+0.0001)/self.total_volume
        )
        # ucb_value = child_action.Q + ucb_base*(child_action.children_density()/node.children_density())
        try:
            assert not np.any(np.isnan(ucb_value))
        except:
            ipdb.set_trace()
        return ucb_value



    # def ucb(self, parent_node: NodeContinuous, action_node: ActionContinuous):
    #     # try: 
    #     #     action_node.child_node
    #     # except: 
    #     #     ipdb.set_trace()
    #     return action_node.Q + self.c_uct * (np.sqrt(parent_node.n + 1) / (action_node.n + 1))


    # def selectionUCT(self, node: NodeContinuous) -> Action:  # type: ignore[return-value, override]
    #     # if node.check_pw(self.c_pw, self.kappa) or len(node.child_actions) == 0:
    #     with torch.no_grad():
    #         N = len(node.child_actions)+1
    #         if N == 1:
    #             self.add_pw_action(node)
    #             action =  node.child_actions[-1]
    #             action.prob = node.prob
    #             action.base_prob = node.base_prob
    #             return action
    #         else:
    #             lam = self.c_uct*(node.n+1)**(-1/2)
    #             assert node.V.shape == ()

    #             for action in node.child_actions:
    #                 if not hasattr(action, "child_node"):
    #                     action.prob = node.prob
    #                     action.base_prob = node.base_prob
    #                     return action

    #             # state = (
    #             #     torch.from_numpy(
    #             #         node.state[None,]
    #             #     )
    #             #     .float()
    #             #     .to(self.device)
    #             # )
    #             # root_state = (
    #             #     torch.from_numpy(
    #             #         self.root_state[None,]
    #             #     )
    #             #     .float()
    #             #     .to(self.device)
    #             # )
    #             # self.epoch = 0
    #             if type(node.density) is torch.Tensor:
    #                 pseudocount = self.epoch*node.density.detach().sum().item()#(self.epoch*self.model.density(root_state, state)).sum().item()
    #             else: 
    #                 pseudocount = self.epoch*node.density

    #             # if np.random.rand() < 0.0001:
    #             #     ipdb.set_trace()

    #             UCT = np.array(
    #                 [ 
    #                     self.ucb(node, child_action, pseudocount) 
    #                     for child_action in (node.child_actions + [PWAction(node, n=N)])
    #                 ]
    #             )

    #             if self.epsilon == 0:
    #                 # do standard UCT action selection if epsilon=0
    #                 try: 
    #                     action_index = argmax(UCT)
    #                     action = (node.child_actions + [PWAction(node)])[action_index]
    #                 except: 
    #                     ipdb.set_trace()
    #                 # action = node.child_actions[winner]
    #             else:
    #                 action = self.epsilon_greedy(node=node, UCT=UCT)

    #             action.prob = action.n/self.global_n
    #             action.base_prob = 1/N*node.base_prob
    #             if type(action) is PWAction:
    #                 self.add_pw_action(node)
    #                 new_action = node.child_actions[-1]
    #                 new_action.prob = action.prob
    #                 new_action.base_prob = action.base_prob
    #                 return new_action
    #             else: 
    #                 return action

    def selectionUCT(self, node: NodeContinuous) -> Action:  # type: ignore[return-value, override]
        # if node.check_pw(self.c_pw, self.kappa) or len(node.child_actions) == 0:
        with torch.no_grad():
            N = len(node.child_actions)+1
            if N == 1:
                self.add_pw_action(node)
                action =  node.child_actions[-1]
                action.prob = node.prob
                action.base_prob = node.base_prob
                return action
            else:
                lam = self.c_uct*(node.n+1)**(-1/2)
                assert node.V.shape == ()

                for action in node.child_actions:
                    if not hasattr(action, "child_node"):
                        action.prob = node.prob
                        action.base_prob = node.base_prob
                        return action

                pseudocount = 0

                UCT = np.array(
                    [ 
                        self.ucb(node, child_action, pseudocount) 
                        for child_action in (node.child_actions + [PWAction(node, n=N)])
                    ]
                )

                if self.epsilon == 0:
                    # do standard UCT action selection if epsilon=0
                    try: 
                        action_index = argmax(UCT)
                        action = (node.child_actions + [PWAction(node)])[action_index]
                    except: 
                        ipdb.set_trace()
                    # action = node.child_actions[winner]
                else:
                    action = self.epsilon_greedy(node=node, UCT=UCT)

                action.prob = action.n/self.global_n
                action.base_prob = 1/N*node.base_prob
                if type(action) is PWAction:
                    self.add_pw_action(node)
                    new_action = node.child_actions[-1]
                    new_action.prob = action.prob
                    new_action.base_prob = action.base_prob
                    return new_action
                else: 
                    return action

    def search(self, Env: gym.Env, epoch=0) -> None:
        self.epoch = epoch
        return super().search(Env)

    # def backprop(self, node: Node, gamma: float) -> None:
    #     R = node.V
    #     inv_dens = node.children_inv_density()
    #     # loop back-up until root is reached
    #     while node.parent_action is not None:
    #         R = node.r + gamma * R
    #         inv_dens = node.local_inv_density() + gamma * inv_dens
    #         action = node.parent_action
    #         action.update(R + 1/np.sqrt(self.epoch)*inv_dens)
    #         action.Q = R + 1/np.sqrt(self.epoch)*inv_dens
    #         node = action.parent_node    

    #         node.update_visit_counts()




class MCTS_Frequency(MCTS_Explorer):
    def ucb(self, node, child_action, pseudocount):
        offset = 1.0
        try:
            time_base = -self.c_uct*(self.epoch+1)**(-0.5)*child_action.children_density()
            ucb_base = self.c_uct*(offset + node.n)**0.5/(offset + child_action.n)
        except:
            ipdb.set_trace()

        ucb_value = child_action.Q + time_base + ucb_base

        try:
            assert not np.any(np.isnan(ucb_value))
        except:
            ipdb.set_trace()
        return ucb_value



class MCTS_One_Shot(MCTS_Volume_2):
    def __init__(
        self,
        model: torch.nn.Module,
        n_rollouts: int,
        c_uct: float,
        c_pw: float,
        kappa: float,
        gamma: float,
        epsilon: float,
        V_target_policy: str,
        device: str,
        root_state: np.ndarray,
        # bounds: list,#[np.ndarray],
        observation_bounds: list,
    ):
        super().__init__(
            model = model,
            n_rollouts = n_rollouts,
            c_uct = c_uct,
            c_pw = c_pw,
            kappa = kappa,
            gamma = gamma,
            epsilon = epsilon,
            V_target_policy = V_target_policy,
            device = device,
            root_state = root_state,
            observation_bounds = observation_bounds
        )
        # self.backprop_mode = "very_greedy"
        # self.backprop_mode = "kernel_very_greedy"
        self.backprop_mode = "kernel_very_greedy_stationary"
        self.lambda_coeff = c_uct#100

    def env_copy(self, env):
        mcts_env = copy.deepcopy(env)
        if not hasattr(mcts_env, "goal"):# is None:
            mcts_env.goal = copy.deepcopy(env.goal)
        if check_goal: assert mcts_env.goal is not None
        return mcts_env

    def planMore(self, n):
        Env = self.Env
        save_state = True
        # save_state = False
        if save_state:
            self.root_node.saved_state = Env.get_save_state()
            self.root_node.goal_obs = copy.deepcopy(Env.get_goal_obs())
            # mcts_env = Env


        for i in range(n):
            # reset to root for new trace
            node = self.root_node
            self.global_n += 1

            if save_state:
                mcts_env = self.env_copy(Env)
            else:
                mcts_env = copy.deepcopy(Env)
                if check_goal: assert mcts_env.goal is not None

            self.depth = 0
            terminal = False
            while not node.terminal and not terminal:
                action = self.selectionUCT(node)
                self.depth += 1
                if save_state:
                    pass
                else:
                    new_state, reward, terminal, _ = mcts_env.step(action.action)
                    # reward /= PENDULUM_R_SCALE
                # else:
                #     new_state = node.state
                #     reward = node.r

                if hasattr(action, "child_node"):
                    node = self.selection(action)
                    continue
                elif type(action) == type(None):
                    continue
                else:
                    if save_state:
                        mcts_env.restore_save_state(node.saved_state)
                        new_state, reward, terminal, _ = mcts_env.step(action.action)
                        # reward /= PENDULUM_R_SCALE
                    else:
                        pass
                    node = self.expansion(
                        action, np.squeeze(new_state), reward, terminal
                    )
                    if save_state:
                        node.saved_state = mcts_env.get_save_state()
                        node.goal_obs = copy.deepcopy(mcts_env.get_goal_obs())
                    else:
                        pass
                    self.add_value_estimate(node)
                    break

            self.backprop(node, self.gamma, terminal=terminal)

        if save_state:
            mcts_env.restore_save_state(self.root_node.saved_state)

#--------------------------------------------------------------------------------------------------------------------------
    #New derivation after this point

    def preprocess_return_results(self):
        self.root_node.depth = 0
        self.root_node.base_prob = 1
        self.preprocess_return_results_helper(self.root_node)

    def preprocess_return_results_helper(self, node):
        policy = calculate_one_shot_policy(node, self.global_n, 
            self.epoch, self.gamma**self.depth,              
            with_pw=True, global_total_volume=self.total_volume)
            
        node.prob = (1-self.gamma + self.gamma*policy[-1])*node.base_prob
        # node.prob = (policy[-1])*node.base_prob
        node.density = node.prob/node.local_unweighted_density()

        if len(node.child_actions) > 0:
            node.new_value = node.r + policy[-1]*node.r/(1-self.gamma) + self.gamma*sum(
            [
                policy[i]*action.Q
                for i, action in enumerate(node.child_actions)
            ])
        else: 
            node.new_value = node.V#node.r/(1-self.gamma)

        for i, action in enumerate(node.child_actions):
            next_node = action.child_node
            next_node.base_prob = node.base_prob*policy[i]
            self.preprocess_return_results_helper(next_node)

        if len(node.child_actions) == 0:
            node

    def return_results(
        self, final_selection: str, node=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        policy_dict = super().return_results(final_selection, node)
        sumval, dens = self.kd_tree.knn_estimate(
            node.state, 
            lambda n: np.array([
                n.V, 
                n.density_bonus,
                n.children_density()/self.total_volume,
                n.children_unweighted_density()/self.total_volume,
                n.children_inv_density()/self.total_volume,
                # n.local_density()/self.total_volume,
                n.local_unweighted_density()/self.total_volume, 
                n.prob*self.total_volume/node.local_unweighted_density()
            ]), 
            k=5
        )
        ave = sumval/(dens+0.00001)

        c = 1/(1-self.gamma)*np.log(self.global_n)
        value = self.kd_tree.nearby_with_penalty(node.state.tolist(), 
            lambda location, x: x.V - c*np.linalg.norm(x.state - node.state)
        )
        policy_dict['V_target'] = max(value, node.V, node.new_value)
        # policy_dict['V_target'] = ave[0]
        # policy_dict['V_target'] = max(node.V, node.new_value)
        # policy_dict['V_target'] = node.new_value
        # policy_dict['V_target'] = node.state.sum()/(1-self.gamma)
        # # policy_dict['visitation_rate'] = ave[1]/self.global_n
        policy_dict['volume'] = ave[-1]
        # policy_dict['volume'] = ave[1]#/self.global_n
        # policy_dict['unweighted_volume'] = ave[3]
        # policy_dict['inv_density'] = ave[4]
        # policy_dict['local_volume'] = ave[5]

        # policy_dict['V_target'] = ave[0]
        # policy_dict['r'] = np.array([node.r])
        # policy_dict['V_target'] = np.array([node.r])
        # policy_dict['V_target'] = np.array([node.state.sum()])
        # policy_dict['visitation_rate'] = ave[1]/self.global_n
        # policy_dict['volume'] = ave[2]
        policy_dict['traj_value'] = node.upstream_reward + self.gamma**node.depth*policy_dict['V_target']

        # policy_dict['volume'] = (
        #     self.total_volume/node.local_unweighted_density()*
        #     len(node.child_actions + [])/self.global_n
        # )
        policy_dict['children_unweighted_density'] = [
            action.children_unweighted_density()/self.total_volume 
            for action in node.child_actions
        ]

        policy_dict['volume'] = (
            # self.total_volume/node.local_unweighted_density()*
            # node.prob
            node.density
        )
        policy_dict['base_prob'] = node.base_prob
        # import ipdb
        # ipdb.set_trace()
        # policy_dict['policy'] = [np.array(1/len(node.child_actions)) for i, child_action in enumerate(node.child_actions)]
        return policy_dict

    def ucb(self, node, child_action, pseudocount):
        value_weight = 1#self.epoch**0.5
        action_weight = 1
        density_weight = 10
        action_bonus = action_weight*self.c_uct*node.n**0.5/(child_action.n*(1+len(node.child_actions)))
        no_search = False
        no_search = True
        if not hasattr(child_action, "child_node") or no_search:
            Q =  value_weight*child_action.Q
            density_bonus = density_weight*self.c_uct*self.global_n**0.5*child_action.density_bonus
        else:
            sumval, dens = self.kd_tree.knn_estimate(
                child_action.child_node.state, 
                lambda n: np.array([n.V, n.density_bonus]), 
                k=5
            )
            ave = sumval/(dens+0.00001)
            Q=ave[0]
            density_bonus=ave[1]
            density_bonus = density_weight*self.c_uct*self.global_n**0.5*child_action.density_bonus
            node.kernel_value = Q
        ucb_value = value_weight*Q + density_bonus + action_bonus
        return ucb_value

    def selectionUCT(self, node: NodeContinuous) -> Action:  # type: ignore[return-value, override]
        # if node.check_pw(self.c_pw, self.kappa) or len(node.child_actions) == 0:
        if len(node.child_actions) == 0:
            self.add_pw_action(node)
            action =  node.child_actions[-1]
            action.prob = node.prob
            action.base_prob = node.base_prob
            return action
        else:
            lam = self.c_uct*(node.n+1)**(-1/2)
            assert node.V.shape == ()

            for action in node.child_actions:
                if not hasattr(action, "child_node"):
                    action.prob = node.prob
                    action.base_prob = node.base_prob
                    return action

            # if self.depth == 1:
            #     import ipdb
            #     ipdb.set_trace()
                #Figure out how to do the random ancestor selection

            policy = calculate_policy_with_volume_2(node, 
                self.global_n, self.gamma**self.depth,
                # self.global_n, self.epoch**0.5*self.gamma**self.depth,
                # with_pw=False, global_total_volume=self.total_volume)                
                with_pw=True, global_total_volume=self.total_volume, 
                lambda_coeff=self.lambda_coeff
                )

            N = len(node.child_actions)+1
            action_index = np.random.choice(list(range(N)), p=policy)
            # action = (node.child_actions[action_index]
            action = (node.child_actions + [PWAction(node)])[action_index]
            action.prob = policy[action_index]*node.prob
            action.base_prob = 1/N*node.base_prob
            if type(action) is PWAction:
                self.add_pw_action(node)
                new_action = node.child_actions[-1]
                new_action.prob = action.prob
                new_action.base_prob = action.base_prob
                return new_action
            else: 
                return action

    def backprop(self, node: Node, gamma: float, terminal=False) -> None:
        R = node.V
        if terminal: 
            R = z + 1/(1-gamma)*node.r
            path_val = z + 1/(1-gamma)*node.r
        pessimistic_R = node.r/(1-gamma)
        optimistic_R = R
        inv_density = node.local_density()
        node.density_bonus = inv_density
        # loop back-up until root is reached
        while node.parent_action is not None:
            # if node.r > 0: 
            #     ipdb.set_trace()
            R = max(1/(1-gamma)*node.r, node.r + gamma * R)
            optimistic_R = max(1/(1-gamma)*node.r, R, node.r + gamma * optimistic_R)
            pessimistic_R = max(1/(1-gamma)*node.r, node.r + gamma * pessimistic_R)

            action = node.parent_action
            action.update(R)
            # action.kernel_value = action.Q
            # node.kernel_value = sum([a.W for a in node.child_actions] + [node.V])/sum([a.n for a in node.child_actions] + [1])

            prev = action.kernel_value if hasattr(action, "kernel_value") else action.Q
            action.kernel_value = max(optimistic_R, action.Q, prev)
            action.density_bonus = inv_density
            action.pessimistic_value = pessimistic_R
            node.kernel_value = max([a.kernel_value for a in node.child_actions] + [1/(1-gamma)*node.r])
            node.V = action.Q
            node.pessimistic_value = max([a.pessimistic_value for a in node.child_actions] + [1/(1-gamma)*node.r])
            node = action.parent_node    
            inv_density = sum([a.density_bonus for a in node.child_actions] + [node.local_density()/(len([node.child_actions])+1)])
            # inv_density = node.local_density()/(len(node.child_actions)+1) + gamma*sum([a.density_bonus for a in node.child_actions])
            node.density_bonus = inv_density

            node.update_visit_counts()

    def add_value_estimate(self, node: NodeContinuous) -> None:
        state = (torch.from_numpy(node.state[None,]).float().to(self.device))
        node.V = (
            np.squeeze(self.model.predict_V(state))
            if not node.terminal
            else np.array(1/(1-self.gamma)*node.r)
        )
        value = self.kd_tree.best_nearby(node.state.tolist(), lambda x: x.V)
        c = 1/(1-self.gamma)*np.log(self.global_n)
        value = self.kd_tree.nearby_with_penalty(node.state.tolist(), 
            lambda location, x: x.V - c*np.linalg.norm(x.state - node.state)
        )
        alpha = 0.5
        node.V = (1-alpha)*node.V + alpha*max(value, 1/(1-self.gamma)*node.r)

        if node.V.shape != ():
            import ipdb
            ipdb.set_trace()

        assert node.V.shape == ()



class MCTS_One_Shot_2(MCTS_One_Shot):
    def __init__(
        self,
        model: torch.nn.Module,
        n_rollouts: int,
        c_uct: float,
        c_pw: float,
        kappa: float,
        gamma: float,
        epsilon: float,
        V_target_policy: str,
        device: str,
        root_state: np.ndarray,
        # bounds: list,#[np.ndarray],
        observation_bounds: list,
    ):
        super().__init__(
            model = model,
            n_rollouts = n_rollouts,
            c_uct = c_uct,
            c_pw = c_pw,
            kappa = kappa,
            gamma = gamma,
            epsilon = epsilon,
            V_target_policy = V_target_policy,
            device = device,
            root_state = root_state,
            observation_bounds = observation_bounds
        )
        self.KDTree_constructor = KDTreeValue

    def preprocess_return_results_helper(self, node):
        policy = calculate_one_shot_policy(node, self.global_n, 
            self.epoch, self.gamma**self.depth,              
            with_pw=True, global_total_volume=self.total_volume)
            
        node.prob = (1-self.gamma + self.gamma*policy[-1])*node.base_prob
        # node.prob = (policy[-1])*node.base_prob
        node.density = node.prob/node.local_unweighted_density()

        if len(node.child_actions) > 0:
            node.new_value = node.r + policy[-1]*node.r/(1-self.gamma) + self.gamma*sum(
            [
                policy[i]*action.Q
                for i, action in enumerate(node.child_actions)
            ])
        else: 
            node.new_value = node.V#node.r/(1-self.gamma)

        for i, action in enumerate(node.child_actions):
            next_node = action.child_node
            next_node.base_prob = node.base_prob*policy[i]
            self.preprocess_return_results_helper(next_node)

        if len(node.child_actions) == 0:
            node
            
    def return_results(
        self, final_selection: str, node=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        policy_dict = super().return_results(final_selection, node)
        sumval, dens = self.kd_tree.knn_estimate(
            node.state, 
            lambda n: np.array([
                n.V, 
                n.density_bonus,
                n.children_density()/self.total_volume,
                n.children_unweighted_density()/self.total_volume,
                n.children_inv_density()/self.total_volume,
                # n.local_density()/self.total_volume,
                n.local_unweighted_density()/self.total_volume, 
                n.density,
                n.prob*self.total_volume/node.local_unweighted_density()
            ]), 
            k=5
        )
        ave = sumval/(dens+0.00001)

        c = 1/(1-self.gamma)*np.log(self.global_n)
        policy_dict['V_target'] = self.kd_tree.the_value_from_halfway_down(node.state.tolist())
        policy_dict['children_unweighted_density'] = [
            action.children_unweighted_density()/self.total_volume 
            for action in node.child_actions
        ]
        policy_dict['local_volume'] = node.local_unweighted_density()
        policy_dict['volume'] = (
            node.density
        )
        return policy_dict

    def backprop(self, node: Node, gamma: float, terminal=False) -> None:
        R = node.V 
        assert type(node.V) is np.ndarray
        if type(node.V) == int: ipdb.set_trace()
        if terminal: 
            R = 1/(1-gamma)*node.r
            path_val =  1/(1-gamma)*node.r
        pessimistic_R = node.r/(1-gamma)
        node.pessimistic_value = pessimistic_R
        optimistic_R = R
        inv_density = node.local_density()
        node.density_bonus = inv_density
        self.kd_tree.backprop(node.state, R)

        # if node.r >= 1:
        #     ipdb.set_trace()

        # loop back-up until root is reached
        while node.parent_action is not None:

            R = max(1/(1-gamma)*node.r, node.r + gamma * R)
            optimistic_R = max(1/(1-gamma)*node.r, R, node.r + gamma * optimistic_R)
            pessimistic_R = max(1/(1-gamma)*node.r, node.r + gamma * pessimistic_R)

            action = node.parent_action
            action.update(R)

            prev = action.kernel_value if hasattr(action, "kernel_value") else action.Q
            action.kernel_value = max(optimistic_R, action.Q, prev)
            action.density_bonus = inv_density
            if not hasattr(action, "pessimistic_value"):
                action.pessimistic_value = pessimistic_R
            else:
                action.pessimistic_value = max(pessimistic_R, action.pessimistic_value)
            node.kernel_value = np.array(
                max([a.kernel_value for a in node.child_actions] + [1/(1-gamma)*node.r])
            )
            node.V = np.array(action.Q)
            assert type(node.V) is np.ndarray

            node = action.parent_node    
            node.pessimistic_value = max([a.pessimistic_value for a in node.child_actions] + [1/(1-gamma)*node.r])
            pessimistic_R = node.pessimistic_value
            inv_density = sum([a.density_bonus for a in node.child_actions] + [node.local_density()/(len([node.child_actions])+1)])
            node.density_bonus = inv_density

            node.update_visit_counts()
            self.kd_tree.backprop(node.state, R)
            if type(node.V) == int: ipdb.set_trace()
        
        node.pessimistic_value = max([a.pessimistic_value for a in node.child_actions] + [1/(1-gamma)*node.r])


    def add_value_estimate(self, node: NodeContinuous) -> None:
        state = (torch.from_numpy(node.state[None,]).float().to(self.device))
        nn_V = np.squeeze(self.model.predict_V(state))
        w, n = self.kd_tree.the_value_from_halfway_down(node.state.tolist(), separate_sums=True)
        value = (nn_V + w)/(n + 1)
        alpha = 1.0
        node.V = (
            np.array(value)
            # alpha*value + (1-alpha)*node.r/(1-self.gamma)
            if not node.terminal
            else np.array(1/(1-self.gamma)*node.r)
        )

        if node.V.shape != ():
            import ipdb
            ipdb.set_trace()
        node.pessimistic_value = node.r/(1-self.gamma)
        if not type(node.V) is np.ndarray:
            import ipdb
            ipdb.set_trace()

        assert node.V.shape == ()

    def selectionUCT(self, node: NodeContinuous) -> Action:  # type: ignore[return-value, override]
        if len(node.child_actions) == 0:
            self.add_pw_action(node)
            action =  node.child_actions[-1]
            action.prob = node.prob
            action.base_prob = node.base_prob
            return action
        else:
            lam = self.c_uct*(node.n+1)**(-1/2)
            if type(node.V) != np.ndarray:
                ipdb.set_trace()
            assert node.V.shape == ()

            for action in node.child_actions:
                if not hasattr(action, "child_node"):
                    action.prob = node.prob
                    action.base_prob = node.base_prob
                    return action


            if hasattr(node, "parent_action") and node.parent_action != None:
                node.prev_volume = node.parent_action.parent_node.prev_volume + node.local_density()
            else: 
                node.prev_volume = node.local_density()

            node.V = np.array(self.kd_tree.the_value_from_halfway_down(node.state.tolist()))
            for action in node.child_actions:
                action.Q = self.kd_tree.the_value_from_halfway_down(action.child_node.state.tolist())
            if type(node.V) == int: ipdb.set_trace()

            policy = calculate_policy_with_volume_2(node, 
                self.global_n, self.gamma**self.depth,             
                with_pw=True, global_total_volume=self.total_volume, 
                lambda_coeff=self.lambda_coeff)

            # policy = calculate_one_shot_policy(node, self.global_n, 
            #     self.epoch, self.gamma**self.depth,              
            #     with_pw=True, global_total_volume=self.total_volume)

            
            # policy = calculate_policy_with_volume_without_volume(node, 
            #     self.global_n, self.gamma**self.depth,             
            #     with_pw=True, global_total_volume=self.total_volume, 
            #     lambda_coeff=self.lambda_coeff)

            N = len(node.child_actions)+1
            action_index = np.random.choice(list(range(N)), p=policy)
            action = (node.child_actions + [PWAction(node)])[action_index]
            action.prob = policy[action_index]*node.prob
            action.base_prob = 1/N*node.base_prob
            if type(action) is PWAction:
                self.add_pw_action(node)
                new_action = node.child_actions[-1]
                new_action.prob = action.prob
                new_action.base_prob = action.base_prob
                action = new_action

            return action


class MCTS_Open_Loop_Continuous(MCTS_One_Shot_2):
    # def selectionUCT(self, node: NodeContinuous) -> Action:  # type: ignore[return-value, override]
    #     return super()
    #     if len(node.child_actions) == 0:
    #         self.add_pw_action(node)
    #         action =  node.child_actions[-1]
    #         action.prob = node.prob
    #         action.base_prob = node.base_prob
    #         return action
    #     else:
    #         lam = self.c_uct*(node.n+1)**(-1/2)
    #         if type(node.V) != np.ndarray:
    #             ipdb.set_trace()
    #         assert node.V.shape == ()

    #         for action in node.child_actions:
    #             if not hasattr(action, "child_node"):
    #                 action.prob = node.prob
    #                 action.base_prob = node.base_prob
    #                 return action


    #         if hasattr(node, "parent_action") and node.parent_action != None:
    #             node.prev_volume = node.parent_action.parent_node.prev_volume + node.local_density()
    #         else: 
    #             node.prev_volume = node.local_density()

    #         node.V = np.array(self.kd_tree.the_value_from_halfway_down(node.state.tolist()))
    #         for action in node.child_actions:
    #             action.Q = self.kd_tree.the_value_from_halfway_down(action.child_node.state.tolist())
    #         if type(node.V) == int: ipdb.set_trace()

    #         # policy = calculate_policy_with_volume_without_volume(node, 
    #         #     self.global_n, self.gamma**self.depth,             
    #         #     with_pw=True, global_total_volume=self.total_volume, 
    #         #     lambda_coeff=self.lambda_coeff)

    #         policy = calculate_policy(node.child_actions, lam)

    #         N = len(node.child_actions)+1
    #         action_index = np.random.choice(list(range(N)), p=policy)
    #         action = (node.child_actions + [PWAction(node)])[action_index]
    #         action.prob = policy[action_index]*node.prob
    #         action.base_prob = 1/N*node.base_prob
    #         if type(action) is PWAction:
    #             self.add_pw_action(node)
    #             new_action = node.child_actions[-1]
    #             new_action.prob = action.prob
    #             new_action.base_prob = action.base_prob
    #             action = new_action

    #         return action

    def selectionUCT(self, node: NodeContinuous) -> Action:  # type: ignore[return-value, override]
        if node.check_pw(self.c_pw, self.kappa):
            self.add_pw_action(node)
            action =  node.child_actions[-1]
            action.prob = node.prob
            action.base_prob = node.base_prob
            return action
        else:
            lam = self.c_uct*(node.n+1)**(-1/2)
            policy = calculate_policy(node.child_actions, lam)
            N = len(node.child_actions)
            action_index = np.random.choice(list(range(N)), p=policy)
            action = np.random.choice(node.child_actions, p=policy)
            action = node.child_actions[action_index]
            action.prob = policy[action_index]*node.prob
            action.base_prob = 1/N*node.base_prob
            return action


def check_node_invariants(node):
    try: 
        node.children_density()
        # assert hasattr(node, "children_volume")
    except: 
        import ipdb
        ipdb.set_trace()

class SSMCTS(MCTS_One_Shot_2):


    def expansion(self,
        action: Action, state: np.ndarray, reward: float, terminal: bool
    ) -> Node:

        # return super().expansion(action, state, reward, terminal)
        ##Add replacement logic here
        kdnode_at_state = self.kd_tree.locate(state.tolist(), inc=False)
        value = self.kd_tree.halfway_down_value_of_kd_node(kdnode_at_state)
        node_at_state = kdnode_at_state.points[0][1]
        bellman = reward + self.gamma*value

        new_traj_value = (
            action.upstream_reward + self.gamma**action.depth*reward 
            + self.gamma**(action.depth + 1)*value
            - self.root_node.V
        )
        old_traj_value = (
            node_at_state.upstream_reward 
            + self.gamma**node_at_state.depth*value
            - self.root_node.V
        )
        split_traj_value = (
            new_traj_value 
            + old_traj_value
            + self.lambda_coeff*self.global_n**0.5*node_at_state.local_density()*np.log(2)
            # (
            #         np.log((len(kdnode_at_state.points)+1)/(len(kdnode_at_state.points)))
            #     )
            # + 1/3*self.lambda_coeff*self.global_n**0.5*node_at_state.local_density()
            # -10000000            
        )
        d = state.shape[0]
        # split_traj_value = (
        #     new_traj_value 
        #     + old_traj_value
        #     + self.lambda_coeff*self.global_n**0.5*(
        #         d*np.log(np.linalg.norm(node_at_state.state - state)+0.0000001)
        #         + np.log(self.global_n)
        #         #D_RKL(psi^ || psi_0) = \int(psi_0 log (psi^/psi_0))
        #         #  1-NN density estimator: 
        #         #   = \int(psi_0 log((1/N*1/(Vd*||x - 1NN(x)||^d)))/psi_0))
        #         #   = \int(psi_0 [ 
        #         #        -log(N) - log(Vd) - d*log(||x - 1NN(x)||) - log(psi_0)
        #         #     ]
        #         #np.log(psi^/psi_0) = np.log(psi^/psi_0)
        #     )
        # )

        split_traj_value = (
            1/2*(new_traj_value 
            + old_traj_value)
            + self.lambda_coeff*self.global_n**(-0.5)*(
                d*np.log(np.linalg.norm(node_at_state.state - state)+0.0000001)
                + np.log(self.global_n)
                + np.log(np.pi)
                #D_RKL(psi^ || psi_0) = \int(psi_0 log (psi^/psi_0))
                #  1-NN density estimator: 
                #   = \int(psi_0 log((1/N*1/(Vd*||x - 1NN(x)||^d)))/psi_0))
                #   = \int(psi_0 [ 
                #        -log(N) - log(Vd) - d*log(||x - 1NN(x)||) - log(psi_0)
                #     ]
                #np.log(psi^/psi_0) = np.log(psi^/psi_0)
            )
        )

        # split_traj_value = (
        #     1/2*(new_traj_value 
        #     + old_traj_value)
        #     + self.lambda_coeff*self.global_n**0.5*node_at_state.local_density()
        #     # + 1/3*self.lambda_coeff*self.global_n**0.5*node_at_state.local_density()
        #     # -10000000            
        # )

        if node_at_state.local_unweighted_density() <= 0.001:
            split_traj_value -= float('inf')

        # new_traj_value = (
        #     node.V*(self.global_n-1)/self.global_n + 
        #     1/self.global_n*(
        #         action.upstream_reward + self.gamma**action.depth*reward
        #         + self.gamma**(action.depth + 1)*value                
        #         - node_at_state.upstream_reward 
        #         - self.gamma**node_at_state.depth*value
        #     )
        # )
        # old_traj_value = node.V
        # split_traj_value = (
        #     node.V*(self.global_n-1)/self.global_n + 
        #     1/self.global_n*(
        #         action.upstream_reward + self.gamma**action.depth*reward
        #         + self.gamma**(action.depth + 1)*value   
        #     )
        #     + self.global_n**(-0.5)*node_at_state.local_density()
        # )

        # old_traj_value = -float("inf")
        # temp_value = bellman #1/(1-self.gamma)*reward #value
        # if ((old_traj_value >= new_traj_value and old_traj_value >= split_traj_value) or 
        #     (np.linalg.norm(node_at_state.state - state) < 0.0000001)):

        temp_value = bellman #1/(1-self.gamma)*reward #value
        if ((old_traj_value >= new_traj_value and old_traj_value >= split_traj_value)):
            action.delete_self()
            return node_at_state
        elif new_traj_value >= old_traj_value and new_traj_value >= split_traj_value:
            node = action.add_child_node(state, reward, terminal)
            node.prob = action.prob
            node.base_prob = action.base_prob
            node.depth = action.depth+1
            node.upstream_reward = action.upstream_reward+self.gamma**node.depth*reward
            # node.V = np.array(temp_value)

            self.kd_tree.replace_kdnode_point(
                kdnode_at_state, 
                node.state.tolist(), 
                node
            )
            check_node_invariants(node)
            check_node_invariants(action)
            kdnode_at_state.pass_volume()

            return node
        else: 
            pass

        node = action.add_child_node(state, reward, terminal)
        node.prob = action.prob
        node.base_prob = action.base_prob
        node.depth = action.depth+1
        node.upstream_reward = action.upstream_reward+self.gamma**node.depth*reward
        # node.V = np.array(temp_value)
        self.check_local_volume(action.parent_node)

        ##Add
        kd_node = self.kd_tree.add(state, node)
        

        self.total_volume == self.root_node.children_density()
        # assert kd_node.points[0][-1] == node
        try: 
            node.children_density()
            assert node.children_volume is not None
            assert node.local_volume is not None
        except: 
            node.set_new_local_volume(node.state, node.state, 0.00000001, 
                self.model.density(
                    # torch.tensor(node.state), torch.tensor(node.state)
                    # torch.tensor(node.state)
                    torch.tensor(self.root_node.state), torch.tensor(node.state)
                ).sum().detach().item()
            )
        assert node.local_volume is not None

        return node


class MCTS_HER(MCTS_One_Shot_2):
    def return_results(
        self, final_selection: str, node=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(node.child_actions) >= 1:
            policy_dict = super().return_results(final_selection, node)
        else:
            policy_dict = {}
        # mcts_env = self.env_copy(self.Env)
        # mcts_env.restore_save_state(node.saved_state)
        # policy_dict['raw_obs'] = mcts_env.get_goal_obs()
        policy_dict['raw_obs'] = self.Env.get_goal_obs()
        policy_dict['her_reward'] = 0
        return policy_dict

    def preprocess_return_results(self):
        self.root_node.depth = 0
        self.root_node.base_prob = 1
        self.preprocess_return_results_helper(self.root_node)

    def preprocess_return_results_helper(self, node):
        policy = calculate_one_shot_policy(node, self.global_n, 
            self.epoch, self.gamma**self.depth,              
            with_pw=True, global_total_volume=self.total_volume)
            
        node.prob = (1-self.gamma + self.gamma*policy[-1])*node.base_prob
        # node.prob = (policy[-1])*node.base_prob
        node.density = node.prob/node.local_unweighted_density()

        if len(node.child_actions) > 0:
            node.new_value = node.r + policy[-1]*node.r/(1-self.gamma) + self.gamma*sum(
            [
                policy[i]*action.Q
                for i, action in enumerate(node.child_actions)
            ])
        else: 
            node.new_value = node.V#node.r/(1-self.gamma)

        for i, action in enumerate(node.child_actions):
            next_node = action.child_node
            next_node.base_prob = node.base_prob*policy[i]
            self.preprocess_return_results_helper(next_node)

        if len(node.child_actions) == 0:
            node
