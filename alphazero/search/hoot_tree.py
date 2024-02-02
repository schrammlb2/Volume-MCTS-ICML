from __future__ import annotations
from typing import List, Optional, Union, cast

from alphazero.search.kd_tree import KDTree, Node, KDTreePolicyAlt, KDTreeNode
from copy import deepcopy
import numpy as np
import torch

from alphazero.search.states import NodeContinuous, ActionContinuous

#MCTS Node definition

class HOOTNode(NodeContinuous):
    def __init__(
        self,
        state: np.ndarray,
        r: float,
        terminal: bool,
        parent_action: Optional[ActionContinuous],
        lo: np.ndarray,
        hi: np.ndarray,
        device,
    ):
        super().__init__(state, r, terminal, parent_action)
        self.device = device
        self.hoot_tree = HOOTTree(self, lo, hi)

    def select(self):
        self.hoot_tree.select(allow_expansion=True)

class HOOTAction(ActionContinuous):
    def add_child_node(
        self, state: np.ndarray, r: float, terminal: bool
    ) -> NodeContinuous:
        self.child_node = HOOTNode(state, r, terminal, 
            self, self.parent.lo, self.parent.hi, self.parent.device)
        return self.child_node


    def update(self, R: float) -> None:
        self.n += 1
        self.W += R
        self.Q = self.W / self.n


#--------------------------------------------------------------------------
#KDTree definition

# class HOOTTree(KDTree): 
# 	def __init__(self, mcts_node,
# 			env_lo, env_hi, 
# 		):
# 		super().__init__()
# 		self.lo = env_lo
# 		self.hi = env_hi
# 		self.children = []
# 		self.max_depth = 2
# 		self.mcts_node = mcts_node
#     # def makeNode(self, pts, splitdim=None):
#     #     if splitdim == None:
#     #         return HOOTTreeNode(pts)
#     #     else:
#     #         return HOOTTreeNode(pts, splitdim=splitdim)
# 	def makeNode(self, parent):
# 		if splitdim == None:
# 			new_node = HOOTTreeNode(parent, points, lo, hi, env_lo, env_hi,
# 			splitdim=0, root_state=root_state)
# 		else:
# 			new_node = HOOTTreeNode(parent, points, lo, hi, env_lo, env_hi,
# 			splitdim=splitdim, root_state=root_state)

# 		self.children.append(new_node)
# 		return new_node

# 	def rebalance(self):
# 		#This should not happen
# 		import ipdb
# 		ipdb.set_trace()

# 	# def select(self):
# 	# 	if node == None: 
# 	# 		node = self.root

# 	# 	if node.splitvalue == None: assert len(node.points) == 0
# 	# 	if len(node.points) == 0: assert node.splitvalue == None

# 	def backprop(self, loc, V):
# 		node = self.locate(loc, inc=False)

# 		node.update(V)
# 		while node.parent is not None:
# 			node = node.parent
# 			node.update(V)

# 	def select(self, allow_expansion=False):
# 		if self.root == None: 
# 			action_vec = np.random.uniform(self.lo, self.hi)
# 			action_node = self.create_pw_action(action_vec)
# 			self.add(action_vec.tolist(), action_node)
# 			return action_node

# 		depth = 0
# 		c=1
# 		hoot_node = self.root
# 		while hoot_node.splitvalue != None and depth < self.max_depth:
# 			depth += 1
# 			N = hoot_node.visitation_count
# 			left_val = hoot_node.left.ucb(N, c)
# 			right_val = hoot_node.right.ucb(N, c)
# 			if left_val > right_val: 
# 				hoot_node=hoot_node.left
# 			else: 
# 				hoot_node=hoot_node.right

# 		if depth >= self.max_depth or not allow_expansion:
# 			#Hit max depth, follow current node
# 			assert len(hoot_node.points) == 1
# 			return hoot_node.points[0][-1]
# 		else: 
# 			# Create New 
# 			action_vec = self.sample_action(hoot_node)
# 			action_node = self.create_pw_action(action_vec)
# 			self.add(action_vec.tolist(), action_node)
# 			return action_node


# 	def sample_action(self, hoot_node):
# 		return  np.random.uniform(hoot_node.lo, hoot_node.hi)

# 	def create_pw_action(self, action_vec) -> None:
# 		state = (
# 			torch.from_numpy(
# 			    self.mcts_node.state[
# 			        None,
# 			    ]
# 			)
# 			.float()
# 			.to(self.mcts_node.device)
# 		)
# 		action_vec = action_vec.reshape(-1)#.squeeze()
# 		action_node = HOOTAction(action_vec, 
# 			parent_node=self.mcts_node, Q_init=self.mcts_node.V)
# 		return action_node


# class HOOTTreeNode(Node):
# 	def __init__(self, parent, points, lo, hi, env_lo, env_hi,
# 			splitdim=0, root_state=None):
# 		super().__init__(points,splitdim=splitdim)
# 		self.epsilon = 10**(-6)
# 		self.reward_sum = 0
# 		self.visitation_count = 0 + self.epsilon

# 		self.parent = parent

# 		self.lo = lo
# 		self.hi = hi

# 		self.v = 1
# 		self.rho = 1/2

# 	def ucb(self, N, c):
# 		value = self.reward_sum/self.visitation_count
# 		B = self.visitation_count**0.5
# 		rho_v = self.v*self.rho**self.depth
# 		return value + c*N**0.5*B + rho_v
# 		raise NotImplemented


class HOOTTree: 
	def __init__(self, mcts_node, env_lo, env_hi,):
		super().__init__()
		self.lo = env_lo
		self.hi = env_hi
		self.children = []
		self.max_depth = 2
		self.mcts_node = mcts_node


	def makeNode(self, parent):
		if splitdim == None:
			new_node = HOOTTreeNode(parent, points, lo, hi, env_lo, env_hi,
			splitdim=0, root_state=root_state)
		else:
			new_node = HOOTTreeNode(parent, points, lo, hi, env_lo, env_hi,
			splitdim=splitdim, root_state=root_state)

		self.children.append(new_node)
		return new_node

	def rebalance(self):
		#This should not happen
		import ipdb
		ipdb.set_trace()

	# def select(self):
	# 	if node == None: 
	# 		node = self.root

	# 	if node.splitvalue == None: assert len(node.points) == 0
	# 	if len(node.points) == 0: assert node.splitvalue == None

	def backprop(self, loc, V):
		node = self.locate(loc, inc=False)

		node.update(V)
		while node.parent is not None:
			node = node.parent
			node.update(V)

	def select(self, allow_expansion=False):
		if self.root == None: 
			action_vec = np.random.uniform(self.lo, self.hi)
			action_node = self.create_pw_action(action_vec)
			self.add(action_vec.tolist(), action_node)
			return action_node

		depth = 0
		c=1
		hoot_node = self.root
		while hoot_node.splitvalue != None and depth < self.max_depth:
			depth += 1
			N = hoot_node.visitation_count
			left_val = hoot_node.left.ucb(N, c)
			right_val = hoot_node.right.ucb(N, c)
			if left_val > right_val: 
				hoot_node=hoot_node.left
			else: 
				hoot_node=hoot_node.right

		if depth >= self.max_depth or not allow_expansion:
			#Hit max depth, follow current node
			assert len(hoot_node.points) == 1
			return hoot_node.points[0][-1]
		else: 
			# Create New 
			action_vec = self.sample_action(hoot_node)
			action_node = self.create_pw_action(action_vec)
			self.add(action_vec.tolist(), action_node)
			return action_node


	def sample_action(self, hoot_node):
		return  np.random.uniform(hoot_node.lo, hoot_node.hi)

	def create_pw_action(self, action_vec) -> None:
		state = (
			torch.from_numpy(
			    self.mcts_node.state[
			        None,
			    ]
			)
			.float()
			.to(self.mcts_node.device)
		)
		action_vec = action_vec.reshape(-1)#.squeeze()
		action_node = HOOTAction(action_vec, 
			parent_node=self.mcts_node, Q_init=self.mcts_node.V)
		return action_node


class HOOTTreeNode:
	def __init__(self, parent, points, lo, hi, env_lo, env_hi,
			splitdim=0, root_state=None):
		super().__init__(points,splitdim=splitdim)
		self.epsilon = 10**(-6)
		self.reward_sum = 0
		self.visitation_count = 0 + self.epsilon

		self.parent = parent

		self.lo = lo
		self.hi = hi

		self.v = 1
		self.rho = 1/2

	def ucb(self, N, c):
		value = self.reward_sum/self.visitation_count
		B = self.visitation_count**0.5
		rho_v = self.v*self.rho**self.depth
		return value + c*N**0.5*B + rho_v
		raise NotImplemented