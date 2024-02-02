from __future__ import annotations
from typing import List, Optional, Union, cast

from alphazero.search.kd_tree import KDTree, Node, KDTreePolicyAlt, KDTreeNode
from copy import deepcopy
import numpy as np
import torch

from alphazero.search.states import NodeContinuous, ActionContinuous
from alphazero.search.poly_hoo import POLY_HOO

#MCTS Node definition

class HOOTNode(NodeContinuous):
	def __init__(
	    self,
	    state: np.ndarray,
	    r: float,
	    terminal: bool,
	    parent_action: Optional[ActionContinuous],
	    dim, 
	    device,
	):
		assert type(dim) is int
		super().__init__(state, r, terminal, parent_action)
		self.device = device


		self.children = {}
		self.immediate_reward = 0
		self.dim = dim

		# self.hoot_tree = HOOTTree(self, lo, hi)
		rho = 2**(-2 / dim)
		nu = 4 * dim
		# alphas = [2.5 for _ in range(MAX_MCTS_DEPTH + 1)]
		# xis = [10.0 for _ in range(MAX_MCTS_DEPTH + 1)]
		# etas = [0.5 for _ in range(MAX_MCTS_DEPTH + 1)]		
		alpha = 2.5
		xi = 10.0
		eta = 0.5 
		HOO_LIMIT_DEPTH = 2
		self.hoo = POLY_HOO(dim=dim, nu=nu, rho=rho, min_value=-1, max_value=1, 
			lim_depth=HOO_LIMIT_DEPTH, alpha=alpha, xi=xi, eta=eta)

    # def select(self):
    #     self.hoot_tree.select(allow_expansion=True)

	def selection(self, depth):
		if self.is_done or depth >= MAX_MCTS_DEPTH:
			return 0
		raw_action = self.hoo.select_action().tolist()
		action = [round(a, KEY_DECIMAL) for a in raw_action]
		if tuple(action) in self.children:
			child = self.children[tuple(action)]
			immediate_reward = child.immediate_reward
			value = child.selection(depth + 1)
			self.hoo.update(value + immediate_reward)
			return immediate_reward + value
		else:
			snapshot, obs, immediate_reward, is_done, _ = env.get_result(self.snapshot, action)
			#create action here
			child = HOOTNode(snapshot, obs, is_done, self, depth + 1, self.dim)
			child.immediate_reward = immediate_reward
			self.children[tuple(action)] = child 
			value = child.selection(depth + 1)
			self.hoo.update(value + immediate_reward)
			return immediate_reward + value


	def delete(self, node):
		for action in node.children:
			node.delete(node.children[action])
		del node 

class HOOTAction(ActionContinuous):
    def add_child_node(
        self, state: np.ndarray, r: float, terminal: bool
    ) -> NodeContinuous:
        self.child_node = HOOTNode(state, r, terminal, 
            self, dim=len(self.action.tolist()), device=self.parent_node.device)
        return self.child_node


    def update(self, R: float) -> None:
        self.n += 1
        self.W += R
        self.Q = self.W / self.n