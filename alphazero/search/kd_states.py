import numpy as np
from alphazero.search.states import (
    Action,
    Node,
    ActionContinuous,
    ActionDiscrete,
    NodeContinuous,
    NodeDiscrete,
)
import ipdb

class KDNodeContinuous(NodeContinuous):	
    # state: np.ndarray
    # r: float
    # terminal: bool
    # parent_action: Optional[KDActionContinuous]
    # n: int
    # V: float
    # child_actions: List[KDActionContinuous]
    def local_density(self):
        return self.local_volume

    def local_unweighted_density(self):
        return self.local_unweighted_volume

    def local_inv_density(self):
        return self.local_inv_volume

    def children_density(self):
        return self.children_volume

    def children_unweighted_density(self):
        return self.children_unweighted_volume
        
    def children_inv_density(self):
        return self.children_inv_volume

    def set_new_local_volume(self, lo, hi, volume, density):
        try: 
            assert volume >= 0
        except:
            import ipdb
            ipdb.set_trace()
        self.lo = lo
        self.hi = hi
        self.density = 1#density
        if len(self.child_actions) == 0:
            self.local_volume = volume*self.density
            self.local_unweighted_volume = volume
            self.local_inv_volume = volume/self.density
            self.children_volume = self.local_volume
            self.children_unweighted_volume = self.local_unweighted_volume
            self.children_inv_volume = self.local_inv_volume

    def set_new_volume(self, lo, hi, volume, density):
        assert volume >= 0
        self.lo = lo
        self.hi = hi
        self.density =  1#density
        self.local_volume = volume*self.density
        self.local_unweighted_volume = volume
        self.local_inv_volume = volume/self.density
        self._pass_volume()

    def remove_action(self, target_action):
        self.child_actions = [a for a in self.child_actions if a != target_action]


    def _pass_volume(self):
        if len(self.child_actions) > 0:
            self.children_volume = sum((c.children_density() for c in self.child_actions)) + self.local_density()
            self.children_unweighted_volume = sum((c.children_unweighted_density() for c in self.child_actions)) + self.local_unweighted_density()
            self.children_inv_volume = sum((c.children_inv_density() for c in self.child_actions)) + self.local_inv_density()
        else: 
            self.children_volume = self.local_density()
            self.children_unweighted_volume = self.local_unweighted_density()
            self.children_inv_volume = self.local_inv_density()
        try: 
            assert self.children_volume > 0
        except: 
            ipdb.set_trace()
        if self.parent_action != None:
            self.parent_action._pass_volume()

class KDActionContinuous(ActionContinuous):
	# action: np.ndarray
 #    parent_node: KDNodeContinuous
 #    W: float
 #    n: int
 #    Q: float
 #    child_node: KDNodeContinuous
    def __init__(self, 
            action: np.ndarray, 
            parent_node: NodeContinuous, 
            Q_init: float, 
            log_prob: float):
        super().__init__(action, parent_node, Q_init)
        self.log_prob = log_prob

    def children_density(self, extra=None):
            # return self.child_node.children_density()
        try: 
            return self.child_node.children_density()
        except: 
            import ipdb
            ipdb.set_trace()

    def children_unweighted_density(self, extra=None):
        return self.child_node.children_unweighted_density()

    def children_inv_density(self, extra=None):
        return self.child_node.children_inv_density()

    def _pass_volume(self):
        self.parent_node._pass_volume()

    def add_child_node(
        self, state: np.ndarray, r: float, terminal: bool
    ) -> NodeContinuous:

        self.child_node = KDNodeContinuous(state, r, terminal, self)
        return self.child_node

    @property
    def has_children(self) -> bool:
        if hasattr(self, "child_node") and self.child_node is not None:
            return True
        else: 
            return False

    def delete_self(self):
        self.parent_node.remove_action(self)
        del self
    # def update(self, R, kd_R, kd_n) -> None:
    #     """Updates this action during the MCTS backup phase.

    #     The following steps are performed during the update:
    #         - Incrementation of the visitation count of this instance.
    #         - Adding the accumulated discounted reward of this trace to the action.
    #         - Update of the action-value with the new cumulative reward.

    #     Parameters
    #     ----------
    #     R: float
    #         Accumulated reward of the current search trace.
    #     """
    #     self.n += 1
    #     self.W += R
    #     self.Q = self.W / self.n


class PWAction(KDActionContinuous):
    def __init__(self, node, n=None, forced_volume=0):
        self.node = node
        self.Q = node.V
        if len(node.child_actions) > 0: 
            self.Q = sum((a.Q for a in node.child_actions))/len(node.child_actions)
            self.log_prob = np.log(1/(len(node.child_actions)+1))
        else: 
            self.Q = node.V
        if n == None:
            self.n = 1
        else:
            self.n = n
        self.n = 1 + len(node.child_actions)
        self.forced_volume = forced_volume

    def children_density(self):
        return self.node.local_density() + self.forced_volume

    def children_unweighted_density(self):
        return self.node.local_unweighted_density()

    def children_inv_density(self):
        return self.node.local_inv_density() + self.forced_volume