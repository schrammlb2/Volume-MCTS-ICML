from __future__ import print_function,division
from six import iteritems

from ..spaces.configurationspace import *
from ..spaces.controlspace import *
from ..spaces.edgechecker import *
from ..spaces.sampler import *
from ..spaces import metric
from ..structures.nearestneighbors import *
from ..structures.kdtree import KDTree, KDTreePolicy, KDTreePolicyAlt
from ..structures import kdtree
from .kinodynamicplanner import Profiler,TreePlanner,RandomControlSelector,popdefault

import math
import ipdb

infty = float('inf')
def is_square(n):
    return math.isqrt(n)**2 == n


def argmax(lst, eval_fn, return_max=False):
    val_list = [eval_fn(e) for e in lst]
    best_i = -1
    best_v = -infty 
    for i, v in zip(lst, val_list):
        if v > best_v:
            best_v = v
            best_i = i
    if return_max:
        return (best_i, best_v)
    else: 
        return best_i


class MCTSNode:
    """A node of a kinodynamic tree"""
    def __init__(self,x, cost_est, gamma, successful=False):
        self.x = x
        # self.uparent = uparent
        # self.eparent = eparent
        self.parent = None
        self.children = []
        # self.rollout_fn = rollout_fn
        # self.controlspace = controlspace

        self.cost_sum = cost_est
        self.n = 1
        self.gamma = gamma
        self.successful = successful
        self.c = 0

    def cost_ave(self):
        return self.cost_sum/self.n

    def destroy(self):
        """Call this to free up the memory stored by this sub tree."""
        if self.parent:
            self.parent.children.remove(self)
            self.parent = None
        for c in self.children:
            c.parent = None
            c.destroy()
        self.children = []

    def unparent(self):
        """Detatches this node from its parent"""
        if self.parent:
            self.parent.children.remove(self)
            self.parent = None

    def addChild(self,c):
        """Adds a child"""
        assert c.parent is None
        c.parent = self
        self.children.append(c)

    def setParent(self,parent, uparent, eparent, parent_cost):
        """Sets the parent of this node to p"""
        # parent.addChild(self)
        self.parent = parent
        self.uparent = uparent
        self.eparent = eparent
        self.parent_cost = parent_cost
        self.c = parent_cost + parent.c
        assert self.c < infty
        # parent.children.append(self)

    def update_volume(self, v_new):
        assert False, "Not Implemented yet"
        pass


    def getPath(self,n):
        """Returns a state-control pair ([x0,...,xn],[u1,...,un])"""
        pathToStart = []
        while n is not None:
            pathToStart.append(n)
            n = n.parent
        pathToGoal = pathToStart
        pathToGoal.reverse()
        # ipdb.set_trace()
        return ([n.x for n in pathToGoal],[n.uparent for n in pathToGoal[1:]])

def mean(lst):
    sum_v = [0, 0]
    if len(lst) == 0: return sum_v
    for elem in lst:
        sum_v[0] += elem[0]
        sum_v[1] += elem[1]

    return [sum_v[0]/len(lst), sum_v[1]/len(lst)]


class MCTSPlanner(TreePlanner):
    def __init__(self, controlSpace,objective,metric,edgeChecker):
        self.controlSpace = controlSpace
        self.objective = objective
        self.edgeChecker = edgeChecker
        self.metric = metric
        self.controlSelector = RandomControlSelector(controlSpace,self.metric,1)
        self.configurationSampler = Sampler(self.controlSpace.configurationSpace())    
        self.goalNodes = []

        self.gamma = 0.9
        self.max_cost = 200
        self.n = 1
        self.num_nodes = 1

        self.bestPathCost = infty
        self.bestPath = None
        self.stats = Profiler()
        self.verbose = False#True
        self.bound_var = 1/(1-self.gamma)
        self.lambda_constant = 0.1

    def bound(self, c):
        # return min(c, np.log(self.n) + self.bound_var)
        return min(c, self.bound_var)

    def setBoundaryConditions(self,x0,goal):
        self.goal = goal
        try: 
            self.goal_center = goal.c
        except:
            self.goal_center = mean([goal.sample() for _ in range(100)])

        self.root = self.new_node_constructor(x0, self.rollout(x0), self.gamma)

    def setControlSelector(self,selector):
        self.controlSelector = selector
        # pass

    def setConfigurationSampler(self,sampler):
        self.configurationSampler = sampler

    def rollout(self, x):
        # return (np.log(self.n+1) + self.bound_var/2)*self.metric(x, self.goal_center)
        return self.metric(x, self.goal_center)

    def ucb(self, parent_node, child_node):
        lam = self.lambda_constant*parent_node.n**0.5
        return -child_node.cost_ave() + lam/child_node.n

    def planMore(self, n):
        for i in range(n):
            xrand = self.configurationSampler.sample()
            # ipdb.set_trace()
            self.mcts_expand(self.root, xrand)

    def should_branch(self, node, xrand):
        if is_square(node.n):
            return True
        else:
            return False

    # def should_branch(self, node, xrand):
    #     if node.n >= 2**len(node.children):
    #         return True
    #     else:
    #         return False

    def discount(self, cost_est, parent_cost):
        return parent_cost + self.gamma*cost_est

    def mcts_expand(self, node, xrand=None):
        if self.should_branch(node, xrand) or len(node.children) == 0:
        # if is_square(len(node.children)+1):
            cost_est, new_node = self.expand_leaf(node, xrand)
        else:
            assert len(node.children) > 0
            # next_node = argmax(node.children, self.ucb)            
            next_node = argmax(node.children, lambda child_node: self.ucb(node, child_node))
            cost_est, new_node = self.mcts_expand(next_node, xrand)

                    #TODO: change to incorporate gamma

        cost_est = self.bound(cost_est)
        node.cost_sum += cost_est
        node.n += 1
        self.n += 1        
        if node.parent is not None:
            parent_cost_est = self.discount(cost_est, node.parent_cost)
        else: 
            parent_cost_est = cost_est

        return parent_cost_est, new_node

    def allNodes(self):
        return self.allSubnodes(self.root)

    def allSubnodes(self, node):
        lst = [node.x]
        for child in node.children:
            lst += self.allSubnodes(child)
        return lst

    def addEdge(self, parent_cost, cost_est, successful, parent, u,edge):
        """Adds an edge to the tree"""
        nnew = self.new_node_constructor(edge.end(), cost_est, self.gamma, successful=successful)
        # pre_len = len(self.allNodes())
        parent.addChild(nnew)
        # mid_len = len(self.allNodes())
        nnew.setParent(parent,u,edge, parent_cost)
        # post_len = len(self.allNodes())
        # if pre_len
        # if pre_len == post_len: 
        # ipdb.set_trace()
        # self.nodes.append(nnew)
        return nnew

    def new_node_constructor(self, *args, **kwargs):
        return MCTSNode(*args, **kwargs)

    def expand_leaf(self, node, xrand):
        feasible = False
        # while not feasible:
        u = self.controlSelector.select(node.x,xrand)
        # ipdb.set_trace()
        edge = self.controlSpace.interpolator(node.x,u)
        if not self.edgeChecker.feasible(edge):
            if self.verbose: print("No new node")
            # alpha = 0.75
            alpha = 0.5
            return node.cost_ave() + 0.1, None
            # return self.max_cost, None
            return node.cost_ave()*alpha + self.max_cost*(1-alpha), None

        end = edge.end()
        cost_est = self.rollout(end)
        # ipdb.set_trace()

        if self.verbose: print(f"New node at {end}")
            # feasible = self.edgeChecker.feasible(edge)
        # newcost = node.c + self.objective.incremental(node.x,u)
        if self.goal.contains(node.x):
            successful = True
            self.goalNodes.append(node)
            if node.c < self.bestPathCost:
                self.bestPathCost = node.c
                self.bestPath = self.getPath(node)
                print(f"New path with cost {self.bestPathCost}")
                # return True
        else: 
            successful = False
        parent_cost = self.objective.incremental(node.x,u)
        new_node = self.addEdge(parent_cost, cost_est, successful, node, u, edge)

        self.num_nodes += 1
        try:
            assert self.num_nodes == len(self.allNodes())
        except:
            ipdb.set_trace()

        return parent_cost + cost_est, new_node

    def getBestPath(self):
        return self.bestPath

    def getPath(self, n = None):
        if n == None: 
            return None
        return self.root.getPath(n)    

    def reset(self):
        self.root.destroy()
        self.n = 1
        self.num_nodes = 1
        self.num_samples = 1
        self.bestPathCost = infty
        self.bestPath = None

    def getRoadmap(self):
        ret_val = super().getRoadmap()
        return ret_val

# class KernelDensityMCTS(MCTSPlanner):



class VolumeMCTSNode(MCTSNode):
    def __init__(self, x, cost, gamma, num_samples, successful=False, density_method='cumulative'):
        super().__init__(x, cost, gamma, successful=False)
        self.local_samples = 0
        self.children_samples = 0
        self.global_samples_at_time_of_creation = num_samples
        self.last_time = num_samples
        self.last_local_time = num_samples
        self.window = 1
        self.density_method = density_method
        # self.density_method = 'time_dropoff'
        self.density_method = "window"

    def local_density(self, curr_time):
        return self.estimate_density(curr_time, self.last_local_time, self.local_samples)

    def children_density(self, curr_time):
        return self.estimate_density(curr_time, self.last_time, self.local_samples + self.children_samples)
        

    def estimate_density(self, curr_time, last_time, samples):
        if self.density_method == 'cumulative':
            return samples/(curr_time - self.global_samples_at_time_of_creation + 1)
        elif self.density_method == 'time_dropoff':
            return 1/(curr_time - last_time + 1)
        elif self.density_method == 'window':
            return 1 if (curr_time - last_time) < self.window else 0
        elif self.density_method == 'mix':
            return 1 if (curr_time - last_time) < self.window else 0.01*samples/(curr_time - self.global_samples_at_time_of_creation + 1)
        else: 
            assert False, f"Density method \"{self.density_method}\" is invalid"




    # def children_density(self, curr_time):
    #     last_time = self.last_time
    #     if self.density_method == 'cumulative':
    #         return (self.local_samples + self.children_samples)/(curr_time - self.global_samples_at_time_of_creation)

    # def local_density(self, curr_time):
    #     return 1 if curr_time == self.last_local_time else 0#1#0.1

    # def children_density(self, curr_time):
    #     return 1 if curr_time == self.last_time else 0#1#0.1

    # def local_density(self, curr_time):
    #     return 1 if (curr_time - self.last_local_time) < self.window else 0.01

    # def children_density(self, curr_time):
    #     return 1 if (curr_time - self.last_time) < self.window else 0.01

    # def local_density(self, curr_time):
    #     return 1/(curr_time - self.last_local_time + 1)**2

    # def children_density(self, curr_time):
    #     return 1/(curr_time - self.last_time + 1)**2

    def num_samples(self):
        return self.local_samples + self.children_samples

    def add_sample(self, curr_time):
        self.local_samples += 1
        self.last_time = curr_time
        self.last_local_time = curr_time
        if self.parent != None:
            self.parent.prop_sample(curr_time)

    def prop_sample(self, curr_time):
        self.children_samples += 1
        self.last_time = curr_time
        if self.parent != None:
            self.parent.prop_sample(curr_time)



class VolumeMCTS(MCTSPlanner):
    def __init__(self, controlSpace,objective,metric,edgeChecker):
        super().__init__(controlSpace,objective,metric,edgeChecker)
        self.nearestNeighbors = KDTree()
        self.num_samples = 1
        self.lambda_constant = 1#0.1
        c = 0#0.2
        # self.volume_const = 0#0.0001#c#0.001#00.00001#0.00001#1#0.01
        # self.cost_const = 0#c#0#0.1#0.0000001#0.001#0.01#1
        self.volume_const = c#0.0001#c#0.001#00.00001#0.00001#1#0.01
        self.cost_const = c#0#0.1#0.0000001#0.001#0.01#1
        self.gamma = .9

    def rollout(self,x):
        weight = 2
        direct = weight*self.metric(x, self.goal_center)
        # return direct
        nn_list = self.nearestNeighbors.knearest(x, 5)
        if len(nn_list) > 0:
            c_bound = min([nn.cost_ave() + weight*self.metric(x, nn_x) for nn_x, nn in nn_list])
        else: 
            c_bound = 1/(1-self.gamma)
        return min(direct, c_bound)

    def setBoundaryConditions(self,x0,goal):
        super().setBoundaryConditions(x0, goal)
        self.nearestNeighbors.add(self.root.x, self.root)

    def new_node_constructor(self, x, cost, gamma, successful=False):
        return VolumeMCTSNode(x, cost, gamma, self.num_samples, successful=False)

    def lam(self, n):
        offset_const = 1000
        power = 0.5
        return (n + offset_const)**power/(offset_const)**power

    def ucb(self, parent_node, child_node):
        volume = self.lambda_constant*child_node.children_density(self.num_samples) + self.volume_const
        return -child_node.cost_ave()*self.cost_const + volume*self.lam(parent_node.n)/child_node.n

    def broadening_ucb(self, node):
        child_n = len(node.children) 
        volume = self.lambda_constant*node.local_density(self.num_samples) + self.volume_const
        return -node.cost_ave()*self.cost_const + volume*self.lam(node.n)/child_n

    def should_branch(self, node, xrand):
        if len(node.children) == 0: 
            return True
        ucb_list = [self.ucb(node, child_node) for child_node in node.children]
        ucb_max = max(ucb_list)
        broad_ucb = self.broadening_ucb(node)
        if broad_ucb >= ucb_max:
            return True
        else: 
            return False

    def planMore(self, n):
        for i in range(n):
            xrand = self.configurationSampler.sample()
            for i in range(1):
                node_x, node = self.nearestNeighbors.nearest(xrand)
                if np.random.rand() < 0.01: 
                    xrand = self.goal_center
                node.prop_sample(self.num_samples)
            # ipdb.set_trace()
            cost, new_node = self.mcts_expand(self.root, xrand)
            # cost, new_node = self._sample(self.root, xrand)
            if new_node is not None: 
                self.nearestNeighbors.add(new_node.x, new_node)
            self.num_samples += 1


    def _sample(self, node, xrand):
        if len(node.children) == 0:
            cost_est, new_node = self.expand_leaf(node, xrand)
        else:
            c = [child.cost_ave() for child in node.children]
            v = [child.children_density(self.num_samples) for child in node.children]

            local_c = node.cost_ave()
            local_v = node.local_density(self.num_samples)

            all_c = [local_c] + c
            all_v = [local_v] + v

            alpha_lower_bound = min(all_c) - self.lam(node.n)
            alpha_upper_bound = min([c_i - self.lam(node.n)*v_i for c_i, v_i in zip(all_c, all_v)])

            # alpha = alpha_upper_bound
            alpha = alpha_lower_bound
            for i in range(1):
                pi = [self.lam(node.n)*(v_i + self.volume_const)/(c_i - alpha) for c_i, v_i in zip(all_c, all_v)]
                total_pi = sum(pi)
                # if total_pi < 1: 
                #     alpha_upper_bound = (alpha_upper_bound + alpha_lower_bound)/2
                # else:
                #     alpha_lower_bound = (alpha_upper_bound + alpha_lower_bound)/2
            pi = [pi_i/total_pi for pi_i in pi]
            if np.random.rand() < pi[0] or len(node.children) == 0:
                cost_est, new_node = self.expand_leaf(node, xrand)
            else: 
                total_pi = sum(pi[1:])
                renormed_pi = [pi_i/total_pi for pi_i in pi[1:]]
                next_node = np.random.choice(node.children, p=renormed_pi)
                cost_est, new_node = self._sample(next_node, xrand)


        cost_est = self.bound(cost_est)
        node.cost_sum += cost_est
        node.n += 1
        self.n += 1        
        if node.parent is not None:
            parent_cost_est = self.discount(cost_est, node.parent_cost)
        else: 
            parent_cost_est = cost_est

        return parent_cost_est, new_node


class KDVolumeMCTSNode(MCTSNode):
    def __init__(self, x, cost, gamma, successful=False):
        super().__init__(x, cost, gamma, successful=False)
        # self.local_volume
        # self.children_volume = 0

    def local_density(self, extra=None):
        return self.local_volume

    def children_density(self, extra=None):
        if len(self.children) > 0:
            return self.children_volume
        else: 
            return self.local_volume

    def set_new_local_volume(self, volume):
        if len(self.children) == 0:
            self.local_volume = volume

    def set_new_volume(self, volume):
        self.local_volume = volume
        self._pass_volume()

    def _pass_volume(self):
        if len(self.children) > 0:
            self.children_volume = sum((c.children_density() for c in self.children)) + self.local_density()
        else: 
            self.children_volume = self.local_density()
        if self.parent != None:
            self.parent._pass_volume()


class KDVolumeMCTS(VolumeMCTS):
    def __init__(self, controlSpace,objective,metric,edgeChecker):
        super().__init__(controlSpace,objective,metric,edgeChecker)
        c = 0.2
        self.volume_const = 0.0#c#0.0001#c#0.001#00.00001#0.00001#1#0.01
        self.cost_const = 0.01#0#c#0#0.1#0.0000001#0.001#0.01#1

    def new_node_constructor(self, x, cost, gamma, successful=False):
        return KDVolumeMCTSNode(x, cost, gamma, successful=False)

    def setBoundaryConditions(self,x0,goal):
        super().setBoundaryConditions(x0, goal)
        # self.nearestNeighbors = KDTree()
        self.nearestNeighbors = KDTreePolicy(distanceMetric=self.metric,
            cspace=self.controlSpace.configurationSpace(), goal=self.goal)

        # try: 
        #     lo, hi = self.controlSpace.configurationSpace().box.bounds()
        # except:
        #     lo, hi = self.controlSpace.configurationSpace().bounds()
        # self.nearestNeighbors = KDTreePolicyAlt(distanceMetric=self.metric,
        #     lo = lo, hi = hi)
        self.nearestNeighbors.pass_volume = True
        self.nearestNeighbors.track_cost = False
        self.add_and_update_volume(self.root)

    def add_and_update_volume(self, node):
        new_kd_node = self.nearestNeighbors.add(node.x, node)
        # volume = new_kd_node.

    def planMore(self, n):
        for i in range(n):
            xrand = self.configurationSampler.sample()
            # for i in range(10):
            #     node_x, node = self.nearestNeighbors.nearest(xrand)
            #     # if np.random.rand() < 0.01: 
            #     #     xrand = self.goal_center
            #     node.prop_sample(self.num_samples)
            cost, new_node = self.mcts_expand(self.root, xrand)
            # cost, new_node = self._sample(self.root, xrand)
            if new_node is not None: 
                self.nearestNeighbors.add(new_node.x, new_node)
            self.num_samples += 1

    def bound(self, c): return c

    def discount(self, cost_est, parent_cost):
        return parent_cost + cost_est

    def expand_leaf(self, node, xrand):
        if np.random.rand() < 0.1:
            pass
        else:
            x = node.x
            kdnode = self.nearestNeighbors.locate(x, False)
            xsample = np.array(self.nearestNeighbors._sample_node(kdnode))
            # xnoise = np.random.normal(0, 0.25/np.log(node.n + 1), size=(len(xrand),))
            xnoise = np.random.normal(0, 0.1, size=(len(xrand),))
            xrand = (xsample + xnoise).tolist()
        return super().expand_leaf(node, xrand)

    def ucb(self, parent_node, child_node):
        volume = (self.lambda_constant*child_node.children_density(self.num_samples)
            # /parent_node.children_density(self.num_samples) 
            + self.volume_const)
        return -child_node.cost_ave()*self.cost_const + volume*self.lam(parent_node.n)/child_node.n

    def broadening_ucb(self, node):
        child_n = len(node.children) 
        volume = (self.lambda_constant*node.local_density(self.num_samples)
            # /node.children_density(self.num_samples) 
            + self.volume_const)
        return -node.cost_ave()*self.cost_const + volume*self.lam(node.n)/child_n