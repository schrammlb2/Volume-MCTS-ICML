from __future__ import print_function,division
from six import iteritems
# from itertools import product

import math
from .knn import *
from ..spaces import metric
# from ..klampt import so2
import numpy as np

import ipdb
#If using a non-Euclidean space, the kd tree should account for
#it.  Rather than setting the the KDTree partitionFn and minDistance
#functions by hand, you can set one of these to true and the space will
#be accounted for if it has SO2 or SE2 in its leading elements.
SO2_HACK = False
SE2_HACK = False

class Node:
    def __init__(self,points,splitdim=0):
        """Arguments:
        - points: a list of (point,data) pairs
        - splitdim: the default split dimension.
        """
        assert isinstance(points,(list,tuple))
        self.points = points
        self.splitdim = splitdim
        self.depth = 0
        self.splitvalue = None
        self.left = None
        self.right = None

        self.n = 1
        self.c_ave = 0
        self.min_bound = []
        self.max_bound = []

class KDTree:
    """
    Attributes:
    - maxPointsPerNode: allows up to this number of points in a single K-D tree
      node.
    - partitionFn: a function (d,value,x) that returns -1,0, or 1
      depending on whether the point is on the positive, negative, or on the
      plane with split value value on the d'th dimension.  By default, it's
      sign(x[d] - value).
    - minDistanceFn: a function (d,value,x) that returns the minimum
      distance to the partition plane on the d'th dimension.  By default, it's
      |x[d] - value| for the euclidean metric, and otherwise it's
      metric(set(x,d,value),x[d]) where set(x,d,value) is a point equal
      to x except for it's d'th entry set to value.  This latter case works for
      all L-p norms, weighted or unweighted.
    """
    def __init__(self,distanceMetric=metric.euclideanMetric):
        # import ipdb
        # ipdb.set_trace()
        self.root = None
        self.metric = distanceMetric
        self.partitionFn = lambda d,value,x: math.copysign(1,x[d]-value)
        self.minDistanceFn = lambda d,value,x: abs(x[d]-value)
        # if self.metric != metric.euclideanMetric:
        #     def minDistFn(d,value,x):
        #         tempp = x[:]
        #         tempp[d] = value

        #         #HACK: for pendulum
        #         if SO2_HACK and d==0:
        #             tempp2 = x[:]
        #             tempp2[d] = math.pi+value
        #             return min(self.metric(tempp,x),self.metric(tempp2,x))
        #         #HACK: for SE2
        #         if SE2_HACK and d==2:
        #             tempp2 = x[:]
        #             tempp2[d] = math.pi+value
        #             return min(self.metric(tempp,x),self.metric(tempp2,x))
        #         return self.metric(tempp,x)
        #     def partitionFn(d,value,x):
        #         #HACK: for pendulum
        #         if SO2_HACK and d == 0:
        #             math.copysign(1,so2.diff(x[d],value))
        #         #HACK: for SE2
        #         if SE2_HACK and d == 2:
        #             math.copysign(1,so2.diff(x[d],value))
        #         return math.copysign(1,x[d]-value)
        #     self.minDistanceFn = minDistFn
        #     self.partitionFn = partitionFn

        self.maxPointsPerNode = 1
        self.maxDepth = 0
        self.numNodes = 0

    def reset(self):
        self.__init__(self.metric)

    def locate(self,point):
        """Find the Node which contains a point"""
        return self._locate(self.root,point)

    def _locate(self,node,point):
        if node.splitvalue == None: return node
        if self.partitionFn(node.splitdim,node.splitvalue,point) < 0: return self._locate(node.left,point)
        else: return self._locate(node.right,point)

    def set(self,points,data):
        """Set the KD tree to contain a list of points.  O(log(n)*n*d)
        running time."""
        self.maxDepth = 0 
        self.numNodes = 10
        self.root = Node(list(zip(points,data)))
        self.recursive_split(self.root,optimize=True)

    def recursive_split(self,node,force=False,optimize=False):
        """Recursively splits the node along the best axis.
        - node: the node to split
        - force: true if you want to force-split this node.  Otherwise,
          uses the test |node.points| > maxPointsPerNode to determine
          whether to split.
        - optimize: true if you want to select the split dimension with
          the widest range.
        Returns the depth of the subtree if node is split, 0 if not split.
        """
        # force = True
        if not force and len(node.points) <= self.maxPointsPerNode:
            return 0
        if len(node.points)==0:
            return 0
        if node.left != None:
            #already split
            raise RuntimeError("Attempting to split node already split")
        d = len(node.points[0][0])
        vmin,vmax = 0,0
        if not optimize:
            #just loop through the dimensions
            for i in range(d):
                vmin = min(p[0][node.splitdim] for p in node.points)
                vmax = max(p[0][node.splitdim] for p in node.points)
                if vmin != vmax:
                    break
                #need to choose a new split dimension
                node.splitdim = (node.splitdim+1)%d
        else:
            rangemax = (0,0)
            dimmax = 0
            for i in range(d):
                vmin = min(p[0][i] for p in node.points)
                vmax = max(p[0][i] for p in node.points)
                if vmax-vmin > rangemax[1]-rangemax[0]:
                    rangemax = (vmin,vmax)
                    dimmax = i
            node.splitdim = dimmax
            vmin,vmax = rangemax
        if vmin == vmax:
            #all points are equal, don't split (yet)
            return 0
        node.splitvalue = (vmin+vmax)*0.5
        leftpts = []
        rightpts = []
        for p in node.points:
            if self.partitionFn(node.splitdim,node.splitvalue,p[0]) < 0: leftpts.append(p)
            else: rightpts.append(p)
        if len(leftpts) == 0 or len(rightpts) == 0:
            #may have numerical error
            node.splitvalue = None
            return 0
        node.left = self.makeNode(leftpts,splitdim=(node.splitdim+1)%d)
        node.right = self.makeNode(rightpts,splitdim=(node.splitdim+1)%d)
        node.left.depth = node.depth+1
        node.right.depth = node.depth+1
        self.numNodes += 2
        self.maxDepth = max(self.maxDepth,node.depth+1)
        d1 = self.recursive_split(node.left,force=False,optimize=optimize)
        d2 = self.recursive_split(node.right,force=False,optimize=optimize)
        node.points = []
        return 1+max(d1,d2)

    def makeNode(self, pts, splitdim=None):
        if splitdim == None:
            return Node(pts)
        else:
            return Node(pts, splitdim=splitdim)

    def add(self,point,data):
        """Add a point to the KD tree (O(log(n)) running time)"""
        if self.root == None:
            self.root = self.makeNode([(point,data)])
            self.maxDepth = 0
            self.numNodes = 1
            return_value = self.root
        else:
            node = self.locate(point)
            node.points.append((point,data))
            if self.recursive_split(node,optimize=False):
                return_value =  self._locate(node,point)
            else:
                return_value =  node

        def check_single_pt(node):
            assert len(node.points) <= 1
        # self.traverse(self.root, check_single_pt)

        return return_value

    def traverse(self, node, func):
        if node is not None:
            func(node)
            self.traverse(node.left, func)
            self.traverse(node.right, func)

    def _locate_with_parent(self,node,point):
        if node.splitvalue == None: return (node,None)
        if self.partitionFn(node.splitdim,node.splitvalue,point) < 0:
            n,p = self._locate_with_parent(node.left,point)
            if p==None: return n,node
            return n,p
        else:
            n,p = self._locate_with_parent(node.right,point)
            if p==None: return n,node
            return n,p


    def remove(self,point,data=None):
        """Removes the point from the KD-tree.  If data is given, then the
        data member is checked for a match too.  Returns the number of points
        removed. (TODO: can only be 0 or 1 at the moment)"""
        n,parent = self._locate_with_parent(self.root,point)
        if n==None: return 0
        found = False
        for i,p in enumerate(n.points):
            if point == p[0] and (data==None or data==p[1]):
                del n.points[i]
                found = True
                break
        if len(n.points)==0:
            #merge siblings back up the tree?
            if parent != None:
                if parent.left.splitvalue == None and parent.right.splitvalue == None:
                    if parent.left == n:
                        parent.points = parent.right.points
                        parent.left = parent.right = None
                    else:
                        assert parent.right==n
                        parent.points = parent.left.points
                        parent.left = parent.right = None
                    parent.splitvalue = None
        if found:
            return 1
        return 0

    def rebalance(self,force=False):
        dorebalance = force
        if not force:
            idealdepth = math.log(self.numNodes)
            #print("ideal depth",idealdepth,"true depth",self.maxDepth)
            if self.maxDepth > idealdepth*10:
                dorebalance = True
        if not dorebalance:
            return False
        print("Rebalancing KD-tree...")
        points = []
        def recurse_add_points(node):
            points += node.points
            if node.left: recurse_add_points(node.left)
            if node.right: recurse_add_points(node.right)
        recurse_add_points(self.root)
        self.set(zip(*points))
        print("Done.")
        return True

    def _nearest(self,node,x,dmin,filter=None):
        if node.splitvalue == None:
            #base case, it's a leaf
            closest = None
            for p in node.points:
                if filter != None and filter(*p):
                    continue
                d = self.metric(p[0],x)
                if d < dmin:
                    closest = p
                    dmin = d
            return (closest,dmin)
        #recursive case, it's a parent
        dhi = 0
        dlo = 0
        if self.partitionFn(node.splitdim,node.splitvalue,x) < 0:
            dlo = 0
            dhi = self.minDistanceFn(node.splitdim,node.splitvalue,x)
        else:
            dhi = 0
            dlo = self.minDistanceFn(node.splitdim,node.splitvalue,x)

        if dhi > dmin:  #only check left
            (lclosest,ld) = self._nearest(node.left,x,dmin,filter)
            return (lclosest,ld)
        elif dlo > dmin: #only check right
            (rclosest,rd) = self._nearest(node.right,x,dmin,filter)
            return (rclosest,rd)
        else:
            first,second = node.left,node.right
            if dlo > dhi:
                first,second = second,first
                dlo,dhi = dhi,dlo
            #check the closest first
            closest = None
            (fclosest,fd) = self._nearest(first,x,dmin,filter)
            if fd < dmin:
                #assert fclosest != None
                #assert fd == self.metric(fclosest[0],x)
                closest,dmin=fclosest,fd
            if dhi < dmin: #check if should prune second or not
                #no pruning, check the second next
                (sclosest,sd) = self._nearest(second,x,dmin,filter)
                if sd < dmin:
                    #assert sclosest != None
                    #assert sd == self.metric(sclosest[0],x)
                    closest,dmin=sclosest,sd
            return (closest,dmin)

    def nearest(self,x,filter=None):
        """Nearest neighbor query:
        Returns the (point,data) pair in the tree closest to the point x"""
        if self.root == None: return []
        closest,dmin = self._nearest(self.root,x,float('inf'),filter)
        if closest is None:
            input()
        return closest

    def _knearest(self,node,x,res,filter=None):
        if node.splitvalue == None:
            #base case, it's a leaf
            for p in node.points:
                if filter != None and filter(*p):
                    continue
                d = self.metric(p[0],x)
                res.tryadd(p,d)
            return res
        #recursive case, it's a parent
        dhi = 0
        dlo = 0
        if self.partitionFn(node.splitdim,node.splitvalue,x) < 0:
            dlo = 0
            dhi = self.minDistanceFn(node.splitdim,node.splitvalue,x)
        else:
            dhi = 0
            dlo = self.minDistanceFn(node.splitdim,node.splitvalue,x)

        if dhi > res.maximum_distance():  #only check left
            res = self._knearest(node.left,x,res,filter)
            return res
        elif dlo > res.maximum_distance(): #only check right
            res = self._knearest(node.right,x,res,filter)
            return res
        else:
            first,second = node.left,node.right
            if dlo > dhi:
                first,second = second,first
                dlo,dhi = dhi,dlo
            #check the closest first
            closest = None
            res = self._knearest(first,x,res,filter)
            if dhi < res.maximum_distance(): #check if should prune second
                #no pruning, check the second next
                res = self._knearest(second,x,res,filter)
            return res


    def knearest(self,x,k,filter=None):
        """K-nearest neighbor query:
        Returns the [(point1,data1),...,(pointk,datak)] in the tree
        that are closest to the point x. Results are sorted by distance."""
        if self.root == None: return []
        res = self._knearest(self.root,x,KNearestResult(k),filter)
        return res.sorted_items()

    def _neighbors(self,node,x,rad,results):
        if node.splitvalue == None:
            #base case, it's a leaf
            for p in node.points:
                d = self.metric(p[0],x)
                if d <= rad:
                    results.append(p)
            return
        #recursive case, it's a parent
        dhi = 0
        dlo = 0
        if self.partitionFn(node.splitdim,node.splitvalue,x) < 0:
            dlo = 0
            dhi = self.minDistanceFn(node.splitdim,node.splitvalue,x)
        else:
            dhi = 0
            dlo = self.minDistanceFn(node.splitdim,node.splitvalue,x)

        if dhi <= rad:  #check right
            self._neighbors(node.right,x,rad,results)
        if dlo <= rad: # check left
            self._neighbors(node.left,x,rad,results)

    def neighbors(self,x,rad):
        """Distance neighbor query:
        Returns the list of (point,data) pairs in the tree within distance
        rad to the point x"""
        if self.root == None: return []
        retval = []
        self._neighbors(self.root,x,rad,retval)
        return retval

class NewNode:
    def __init__(self,parent, points,bounds,splitdim=0):
        """Arguments:
        - points: a list of (point,data) pairs
        - splitdim: the default split dimension.
        """
        assert isinstance(points,(list,tuple))
        self.points = points
        assert len(bounds[0]) == 2
        assert len(bounds[1]) == 2
        self.splitdim = splitdim
        self.depth = 0
        self.splitvalue = None
        self.left = None
        self.right = None
        self.parent = parent

        self.n = 1
        self.n_cost_samples = 0.1
        self.cost_sum = 0
        self.cost_bound = 0
        self.bounds = bounds
        # self.min_bound = []
        # self.max_bound = []
        # self.pass_volume()
        # [p[1].local_volume for p in self.points]
        self.set_local_volume()

    def update_cost_sum(self, cost):
        # ipdb.set_trace()
        self.cost_sum += cost
        self.n_cost_samples += 1

    def set_cost_ave(self, cost):
        self.cost_sum = cost
        self.n_cost_samples = 1

    def cost_ave(self):
        return self.cost_sum/self.n_cost_samples

    def volume(self):
        widths = [max_i - min_i for (max_i, min_i) in zip(self.bounds[0], self.bounds[1])]
        return np.prod(widths)

    def set_local_volume(self):
        v = self.volume()
        for point in self.points:
            point[1].set_new_local_volume(v/len(self.points))


    def pass_volume(self):
        v = self.volume()
        for point in self.points:
            point[1].set_new_volume(v/len(self.points))



    # def update_cost_bound(self, cost_bound):
    #     if cost_bound < self.cost_bound:
    #         self.cost_bound = cost_bound
    #         if self.parent is not None:
    #             self.parent.update_cost_bound(cost_bound)
        # self.cost_sum += cost_bound



#     def mcts_select(self):
#         if self.left is None or self.right is None: 
#             return self
#         else:
#             lam = (self.n+1)**0.5
#             c_1 = self.left.c_ave
#             c_2 = self.right.c_ave


class KDTreePolicy(KDTree):
#     # def volume(self, node):
#     #     raise NotImplemented
    def __init__(self,distanceMetric=metric.euclideanMetric,cspace=None, goal=None):
    # def __init__(self,distanceMetric=lambda x, y: math.sqrt(sum([(xi - yi)**2 for xi, yi in zip(x, y)])),
    #         cspace=None, goal=None):
    # def __init__(self,distanceMetric=None,
    #         cspace=None, goal=None):
        super().__init__(distanceMetric)
        assert goal is not None
        assert cspace is not None
        self.goal = goal
        self.configuration_space = cspace
        try: 
            self.lo, self.hi = cspace.box.bounds()
        except:
            self.lo, self.hi = cspace.bounds()
        self.pass_volume = False
        self.track_cost = True
        # import ipdb
        # ipdb.set_trace()

    def reset(self):
        self.__init__(distanceMetric=self.metric, cspace=self.configuration_space, goal=self.goal)

    def sample(self):
        samp = self._sample(self.root) 
        # samp = self._mcts_sample(self.root)
        n_val = 1/20*(np.log(self.root.n_cost_samples + 1)/2)**(-0.5)
        # n_val = self.root.n_cost_samples**0.5
        offset = (np.random.normal(size=(len(samp),))*n_val).tolist()
        # offset = [0,0]
        return [s + o for s, o in zip(samp, offset)]
        # return self._mcts_sample(self.root)


    def locate(self,point, inc):
        """Find the Node which contains a point"""
        return self._locate(self.root,point, inc)

    def _locate(self,node,point, inc):
        if node.splitvalue == None: return node
        if self.partitionFn(node.splitdim,node.splitvalue,point) < 0: 
            if inc: node.n += 1
            return self._locate(node.left,point,inc)
        else: 
            if inc: node.n += 1
            return self._locate(node.right,point,inc)

    def add(self,point,data):
        """Add a point to the KD tree (O(log(n)) running time)"""
        if self.root == None:
            # self.root = self.makeNode([(point,data)], (self.lo, self.hi))
            self.root = NewNode(None, [(point,data)], (self.lo, self.hi))
            self.maxDepth = 0
            self.numNodes = 1
            return_value = self.root
            self.root.pass_volume()
        else:
            node = self.locate(point, inc=True)
            node.points.append((point,data))
            if self.recursive_split(node,optimize=False):
                return_value =  self._locate(node,point, inc=False)
            else:
                return_value =  node

        def check_single_pt(node):
            assert len(node.points) <= 1
        # self.traverse(self.root, check_single_pt)

        return return_value

    def recursive_split(self,node,force=False,optimize=False):
        """Recursively splits the node along the best axis.
        - node: the node to split
        - force: true if you want to force-split this node.  Otherwise,
          uses the test |node.points| > maxPointsPerNode to determine
          whether to split.
        - optimize: true if you want to select the split dimension with
          the widest range.
        Returns the depth of the subtree if node is split, 0 if not split.
        """
        # force = True
        if not force and len(node.points) <= self.maxPointsPerNode:
            ctc = node.points[0][1].c
            ctg = self.ctg_upper_bound(node.points[0][0])
            # self.set_new_cost_bound(points, new_cost)
            node.set_cost_ave(ctc + ctg)
            return 0
        if len(node.points)==0:
            return 0
        if node.left != None:
            #already split
            raise RuntimeError("Attempting to split node already split")
        d = len(node.points[0][0])
        vmin,vmax = 0,0
        if not optimize:
            #just loop through the dimensions
            for i in range(d):
                vmin = min(p[0][node.splitdim] for p in node.points)
                vmax = max(p[0][node.splitdim] for p in node.points)
                if vmin != vmax:
                    break
                #need to choose a new split dimension
                node.splitdim = (node.splitdim+1)%d
        else:
            rangemax = (0,0)
            dimmax = 0
            for i in range(d):
                vmin = min(p[0][i] for p in node.points)
                vmax = max(p[0][i] for p in node.points)
                if vmax-vmin > rangemax[1]-rangemax[0]:
                    rangemax = (vmin,vmax)
                    dimmax = i
            node.splitdim = dimmax
            vmin,vmax = rangemax
        if vmin == vmax:
            #all points are equal, don't split (yet)
            return 0
        node.splitvalue = (vmin+vmax)*0.5
        leftpts = []
        rightpts = []
        for p in node.points:
            if self.partitionFn(node.splitdim,node.splitvalue,p[0]) < 0: leftpts.append(p)
            else: rightpts.append(p)
        if len(leftpts) == 0 or len(rightpts) == 0:
            #may have numerical error
            node.splitvalue = None
            return 0

        new_bounds = self.newBounds(node.bounds, node.splitdim, node.splitvalue)

        node.left = NewNode(node, leftpts, new_bounds[0], splitdim=(node.splitdim+1)%d)
        node.right = NewNode(node, rightpts, new_bounds[1], splitdim=(node.splitdim+1)%d)
        node.left.pass_volume()
        node.right.pass_volume()
        node.left.depth = node.depth+1
        node.right.depth = node.depth+1
        self.numNodes += 2
        self.maxDepth = max(self.maxDepth,node.depth+1)
        d1 = self.recursive_split(node.left,force=False,optimize=optimize)
        d2 = self.recursive_split(node.right,force=False,optimize=optimize)
        node.points = []
        return 1+max(d1,d2)

    # def makeNode(self, pts, bound, splitdim=None):
    #     if splitdim == None:
    #         return NewNode(pts, bound)
    #     else:
    #         return NewNode(pts, bound, splitdim=splitdim)

    def newBounds(self, bounds, splitdim, splitvalue):
        lo_1, lo_2 = bounds[0].copy(), bounds[0].copy()
        hi_1, hi_2 = bounds[1].copy(), bounds[1].copy()

        lo_2[splitdim] = splitvalue
        hi_1[splitdim] = splitvalue
        return ((lo_1, hi_1),(lo_2, hi_2))


    def set_new_cost_bound(self, points, new_cost):
        if not self.track_cost:
            return
        for point in points: 
            self._set_new_cost_bound(self.root, point, new_cost)

    def _set_new_cost_bound(self, node, point, new_cost):
        if not self.track_cost:
            return
        node.update_cost_sum(new_cost)
        # node.cost_sum += new_cost
        # node.n_cost_samples += 1   
        if verbose: print(f"KDTree cost bound -- \n\tPoint: {point}"
            f"\n\tNode: ({node.splitdim}, {node.splitvalue})"
            f"\n\tNew Cost: {new_cost}"
            # f"\n\tN Samples: {node.n}"
            f"\n\tNum Samples: {node.n_cost_samples}"
            f"\n\tAve Cost: {node.cost_ave()}"
            )     
        if node.splitvalue == None: 
            # ipdb.set_trace()
            return None
        if node.cost_bound > new_cost:
            node.cost_bound = new_cost
        if self.partitionFn(node.splitdim,node.splitvalue,point) < 0: 
            return self._set_new_cost_bound(node.left,point,new_cost)
        else: 
            return self._set_new_cost_bound(node.right,point,new_cost)


    def _sample_node(self, node):
        sample = self.configuration_space.sampleAsymmetricNeighborhood(node.bounds[0], node.bounds[1])
        # assert self.metric(sample, node.points[0][0]) > 0
        return sample

    def _sample(self, node):
        # if np.random.rand() < .1: return self.configuration_space.sample()
        if node.left is None or node.right is None:
            # return self.configuration_space.sampleAsymmetricNeighborhood(lo, hi)
            # return node
            return self._sample_node(node)
        else:
            lam = 10000#node.n_cost_samples**(-.5)
            # c_1 = 1
            # c_2 = 1
            # c_1 = node.left.cost_bound
            # c_2 = node.right.cost_bound
            c_1 = node.left.cost_ave()
            c_2 = node.right.cost_ave()
            # vol_1 = 1
            # vol_2 = 1
            # lam = (self.node.n+1)**0.5
            # c_1 = self.node.left.c_ave
            # c_2 = self.node.right.c_ave
            vol_1 = node.left.volume()
            vol_2 = node.right.volume()

            # a = lam
            # b = -lam*(c_1 + c_2) + (vol_1 + vol_2)
            # c = lam*c_1*c_2 - (vol_1*c_1 + vol_2*c_2)
            # alpha_plus = (- b + (b**2 - 4*a*c)**0.5)/(2*a)
            # alpha_min = (- b - (b**2 - 4*a*c)**0.5)/(2*a)

            # pi = lambda a: (1/(c_1- alpha)*vol_1, 1/(c_2- alpha)*vol_2)
            # def v(a):
            #     pi_1, pi_2 = pi(a)
            #     return c_1*pi_1 + c_2*pi_2 - lam*

            alpha_lower_bound = min([c_1, c_2]) - lam
            alpha_upper_bound = min([c_1 - lam*vol_1, c_2 - lam*vol_2])

            # alphas = []
            # if alpha_min >= alpha_lower_bound and alpha_min <= alpha_upper_bound:
            #     alphas.append(alpha_min)
            # if alpha_plus >= alpha_lower_bound and alpha_plus <= alpha_upper_bound:
            #     alphas.append(alpha_plus)
            # import ipdb
            # ipdb.set_trace()
            # assert len(alphas) > 0
            # alpha = alphas[-1]
            alpha = alpha_upper_bound
            #will need to flesh this out later with bifurcation search
            lam_inv = 0.01*node.n_cost_samples**(.5) #breaks            
            lam_inv = 0.0001*node.n_cost_samples**(.5)
            lam_alpha_upper_bound = min([lam_inv*c_1 - vol_1, lam_inv*c_2 - vol_2])
            alpha = lam_alpha_upper_bound
            if lam_inv*c_1 - alpha <= 0 or lam_inv*c_2 - alpha <= 0:
                import ipdb
                ipdb.set_trace()
            # if np.random.rand()*(lam*vol_1/(c_1 - alpha) + lam*vol_2/(c_2 - alpha)) < lam*vol_1/(c_1 - alpha):
            if np.random.rand()*(vol_1/(lam_inv*c_1 - alpha) + vol_2/(lam_inv*c_2 - alpha)) < vol_1/(lam_inv*c_1 - alpha):
            # if np.random.rand()*(vol_1 + vol_2) < vol_1:
                return self._sample(node.left)
            else: 
                return self._sample(node.right)


    def _mcts_sample(self, node):
        # return self._sample_node(node)
        # node.n_cost_samples += 1
        node.n += 1
        if np.random.rand() < 0.1: return self.configuration_space.sample()
        if verbose: print(f"MCTS Sample -- "
            f"\n\tNode: ({node.splitdim}, {node.splitvalue})"
            f"\n\tN Samples: {node.n}"
            )     

        if node.left is None or node.right is None: 
            return self._sample_node(node)
        else:
            lam = (node.n_cost_samples)**0.5
            # lam = (node.n)**0.5
            c_1 = node.left.cost_ave()
            c_2 = node.right.cost_ave()
            vol_1 = node.left.volume()
            vol_2 = node.right.volume()

            # ucb_1 = -c_1 + vol_1*lam/(node.left.n_cost_samples)
            # ucb_2 = -c_2 + vol_2*lam/(node.right.n_cost_samples)
            ucb_1 = -c_1 + vol_1*lam/(node.left.n_cost_samples)
            ucb_2 = -c_2 + vol_2*lam/(node.right.n_cost_samples)
            # lam = 1
            # ucb_1 = vol_1*lam/(node.left.n)
            # ucb_2 = vol_2*lam/(node.right.n)
            # ucb_1 = -c_1 + vol_1*lam/(node.left.n)
            # ucb_2 = -c_2 + vol_2*lam/(node.right.n)
            # ucb_1 = vol_1*np.random.rand()
            # ucb_2 = vol_2*np.random.rand()

            if ucb_1 > ucb_2:
                return self._mcts_sample(node.left)
            else:
                return self._mcts_sample(node.right)

#     def cost_est(point):
#         L = 10
#         base_k = 5
#         neighbors = self.knearest(point, base_k)
#         print("Look at the structure of data")
#         import ipdb
#         ipdb.set_trace()
#         # costs = [n[1] for n in neighbors]
#         # c_est = min([L*self.metric(point, loc) + cost for (loc, cost) in neighbors])
#         c_est = min([L*self.metric(point, self.goal) + cost for (loc, cost) in neighbors])
    def set_other_kdtree(self, other_kdtree):
        self.other_kdtree = other_kdtree
        
    def ctg_upper_bound(self, x):
        if not self.track_cost:
            return 0
        ctg = lambda x, c: 50*self.metric(x,c)
        # return ctg(x, self.goal.c)
        pts = [pt[1] for pt in self.other_kdtree.knearest(x, 5)]
        # return min(ctg(x, self.goal.c), min([ctg(x, pt.x) for pt in pts]))
        # nearest = self.pickNode(x)
        min_dist = min(pt.c + ctg(x, pt.x) for pt in pts)
        # return min(ctg(x, self.goal.c),nearest.c + ctg(x, nearest.x))
        return min_dist
        # return min(ctg(x, self.other_kdtree.root.points[0]), min_dist)

# class KDTreePair:
#     """docstring for KDTreePair"""
#     def __init__(self, forward_tree, backwar):
#         self.arg = arg

#     def reset(self):
#         self.__init__(distanceMetric=self.metric, cspace=self.configuration_space, goal=self.goal)

#     def sample(self):

#     def locate(self,point, inc):

#     def add(self,point,data):

#     def recursive_split(self,node,force=False,optimize=False):




class KDTreePolicyAlt(KDTree):
#     # def volume(self, node):
#     #     raise NotImplemented
    def __init__(self,distanceMetric=metric.euclideanMetric, lo = None, hi = None):
        assert lo is not None
        assert hi is not None
        super().__init__(distanceMetric)
        self.lo = lo
        self.hi = hi
        self.pass_volume = False
        self.track_cost = True
        # import ipdb
        # ipdb.set_trace()

    def reset(self):
        self.__init__(distanceMetric=self.metric, lo=self.lo, hi=self.hi)

    def locate(self,point, inc):
        """Find the Node which contains a point"""
        return self._locate(self.root,point, inc)

    def _locate(self,node,point, inc):
        if node.splitvalue == None: return node
        if self.partitionFn(node.splitdim,node.splitvalue,point) < 0: 
            if inc: node.n += 1
            return self._locate(node.left,point,inc)
        else: 
            if inc: node.n += 1
            return self._locate(node.right,point,inc)

    def add(self,point,data):
        """Add a point to the KD tree (O(log(n)) running time)"""
        if self.root == None:
            # self.root = self.makeNode([(point,data)], (self.lo, self.hi))
            self.root = NewNode(None, [(point,data)], (self.lo, self.hi))
            self.maxDepth = 0
            self.numNodes = 1
            return_value = self.root
            self.root.pass_volume()
        else:
            node = self.locate(point, inc=True)
            node.points.append((point,data))
            if self.recursive_split(node,optimize=False):
                return_value =  self._locate(node,point, inc=False)
            else:
                return_value =  node

        def check_single_pt(node):
            assert len(node.points) <= 1
        # self.traverse(self.root, check_single_pt)

        return return_value

    def recursive_split(self,node,force=False,optimize=False):
        """Recursively splits the node along the best axis.
        - node: the node to split
        - force: true if you want to force-split this node.  Otherwise,
          uses the test |node.points| > maxPointsPerNode to determine
          whether to split.
        - optimize: true if you want to select the split dimension with
          the widest range.
        Returns the depth of the subtree if node is split, 0 if not split.
        """
        # force = True
        if not force and len(node.points) <= self.maxPointsPerNode:
            return 0
        if len(node.points)==0:
            return 0
        if node.left != None:
            #already split
            raise RuntimeError("Attempting to split node already split")
        d = len(node.points[0][0])
        vmin,vmax = 0,0
        if not optimize:
            #just loop through the dimensions
            for i in range(d):
                vmin = min(p[0][node.splitdim] for p in node.points)
                vmax = max(p[0][node.splitdim] for p in node.points)
                if vmin != vmax:
                    break
                #need to choose a new split dimension
                node.splitdim = (node.splitdim+1)%d
        else:
            rangemax = (0,0)
            dimmax = 0
            for i in range(d):
                vmin = min(p[0][i] for p in node.points)
                vmax = max(p[0][i] for p in node.points)
                if vmax-vmin > rangemax[1]-rangemax[0]:
                    rangemax = (vmin,vmax)
                    dimmax = i
            node.splitdim = dimmax
            vmin,vmax = rangemax
        if vmin == vmax:
            #all points are equal, don't split (yet)
            return 0
        node.splitvalue = (vmin+vmax)*0.5
        leftpts = []
        rightpts = []
        for p in node.points:
            if self.partitionFn(node.splitdim,node.splitvalue,p[0]) < 0: leftpts.append(p)
            else: rightpts.append(p)
        if len(leftpts) == 0 or len(rightpts) == 0:
            #may have numerical error
            node.splitvalue = None
            return 0

        new_bounds = self.newBounds(node.bounds, node.splitdim, node.splitvalue)

        node.left = NewNode(node, leftpts, new_bounds[0], splitdim=(node.splitdim+1)%d)
        node.right = NewNode(node, rightpts, new_bounds[1], splitdim=(node.splitdim+1)%d)
        node.left.pass_volume()
        node.right.pass_volume()
        node.left.depth = node.depth+1
        node.right.depth = node.depth+1
        self.numNodes += 2
        self.maxDepth = max(self.maxDepth,node.depth+1)
        d1 = self.recursive_split(node.left,force=False,optimize=optimize)
        d2 = self.recursive_split(node.right,force=False,optimize=optimize)
        node.points = []
        return 1+max(d1,d2)

    def newBounds(self, bounds, splitdim, splitvalue):
        lo_1, lo_2 = bounds[0].copy(), bounds[0].copy()
        hi_1, hi_2 = bounds[1].copy(), bounds[1].copy()

        lo_2[splitdim] = splitvalue
        hi_1[splitdim] = splitvalue
        return ((lo_1, hi_1),(lo_2, hi_2))




verbose = False