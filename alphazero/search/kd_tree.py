import math
# from ..spaces import metric
import numpy as np
import torch
import ipdb

euclideanMetric = lambda x, y: math.sqrt(sum([(xi - yi)**2 for xi, yi in zip(x, y)]))

class KNearestResult:
    def __init__(self,k):
        assert k > 0
        self.items = [None]*k
        self.distances = [float('inf')]*k
        self.imin = 0
        self.imax = 0
    def tryadd(self,item,distance):
        if distance < self.distances[self.imax]:
            self.distances[self.imax] = distance
            self.items[self.imax] = item
            if distance < self.distances[self.imin]:
                self.imin = self.imax
            #update imin
            for i in range(len(self.items)):
                if self.distances[i] > self.distances[self.imax]:
                    self.imax = i
    def minimum_distance(self):
        return self.distances[self.imin]
    def maximum_distance(self):
        return self.distances[self.imax]
    def sorted_items(self):
        sorted_res = sorted([(d,i) for (i,d) in zip(self.items,self.distances) if d!=float('inf')], key=lambda x:x[0])
        return [v[1] for v in sorted_res]

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


class KDTree:
    # def __init__(self,distanceMetric=metric.euclideanMetric):
    def __init__(self,distanceMetric=euclideanMetric):
        # import ipdb
        # ipdb.set_trace()
        self.root = None
        self.metric = distanceMetric
        self.partitionFn = lambda d,value,x: math.copysign(1,x[d]-value)
        self.minDistanceFn = lambda d,value,x: abs(x[d]-value)

        self.maxPointsPerNode = 10
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
        # self.root = Node(list(zip(points,data)))
        self.root = self.makeNode(list(zip(points,data)))
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
        
    def knn_density(self,x,k,filter=None):
        if k>self.numNodes:
            k=self.numNodes
        knn = self.knearest(x, k, filter=filter)
        dim = len(x)
        density = k/self.numNodes*1/self.metric(knn[-1][0],x)**dim
        return density

    def knn_inv_density(self,x,k,filter=None):
        if k>self.numNodes:
            k=self.numNodes
        knn = self.knearest(x, k, filter=filter)
        dim = len(x)
        inv_density = self.metric(knn[-1][0],x)**dim*self.numNodes/k
        return density

    def kernel_density(self,x,k,filter=None):
        if k>self.numNodes:
            k=self.numNodes
        knn = self.knearest(x, k, filter=filter)
        dim = len(x)
        sigma = 0.1
        density = 1/(sigma*np.sqrt(np.pi))*1/self.numNodes*(sum(
            np.exp(-(self.metric(point[0],x)/sigma)**2)
            for point in knn
        ))
        return density

    def kernel_regression(self, x, func, k=5,filter=None):
        if k>self.numNodes:
            k=self.numNodes
        # import ipdb
        # ipdb.set_trace()
        knn = self.knearest(x, k, filter=filter)
        dim = len(x)
        sigma = 0.05/(1+np.log(k))
        dens = lambda pt, x: pt[0].n*np.exp(-(self.metric(pt[0],x)/sigma)**2)
        density = sum(
            dens(point, x)
            for point in knn
        )
        total = sum(
            # func(point[1])*np.exp(-(self.metric(point[0],x)/sigma)**2)
            func(point[1])*dens(point, x)
            for point in knn
        )
        return total/density


    def knn_estimate(self, x, func, k=1,filter=None):
        if k>self.numNodes:
            k=self.numNodes
        # k=1
        knn = self.knearest(x, k, filter=filter)
        dens = lambda pt, x: pt[1].n/k
        density = sum(
            dens(point, x)
            for point in knn
        )
        total = sum(
            # func(point[1])*np.exp(-(self.metric(point[0],x)/sigma)**2)
            func(point[1])*dens(point, x)
            for point in knn
        )
        return total, density

    def knn_regression(self, x, func, k=1,filter=None):
        if k>self.numNodes:
            k=self.numNodes
        # k=1
        knn = self.knearest(x, k, filter=filter)
        ave = sum(func(point[1]) for point in knn)/len(knn)
        return ave

    def density_target(self, x,  k=1,filter=None):
        if k>self.numNodes:
            k=self.numNodes
        func = lambda x: x.prob
        knn = self.knearest(x, k, filter=filter)
        ave = sum(func(point[1]) for point in knn)/len(knn)
        return ave

    def best_nearby(self, x, func, k=10,filter=None):
        if k>self.numNodes:
            k=self.numNodes
        knn = self.knearest(x, k, filter=filter)
        best = max(func(point[1]) for point in knn)
        return best

    def nearby_with_penalty(self, x, func, k=10,filter=None):
        if k>self.numNodes:
            k=self.numNodes
        knn = self.knearest(x, k, filter=filter)
        best = max(func(point[0], point[1]) for point in knn)
        return best

    def exploration_bonus(self, x):
        # import ipdb
        # ipdb.set_trace()
        # return 1/np.sqrt(self.kernel_density(x, 5) + 0.01)
        # return np.sqrt(np.log(self.numNodes)/(self.kernel_density(x, 5) + 0.01))
        return np.sqrt(self.numNodes)/(self.kernel_density(x, 5) + 0.01)


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


    def walk_node(self, node):
        left_points  = [] if node.left  == None else self.walk_node(node.left )
        right_points = [] if node.right == None else self.walk_node(node.right)
        return node.points + left_points + right_points

class KDTreeNode:
    def __init__(self, density_model, parent, points,lo, hi, 
            env_lo, env_hi, 
            splitdim=0, root_state=None
        ):
        """Arguments:
        - points: a list of (point,data) pairs
        - splitdim: the default split dimension.
        """
        # import ipdb
        # ipdb.set_trace()
        assert isinstance(points,(list,tuple))
        self.points = points
        # assert len(bounds[0]) == 2
        # assert len(bounds[1]) == 2
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
        self.lo = lo
        self.hi = hi
        self.env_lo = env_lo
        self.env_hi = env_hi

        self.bound = 10**6
        self.scale = 10
        self.unbounded = [l < -self.bound or h > self.bound for l, h in zip(lo, hi)]


        self.value_sum = 0
        self.value_num = 0

        if root_state is None:
            import ipdb
            ipdb.set_trace()
        # self.local_volume = None
        # density = density_model.density(torch.tensor(lo), torch.tensor(hi))#*0 + 1    
        # density = density_model.density(torch.tensor(points[0][0]))#*0 + 1      
        # density = density_model.density(torch.tensor(root_state), torch.tensor(points[0][0]))#*0 + 1      
        # density = density_model.density(torch.tensor(root_state), torch.tensor(points[0][0])).sum().detach().item()#*0 + 1      
        density = 1
        if type(density) == torch.Tensor:
            self.density = density.item()
        else: 
            self.density = density

        self.set_local_volume()
        # self.density_model = density_model
        try:
            for point in points: 
                for dim in range(len(point)):
                    assert point[0][dim] <= hi[dim] 
                    assert point[0][dim] >= lo[dim] 
                    assert lo[dim] < hi[dim]
        except:
            import ipdb
            ipdb.set_trace()
                # assert
        # assert self.local_volume is not None

    def width(self, max_i, min_i, i):
        try: 
            assert min_i >= self.env_lo[i]
            assert max_i >=  self.env_lo[i]
            assert min_i <= self.env_hi[i]
            assert max_i <= self.env_hi[i]
        except: 
            import ipdb
            ipdb.set_trace()
        if self.env_lo[i] >= -self.bound and self.env_hi[i] <= self.bound:
            #Axis is bounded. Find volume
            rv =  (max_i - min_i)/(self.env_hi[i] - self.env_lo[i])

        elif self.env_lo[i] < -self.bound and self.env_hi[i] <= self.bound: 
            #Unbounded on the low end
            rv = (
                np.exp((max_i - self.env_hi[i])/self.scale)/self.scale - 
                np.exp((min_i - self.env_hi[i])/self.scale)/self.scale 
            )
        elif self.env_lo[i] >= -self.bound and self.env_hi[i] > self.bound: 
            #Unbounded on the hi end
            rv = (
                np.exp((self.env_lo[i] - min_i)/self.scale)/self.scale - 
                np.exp((self.env_lo[i] - max_i)/self.scale)/self.scale 
            )
        else:  
            #Unbounded on both ends
            if min_i == 0 or max_i == 0:
                sign = 0
            elif (min_i > 0 and max_i < 0) or (min_i < 0 and max_i > 0):
                sign = -1
            else: 
                sign=1                
            # sign = min_i*max_i
            # if sign > 0:
            #     #Max and min bounds both on same side of 0
            #     # rv = np.exp(-np.abs(max_i - min_i)/self.scale)/(2*self.scale)
            #     rv = np.abs(
            #         np.exp(-np.abs(max_i)/self.scale)/2 - 
            #         np.exp(-np.abs(min_i)/self.scale)/2
            #     )
            #     # rv = np.exp(-np.abs(max_i - min_i)/self.scale)/(2)
            # elif sign < 0:
            #     #Max and min bounds on opposite sides of 0
            #     rv = ( 
            #         1 + 
            #         - np.exp(-np.abs(max_i)/self.scale)/(2)
            #         - np.exp(-np.abs(min_i)/self.scale)/(2)
            #     )
            # else:
            #     #One bound is 0
            #     rv = (
            #         np.exp(-np.abs(max_i)/self.scale)/(2) + 
            #         np.exp(-np.abs(min_i)/self.scale)/(2) 
            #     )

            if sign >= 0:
                #Max and min bounds both on same side of 0
                rv = np.abs(
                    np.exp(-np.abs(max_i)/self.scale)/2 - 
                    np.exp(-np.abs(min_i)/self.scale)/2
                )
            elif sign < 0:
                #Max and min bounds on opposite sides of 0
                rv = ( 
                    1 + 
                    - np.exp(-np.abs(max_i)/self.scale)/(2)
                    - np.exp(-np.abs(min_i)/self.scale)/(2)
                )

        if rv < 0 or rv > 2: 
            import ipdb
            ipdb.set_trace()
        return rv

    def volume(self):
        epsilon = 10**(-10)
        # widths = [max_i - min_i for (max_i, min_i) in zip(self.hi, self.lo)]
        widths = [self.width(self.hi[i], self.lo[i], i) for i in range(len(self.lo))]
        try: 
            for w in widths:
                # assert w > epsilon
                assert w < 10
        except: 
            ipdb.set_trace()
        for w in widths: 
            if w < 0: 
                ipdb.set_trace()
            # assert w > 0 
        v = np.prod(widths)#*self.density     
        # ipdb.set_trace()
        if v < 0: 
            ipdb.set_trace()
        if v == 0:
            v = 10**(-8)
        return v

    def set_local_volume(self):
        v = self.volume()
        for point in self.points:
            # point[1].set_new_local_volume(v/len(self.points))
            point[1].set_new_local_volume(self.lo, self.hi, v/len(self.points), self.density)


    def pass_volume(self):
        v = self.volume()
        for point in self.points:
            # point[1].set_new_volume(v/len(self.points))
            point[1].set_new_volume(self.lo, self.hi, v/len(self.points), self.density)

    def zero_volume(self):
        for point in self.points:
            point[1].set_new_volume(point[0], point[0], 0.00000001, self.density)

    def update(self, V):
        self.value_sum += V
        self.value_num += 1


    def value(self):
        epsilon = 0.00000001
        return (self.value_sum + 20*sum((pt[1].r for pt in self.points)))/(self.value_num + len(self.points) + epsilon)
        if self.value_num == 0.:
            return 0
        else: 
            return self.value_sum/self.value_num



class KDTreePolicyAlt(KDTree):
    def __init__(self, distanceMetric=euclideanMetric, lo = None, hi = None, density_model=None):
        assert lo is not None
        assert hi is not None
        super().__init__(distanceMetric)
        
        self.lo = lo
        self.hi = hi
        self.pass_volume = False
        self.track_cost = True
        self.root = None
        self.root_state=None
        self.numNodes = 0
        self.maxPointsPerNode = 1
        # import ipdb
        # ipdb.set_trace()
        self.density_model = density_model
        # assert model is not None
        try:
            for dim in range(len(lo)):
                assert lo[dim] < hi[dim]
        except:
            import ipdb
            ipdb.set_trace()

    def makeNode(self,
            density_model, parent, points,lo, hi, 
            env_lo, env_hi, 
            splitdim=0, root_state=None
        ):
    
        return KDTreeNode(
            density_model, parent, points,lo, hi, 
            env_lo, env_hi, 
            splitdim=splitdim, root_state=root_state
        )

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

    def add(self,point,data, displaced=False):
        try: 
            for i in range(len(self.lo)):
                assert point[i] >= self.lo[i]
                assert point[i] <= self.hi[i]
        except: 
            import ipdb
            ipdb.set_trace()

        """Add a point to the KD tree (O(log(n)) running time)"""
        # if type(point) is np.ndarray:
        #     point = point.tolist()

        if self.root == None:
            # self.root = self.makeNode([(point,data)], (self.lo, self.hi))
            self.root_state=point
            self.root = self.makeNode(self.density_model, None, [(point,data)], self.lo, self.hi, 
                self.lo, self.hi,
                root_state=point)
            self.maxDepth = 0
            self.numNodes = 1
            return_value = self.root
            # self.root.pass_volume()
            self.root.set_local_volume()
            return_value = self.root
        else:
            node = self.locate(point, inc=True)
            changed_node = node
            node.points.append((point,data))
            # ipdb.set_trace()
            if self.recursive_split(node,optimize=False, data=data):
                return_value =  self._locate(node,point, inc=False)
                # [pt[-1].local_density() for pt in return_value.points]
            else:
                return_value =  node
                # [pt[-1].local_density() for pt in return_value.points]

            # [pt[-1].local_density() for pt in return_value.points]
            # ipdb.set_trace()


        # assert re
        [p[1] for p in return_value.points]
        def check_single_pt(node):
            assert len(node.points) <= 1
        # self.traverse(self.root, check_single_pt)
        if displaced: 
            if return_value.parent == None:
                return return_value, return_value.points
            else: 
                return return_value, self.walk_node(return_value.parent)
        else: 
            return return_value

    def recursive_split(self,node,force=False,optimize=False, data=None):
        # assert node.local_volume is not None
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
            # ipdb.set_trace()
            # node.update(data.V)
            return 0
        if len(node.points)==0:
            ipdb.set_trace()
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
            return 0
        node.splitvalue = float((vmin+vmax)*0.5)
        leftpts = []
        rightpts = []
        for p in node.points:
            if self.partitionFn(node.splitdim,node.splitvalue,p[0]) < 0: leftpts.append(p)
            else: rightpts.append(p)
        if len(leftpts) == 0 or len(rightpts) == 0:
            #may have numerical error
            # node.update(data.V)
            node.splitvalue = None
            # ipdb.set_trace()
            return 0

        vol = node.volume()
        old_bounds = ([p for p in node.lo], [p for p in node.hi])
        assert vol > 0
        new_bounds = self.newBounds(node.lo, node.hi, node.splitdim, node.splitvalue)
        # ipdb.set_trace()

        node.left = self.makeNode(self.density_model, node, leftpts, 
                new_bounds[0][0], new_bounds[0][1],
                self.lo, self.hi,
                splitdim=(node.splitdim+1)%d, 
                root_state=self.root_state)
        node.right = self.makeNode(self.density_model, node, rightpts, 
                new_bounds[1][0], new_bounds[1][1], 
                self.lo, self.hi,
                # new_bounds[0][0], new_bounds[0][1], 
                splitdim=(node.splitdim+1)%d, 
                root_state=self.root_state)

        assert node.left.volume() > 0
        assert node.right.volume() > 0
        node.left.pass_volume()
        node.right.pass_volume()
        node.left.depth = node.depth+1
        node.right.depth = node.depth+1
        self.numNodes += 2
        self.maxDepth = max(self.maxDepth,node.depth+1)
        d1 = self.recursive_split(node.left,force=False,optimize=optimize, data=data)
        d2 = self.recursive_split(node.right,force=False,optimize=optimize, data=data)
        node.points = []
        return 1+max(d1,d2)

    def newBounds(self, lo, hi, splitdim, splitvalue):
        try:
            for dim in range(len(lo)):
                assert lo[dim] < hi[dim]
        except:
            import ipdb
            ipdb.set_trace()
        lo_1, lo_2 = lo.copy(), lo.copy()
        hi_1, hi_2 = hi.copy(), hi.copy()

        # ipdb.set_trace()
        # lo_2[splitdim] = splitvalue
        # hi_1[splitdim] = splitvalue
        lo_2 = [(l if i!= splitdim else splitvalue) for i, l in enumerate(lo)]
        hi_1 = [(h if i!= splitdim else splitvalue) for i, h in enumerate(hi)]
        try:
            for dim in range(len(lo)):
                assert lo_1[dim] < hi_1[dim]
                assert lo_2[dim] < hi_2[dim]
        except:
            import ipdb
            ipdb.set_trace()
        return ((lo_1, hi_1),(lo_2, hi_2))



class KDTreeValue(KDTreePolicyAlt):
    # def __init__(self, distanceMetric=euclideanMetric, lo = None, hi = None, density_model=None):
    #     super().__init__(distanceMetric, lo, hi, density_model)
    #     self.Node = KDTreeNodeValue

    def backprop(self, loc, V):
        node = self.locate(loc, inc=False)

        node.update(V)
        while node.parent is not None:
            node = node.parent
            node.update(V)

    def the_value_from_halfway_down(self, loc, separate_sums=False):
        node = self.locate(loc, inc=False)
        return self.halfway_down_value_of_kd_node(node, separate_sums=separate_sums)

    def halfway_down_value_of_kd_node(self, node, separate_sums=False):
        num_steps = 0 #node.depth//4
        for _ in range(num_steps):
            node = node.parent

        if separate_sums:
            return node.value_sum, node.value_num
        else: 
            return node.value()


    def replace_kdnode_point(self, kdnode, new_point, new_entry):
        old_points = kdnode.points

        kdnode.points = [(new_point, new_entry)]
        # kdnode.set_local_volume()
        kdnode.pass_volume()

        for point in old_points:
            point[1].set_new_volume(point[0], point[0], 0.00000001, kdnode.density)



