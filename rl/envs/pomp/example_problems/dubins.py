from OpenGL.GL import *
from .geometric import *
from ..spaces.objectives import *
from ..spaces.statespace import *
from ..spaces.configurationspace import *
from ..spaces.edgechecker import *
from ..spaces.metric import *
from ..planners.problem import PlanningProblem
from ..spaces.so2space import *
from ..klampt import so2

def cmp(x,y):
    if x < y: return -1
    elif x > y: return 1
    return 0
    

class DubinsVisualizer:
    def __init__(self,workspace):
        self.base = workspace

    def toScreen(self,q):
        return q[0],q[1]

    def toState(self,x,y):
        return (x,y,0)

    def drawObstaclesGL(self):
        self.base.drawObstaclesGL()

    def drawVerticesGL(self,qs):
        self.base.drawVerticesGL(qs)

    def drawRobotGL(self,q):
        glColor3f(0,0,1)
        glPointSize(7.0)
        self.drawVerticesGL([q])
        l = 0.05
        d = (math.cos(q[2]),math.sin(q[2]))
        glBegin(GL_LINES)
        glVertex2f(q[0],q[1])
        glVertex2f(q[0]+l*d[0],q[1]+l*d[1])
        glEnd()

    def drawGoalGL(self,goal):
        self.base.drawGoalGL(goal)
        if isinstance(goal,NeighborhoodSubset):
            l = 0.05
            q = goal.c
            glBegin(GL_LINE_STRIP)
            d = (math.cos(q[2]+goal.r*math.pi*2),math.sin(q[2]+goal.r*math.pi*2))
            glVertex2f(q[0]+d[0]*l,q[1]+d[1]*l)
            glVertex2f(q[0],q[1])
            d = (math.cos(q[2]-goal.r*math.pi*2),math.sin(q[2]-goal.r*math.pi*2))
            glVertex2f(q[0]+d[0]*l,q[1]+d[1]*l)
            glEnd()

    def drawInterpolatorGL(self,interpolator):
        self.base.drawInterpolatorGL(interpolator)


class DubinsCarInterpolator(Interpolator):
    def __init__(self,space,x,u):
        Interpolator.__init__(self,x,space.nextState(x,u))
        self.space = space
        self.x = x
        self.control = u
    def length(self):
        return abs(self.control[0])
    def eval(self,u):
        return self.space.nextState(self.x,[self.control[0]*u,self.control[1]])
    def split(self,u):
        return DubinsCarInterpolator(self.space,self.x,[self.control[0]*u,self.control[1]]),DubinsCarInterpolator(self.space,self.eval(u),[self.control[0]*(1.0-u),self.control[1]])

class DubinsControlSet(BoxSet):
    """Compared to a standard BoxSet, this does a better job sampling
    uniformly from the reachable set of states"""
    def __init__(self,distanceMin,distanceMax,turnMin,turnMax):
        BoxSet.__init__(self,[distanceMin,turnMin],[distanceMax,turnMax])
    def sample(self):
        d = math.sqrt(random.random())
        if self.bmin[0] < 0:
            if random.random() < 0.5:
                d *= self.bmin[0]
            else:
                d *= self.bmax[0]
        phi = random.uniform(self.bmin[1],self.bmax[1])
        return [d,phi]

# class DubinsCarSpace (ControlSpace):
#     """u = (distance,turnRate)"""
#     def __init__(self,cspace):
#         self.space = cspace
#         self.space.setDistanceWeights([1,0.5/math.pi])
#         #self.controls = BoxSet([-1,-1],[1,1])
#         self.controls = DubinsControlSet(-1,1,-1,1)
#     def setDistanceBounds(self,minimum,maximum):
#         self.controls.bmin[0] = minimum
#         self.controls.bmax[0] = maximum
#     def setTurnRateBounds(self,minimum,maximum):
#         self.controls.bmin[1] = minimum
#         self.controls.bmax[1] = maximum
#     def configurationSpace(self):
#         return self.space
#     def controlSet(self,x):
#         return self.controls
#     def nextState(self,x,u):
#         # import ipdb
#         # ipdb.set_trace()
#         pos = [x[0],x[1]]
#         fwd = [math.cos(x[2]),math.sin(x[2])]
#         right = [-fwd[1],fwd[0]]
#         phi = u[1]
#         d = u[0]
#         if abs(phi)<1e-8:
#             newpos = vectorops.madd(pos,fwd,d)
#             return newpos + [x[2]]
#         else:
#             #rotate about a center of rotation, with radius 1/phi
#             cor = vectorops.madd(pos,right,1.0/phi)
#             sign = cmp(d*phi,0)
#             d = abs(d)
#             phi = abs(phi)
#             theta=0
#             thetaMax=d*phi
#             newpos = vectorops.add(so2.apply(sign*thetaMax,vectorops.sub(pos,cor)),cor)
#             return newpos + [so2.normalize(x[2]+sign*thetaMax)]
#     def interpolator(self,x,u):
#         return DubinsCarInterpolator(self,x,u)
#     def connection(self,x,y):
#         #TODO: dubins curves
#         return None

def action_normalize(s, lower_bound, upper_bound):
    return (s-lower_bound)*2/(upper_bound-lower_bound) - 1
    # return s*2/size - 1

def action_denormalize(s, lower_bound, upper_bound):
    return ((s+1)*(upper_bound-lower_bound)/2 + lower_bound)

class DubinsCarSpace (ControlSpace):
    """u = (distance,turnRate)"""
    def __init__(self,cspace):
        self.space = cspace
        self.space.setDistanceWeights([1,0.5/math.pi])
        #self.controls = BoxSet([-1,-1],[1,1])
        self.controls = DubinsControlSet(-1,1,-1,1)
    def setDistanceBounds(self,minimum,maximum):
        self.dist_min = minimum
        self.dist_max = maximum
        # self.controls.bmin[0] = minimum
        # self.controls.bmax[0] = maximum
    def setTurnRateBounds(self,minimum,maximum):
        self.turn_min = minimum
        self.turn_max = maximum
        # self.controls.bmin[1] = minimum
        # self.controls.bmax[1] = maximum
    def configurationSpace(self):
        return self.space
    def controlSet(self,x):
        return self.controls
    def nextState(self,x,u):
        action_bounds = (   
            np.array([self.dist_min, self.turn_min]),
            np.array([self.dist_max, self.turn_max])
        )
        # u = action_denormalize(u, action_bounds[0], action_bounds[1])

        pos = [x[0],x[1]]
        try: 
            fwd = [math.cos(x[2]),math.sin(x[2])]
        except: 
            import ipdb
            ipdb.set_trace()
        right = [-fwd[1],fwd[0]]
        phi = u[1]
        d = u[0]
        phi = action_denormalize(u[1], self.turn_min, self.turn_max)
        d = action_denormalize(u[0], self.dist_min, self.dist_max)
        # d = action_denormalize(u[0], -1, 1)
        if abs(phi)<1e-8:
            newpos = vectorops.madd(pos,fwd,d)
            return newpos + [x[2]]
        else:
            #rotate about a center of rotation, with radius 1/phi
            cor = vectorops.madd(pos,right,1.0/phi)
            sign = cmp(d*phi,0)
            d = abs(d)
            phi = abs(phi)
            theta=0
            thetaMax=d*phi
            newpos = vectorops.add(so2.apply(sign*thetaMax,vectorops.sub(pos,cor)),cor)
            return newpos + [so2.normalize(x[2]+sign*thetaMax)]
    def interpolator(self,x,u):
        return DubinsCarInterpolator(self,x,u)
    def connection(self,x,y):
        #TODO: dubins curves
        return None

class DubinsCarDistanceObjectiveFunction(ObjectiveFunction):
    def __init__(self,n):
        self.n = n
    def incremental(self,x,u):
        return sum(abs(u[i*2]) for i in range(self.n))


class DubinsSubset(NeighborhoodSubset):
    """A ball of radius r around a point c, with a distance given by
    a ConfigurationSpace.
    """
    def __init__(self,space,c,r):
        self.space = space
        self.c = c
        self.r = r

    def contains(self,x):
        val = self.space.distance(x[:2],self.c[:2]) <= self.r
        if val: 
            print("success!")
        return val

    def sample(self):
        x = self.space.sampleNeighborhood(self.c,self.r)
        # import ipdb
        # ipdb.set_trace()
        x[2] = np.random.rand()*2*math.pi
        return x

    def project(self,x):
        d = self.space.distance(x[:2],self.c[:2])
        if d <= self.r: return x
        return self.space.interpolate(x,self.c,(d-self.r)/d)

    def signedDistance(self,x):
        d = self.space.distance(x[:2],self.c[:2])
        return d - self.r


def dubinsCarTest():
    cspace = Geometric2DCSpace()
    vspace = BoxConfigurationSpace([-1,-1],[1,1])
    start = [0.5,0.3,0]
    goal = [0.5,0.7,0]
    controlSpace = DubinsCarSpace(MultiConfigurationSpace(cspace,SO2Space()))
    #controlSpace.setDistanceBounds(-0.25,0.25)
    controlSpace.setDistanceBounds(-0.1,0.1)
    controlSpace.setTurnRateBounds(-3.14,3.14)
    numControlsPerSample = 1
    if numControlsPerSample > 1:
        controlSpace = RepeatedControlSpace(controlSpace,numControlsPerSample)
    objective = DubinsCarDistanceObjectiveFunction(numControlsPerSample)
    goalRadius = 0.1
    return PlanningProblem(controlSpace,start,goal,
                           objective=objective,
                           visualizer=DubinsVisualizer(cspace),
                           goalRadius=goalRadius,
                           costLowerBound=lambda x,y:vectorops.distance(x[:2],y[:2]))

def dubinsTest2():
    cspace = Geometric2DCSpace()
    cspace.addObstacle(Box(0.5,0.4,0.9,0.7))
    cspace.addObstacle(Box(0.0,0.4,0.2,0.7))
    cspace.addObstacle(Box(0.0,0.9,1.0,1.0))
    vspace = BoxConfigurationSpace([-1,-1],[1,1])
    start = [0.8,0.3,0]
    goal = [0.8,0.8,math.pi]
    controlSpace = DubinsCarSpace(MultiConfigurationSpace(cspace,SO2Space()))
    controlSpace.setDistanceBounds(-0.25,0.25)
    controlSpace.setTurnRateBounds(-3.14,3.14)
    numControlsPerSample = 1
    if numControlsPerSample > 1:
        controlSpace = RepeatedControlSpace(controlSpace,numControlsPerSample)
    objective = DubinsCarDistanceObjectiveFunction(numControlsPerSample)
    goalRadius = 0.1
    return PlanningProblem(controlSpace,start,goal,
                           objective=objective,
                           visualizer=DubinsVisualizer(cspace),
                           goalRadius=goalRadius,
                           costLowerBound=lambda x,y:vectorops.distance(x[:2],y[:2]))

def dubinsMazeTest(n_divs=7, goal_setting='fixed'):
    d=2
    box_width = 1/(2*n_divs)
    start=[box_width for _ in range(d)] + [math.pi/2]
    if goal_setting == 'fixed':
        goal=[1-box_width for _ in range(d)] + [0]
    elif goal_setting == 'random':
        x = (1 + 2*np.random.randint(n_divs))/(2*n_divs)
        y = (1 + 2*np.random.randint(n_divs))/(2*n_divs)
        goal=[x, y]
    else: 
        assert False, 'Goal setting must be either \"fixed\" or \"random\"'
    goalRadius = box_width

    cspace = make_maze_space(n_divs, d)

    vspace = BoxConfigurationSpace([-1,-1],[1,1])

    controlSpace = DubinsCarSpace(MultiConfigurationSpace(cspace,SO2Space()))
    dist = 0.8/n_divs
    turn_rate = 0.5*math.pi*n_divs
    # controlSpace.setDistanceBounds(-0.25,0.25)
    # controlSpace.setTurnRateBounds(-3.14,3.14)
    controlSpace.setDistanceBounds(-dist,dist)
    controlSpace.setTurnRateBounds(-turn_rate,turn_rate)
    numControlsPerSample = 1
    if numControlsPerSample > 1:
        controlSpace = RepeatedControlSpace(controlSpace,numControlsPerSample)
    objective = DubinsCarDistanceObjectiveFunction(numControlsPerSample)
    # goalRadius = 0.1
    # goal = DubinsSubset(cspace, goal, goalRadius)
    return PlanningProblem(controlSpace,start,goal,
                           objective=objective,
                           visualizer=DubinsVisualizer(cspace),
                           goalRadius=goalRadius,
                           costLowerBound=lambda x,y:vectorops.distance(x[:2],y[:2]))
