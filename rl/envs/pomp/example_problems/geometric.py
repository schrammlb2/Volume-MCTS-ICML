from OpenGL.GL import *
from ..klampt import gldraw
from ..spaces.objectives import *
from ..spaces.configurationspace import *
from ..spaces.edgechecker import *
from ..spaces.metric import *
from ..spaces.configurationspace import NeighborhoodSubset,SingletonSubset
from ..planners.problem import PlanningProblem
from ..example_problems.maze_generation import build_maze
import ipdb
import numpy as np

class Circle:
    def __init__(self,x=0,y=0,radius=1):
        self.center = (x,y)
        self.radius = radius
        
    def contains(self,point):
        return (vectorops.distance(point,self.center) <= self.radius)

    def signedDistance(self,point):
        return (vectorops.distance(point,self.center) - self.radius)
    
    def signedDistance_gradient(self,point):
        d = vectorops.sub(point,self.center)
        return vectorops.div(d,vectorops.norm(d))

    def drawGL(self,res=0.01):
        numdivs = int(math.ceil(self.radius*math.pi*2/res))
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(*self.center)
        for i in range(numdivs+1):
            u = float(i)/float(numdivs)*math.pi*2
            glVertex2f(self.center[0]+self.radius*math.cos(u),self.center[1]+self.radius*math.sin(u))
        glEnd()


class Box(BoxSet):
    def __init__(self,x1=0,y1=0,x2=0,y2=0):
        BoxSet.__init__(self,[min(x1,x2),min(y1,y2)],[max(x1,x2),max(y1,y2)])
        
    def drawGL(self):
        glBegin(GL_QUADS)
        glVertex2f(*self.bmin)
        glVertex2f(self.bmax[0],self.bmin[1])
        glVertex2f(*self.bmax)
        glVertex2f(self.bmin[0],self.bmax[1])
        glEnd()


class Geometric2DCSpace (BoxConfigurationSpace):
    def __init__(self):
        BoxConfigurationSpace.__init__(self,[0,0],[1,1])
        self.obstacles = []

    def addObstacle(self,obs):
        self.obstacles.append(obs)

    def feasible(self,x):
        if not BoxConfigurationSpace.feasible(self,x): return False
        for o in self.obstacles:
            if o.contains(x): return False
        return True

    def toScreen(self,q):
        return (q[0]-self.box.bmin[0])/(self.box.bmax[0]-self.box.bmin[0]),(q[1]-self.box.bmin[1])/(self.box.bmax[1]-self.box.bmin[1])

    def toState(self,x,y):
        return (self.box.bmin[0]+x*(self.box.bmax[0]-self.box.bmin[0]),
                self.box.bmin[1]+y*(self.box.bmax[1]-self.box.bmin[1]))

    def beginDraw(self):
        if self.box.bmin != [0,0] or self.box.bmin != [1,1]:
            glPushMatrix()
            glScalef(1.0/(self.box.bmax[0]-self.box.bmin[0]),1.0/(self.box.bmax[1]-self.box.bmin[1]),1)
            glTranslatef(-self.box.bmin[0],-self.box.bmin[1],0)

    def endDraw(self):
        if self.box.bmin != [0,0] or self.box.bmin != [1,1]:
            glPopMatrix()

    def drawObstaclesGL(self):
        self.beginDraw()
        glColor3f(0.2,0.2,0.2)
        for o in self.obstacles:
            o.drawGL()
        self.endDraw()

    def drawVerticesGL(self,qs):
        self.beginDraw()
        glBegin(GL_POINTS)
        for q in qs:
            glVertex2f(q[0],q[1])
        glEnd()
        self.endDraw()

    def drawRobotGL(self,q):
        glColor3f(0,0,1)
        glPointSize(7.0)
        self.drawVerticesGL([q])

    def drawGoalGL(self,goal):
        self.beginDraw()
        if isinstance(goal,NeighborhoodSubset):
            q = goal.c
            glColor3f(1,0,0)
            gldraw.circle(q,goal.r,filled=False)
            glPointSize(7.0)
            glBegin(GL_POINTS)
            glVertex2f(q[0],q[1])
            glEnd()
        elif isinstance(goal,SingletonSubset):
            q = goal.x
            glColor3f(1,0,0)
            glPointSize(7.0)
            glBegin(GL_POINTS)
            glVertex2f(q[0],q[1])
            glEnd()
        else:
            glColor3f(1,0,0)
            glPointSize(7.0)
            glBegin(GL_POINTS)
            for i in range(50):
                q = goal.sample()
                glVertex2f(q[0],q[1])
            glEnd()
        self.endDraw()

    def drawInterpolatorGL(self,interpolator):
        self.beginDraw()
        if isinstance(interpolator,LinearInterpolator):
            #straight line paths
            glBegin(GL_LINES)
            glVertex2f(interpolator.a[0],interpolator.a[1])
            glVertex2f(interpolator.b[0],interpolator.b[1])
            glEnd()
        elif isinstance(interpolator,PiecewiseLinearInterpolator):
            glBegin(GL_LINE_STRIP)
            for x in interpolator.path:
                glVertex2f(x[0],x[1])
            glEnd()
        else:
            glBegin(GL_LINE_STRIP)
            for s in range(10):
                u = float(s) / (9.0)
                x = interpolator.eval(u)
                glVertex2f(x[0],x[1])
            glEnd()
        self.endDraw()

    def clearance(self,x):
        res = BoxConfigurationSpace.clearance(self,x)
        for o in self.obstacles:
            res.append(o.signedDistance(x))
        return res
    
    def clearance_gradient(self,x):
        res = [BoxConfigurationSpace.clearance_gradient(self,x)]
        for o in self.obstacles:
            res.append(o.signedDistance_gradient(x))
        return np.vstack(res)


def circleTest():
    space = Geometric2DCSpace()
    space.addObstacle(Circle(0.5,0.4,0.39))
    start=[0.06,0.25]
    goal=[0.94,0.25]
    objective = PathLengthObjectiveFunction()
    goalRadius = 0.1
    return PlanningProblem(space,start,goal,
                           objective=objective,
                           visualizer=space,
                           costLowerBound = vectorops.distance,
                           goalRadius = goalRadius,
                           euclidean = True)


def rrtChallengeTest():
    w = 0.03
    eps = 0.01
    space = Geometric2DCSpace()
    space.box = BoxSet([0,0],[1,w*2+eps])
    space.addObstacle(Box(0,w*2+eps,1,1))
    space.addObstacle(Box(w,w,1,w+eps))
    start=[1-w*0.5,w+eps+w*0.5]
    goal=[1-w*0.5,w*0.5]
    goalRadius = w*0.5
    objective = PathLengthObjectiveFunction()
    return PlanningProblem(space,start,goal,
                           objective=objective,
                           visualizer=space,
                           costLowerBound = vectorops.distance,
                           goalRadius = goalRadius,
                           euclidean = True)

def kinkTest():
    space = Geometric2DCSpace()
    w = 0.02
    space.addObstacle(Box(0.3,0,0.5+w*0.5,0.2-w*0.5))
    space.addObstacle(Box(0.5+w*0.5,0,0.7,0.3-w*0.5))
    space.addObstacle(Box(0.3,0.2+w*0.5,0.5-w*0.5,0.7))
    space.addObstacle(Box(0.5-w*0.5,0.3+w*0.5,0.7,0.7))
    start=[0.06,0.25]
    goal=[0.94,0.25]
    goalRadius = 0.1
    objective = PathLengthObjectiveFunction()
    return PlanningProblem(space,start,goal,
                           objective=objective,
                           visualizer=space,
                           costLowerBound = vectorops.distance,
                           goalRadius = goalRadius,
                           euclidean = True)

def bugtrapTest():
    space = Geometric2DCSpace()
    w = 0.1
    """
    space.addObstacle(Box(0.55,0.25,0.6,0.75))
    space.addObstacle(Box(0.15,0.25,0.55,0.3))
    space.addObstacle(Box(0.15,0.7,0.55,0.75))
    space.addObstacle(Box(0.15,0.25,0.2,0.5-w*0.5))
    space.addObstacle(Box(0.15,0.5+w*0.5,0.2,0.75))
    start=[0.5,0.5]
    goal=[0.65,0.5]
    """
    start=[0.1,0.5]
    goal=[0.9,0.5]
    space.addObstacle(Box(0.3,0.3,0.7,0.7))
    goalRadius = 0.1
    objective = PathLengthObjectiveFunction()
    return PlanningProblem(space,start,goal,
                           objective=objective,
                           visualizer=space,
                           costLowerBound = vectorops.distance,
                           goalRadius = goalRadius,
                           euclidean = True)

def make_maze_space(n_divs, d):  
    shape = (n_divs,)*d
    w_base = 0.2
    w = w_base/n_divs
      
    show_paths = False#True
    nodes, edges = build_maze(n=n_divs, d=2)
    edge_list = [loc for loc, edge in np.ndenumerate(edges) if edge]

    def edge_to_wall(edge_loc):
        loc1 = tuple(edge_loc)[:2]
        loc2 = tuple(edge_loc)[2:]

        x_min = min([loc1[0], loc2[0]])/n_divs + 1/(2*n_divs)
        x_max = max([loc1[0], loc2[0]])/n_divs + 1/(2*n_divs)
        y_min = min([loc1[1], loc2[1]])/n_divs + 1/(2*n_divs)
        y_max = max([loc1[1], loc2[1]])/n_divs + 1/(2*n_divs)
        return Box(x_min - w/2, y_min - w/2, x_max + w/2, y_max + w/2)

    # print(edge_list)

    adjacency_list = []
    for loc, node in np.ndenumerate(nodes):
        aj_l_temp = []
        for x, edge in np.ndenumerate(edges[loc]):
            if edge: aj_l_temp.append(tuple(x))
        adjacency_list.append((loc, aj_l_temp))

    if d==2:
        space = Geometric2DCSpace()
    else: 
        space = BoxConfigurationSpace()

    if show_paths: 
        for edge in edge_list:
            space.addObstacle(edge_to_wall(edge))

    def get_positive_offset_nodes(node, sign=1):
        node_list = []
        for i in range(len(shape)):
            offset = np.zeros(len(shape), dtype='int64')
            offset[i] = sign
            other_node = offset + node
            if (other_node >= np.array(shape)).any():
                continue
            if edges[tuple(node) + tuple(other_node)]: 
                continue 
            else:
                node_list.append((i,other_node))
        if sign == 1: 
            node_list = node_list #+ get_positive_offset_nodes(node, -1)
        return node_list

    def create_box_from_edge(i, n1, n2):
        bmin = [(n1[j] + (1 if i == j else 0)-w_base/2)/n_divs for j, x in enumerate(n1)]
        bmax = [(n1[j] + 1 + w_base/2)/n_divs for j, x in enumerate(n1)]
        if d==2:
            return Box(bmin[0], bmin[1], bmax[0], bmax[1])
        else: 
            return BoxSet(bmin, bmax)

    if not show_paths:
        for node, node_val in np.ndenumerate(nodes):
            adjacent_node_list = get_positive_offset_nodes(node)
            for i, other_node in adjacent_node_list:
                box = create_box_from_edge(i, node, other_node)
                space.addObstacle(box)
            # ipdb.set_trace()

    return space

def mazeTest(n_divs=10, goal_setting='fixed'):
    # n_divs = 10
    d=2
    box_width = 1/(2*n_divs)
    start=[box_width for _ in range(d)]
    if goal_setting == 'fixed':
        goal=[1-box_width for _ in range(d)]
    elif goal_setting == 'random':
        x = (1 + 2*np.random.randint(n_divs))/(2*n_divs)
        y = (1 + 2*np.random.randint(n_divs))/(2*n_divs)
        goal=[x, y]
    else: 
        assert False, 'Goal setting must be either \"fixed\" or \"random\"'
    goalRadius = box_width

    space = make_maze_space(n_divs, d)


    objective = PathLengthObjectiveFunction()
    return PlanningProblem(space,start,goal,
                           objective=objective,
                           visualizer=space,
                           costLowerBound = vectorops.distance,
                           goalRadius = goalRadius,
                           euclidean = True)
