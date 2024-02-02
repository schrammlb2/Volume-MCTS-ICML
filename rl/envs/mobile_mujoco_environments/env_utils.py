import numpy as np
from math import cos, sin
from scipy.spatial.transform import Rotation as R

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def quat2euler(q_mj):
    q_scipy = np.array([q_mj[1], q_mj[2], q_mj[3], q_mj[0]])
    r = R.from_quat(q_scipy)
    return r.as_euler('xyz', degrees=False)

def rot_vector(theta, v):
    rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    return np.dot(rot, v)
