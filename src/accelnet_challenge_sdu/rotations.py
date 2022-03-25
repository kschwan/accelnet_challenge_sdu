import math
import numpy as np


def Rx(a):
    """Basic rotation about the x axis with angle a"""
    ca = math.cos(a)
    sa = math.sin(a)

    return np.array([
        [1,   0,   0,   0 ],
        [0,   ca,  -sa, 0 ],
        [0,   sa,  ca,  0 ],
        [0,   0,   0,   1 ],
    ])


def Ry(a):
    """Basic rotation about the y axis with angle a"""
    ca = math.cos(a)
    sa = math.sin(a)

    return np.array([
        [ca,  0,   sa,  0 ],
        [0,   1,   0,   0 ],
        [-sa, 0,   ca , 0 ],
        [0,   0,   0,   1 ],
    ])


def Rz(a):
    """Basic rotation about the z axis with angle a"""
    ca = math.cos(a)
    sa = math.sin(a)

    return np.array([
        [ca,  -sa, 0,   0 ],
        [sa,  ca,  0,   0 ],
        [0,   0,   1,   0 ],
        [0,   0,   0,   1 ],

    ])


def cross_matrix(v):
    """Return 3x3 skew-symmetric matrix from a 3-dimensional vector.

    Can be used to represent cross products as matrix multiplication, e.g.
    v x u = [v]_x u, where [v]_x is a skew-symmetric matrix.
    """
    return np.array([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0],
    ])
