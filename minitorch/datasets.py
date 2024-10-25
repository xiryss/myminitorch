import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N):
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N):
    """
    Classifies points with respect to line x = 0.5
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N):
    """
    Classifies points with respect to line x + y = 0.5
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N):
    """
    Classifies point to class 0, if it is between lines x = 0.2, x = 0.8, otherwise to class 1
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N):
    """
    Classifies point to class 1, if it is in 2-nd or 4-th quadrant, otherwise 0
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N):
    """
    Classifies points according to their relation to circle centered at (0.5,0.5) with certain radius
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N):

    """
    Classifies points according to their relation to some spiral.
    """
    def x(t):
        return t * math.cos(t) / 20.0

    def y(t):
        return t * math.sin(t) / 20.0
    X = [(x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N //
        2))) + 0.5) for i in range(5 + 0, 5 + N // 2)]
    X = X + [(y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) /
        (N // 2))) + 0.5) for i in range(5 + 0, 5 + N // 2)]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {'Simple': simple, 'Diag': diag, 'Split': split, 'Xor': xor,
    'Circle': circle, 'Spiral': spiral}
