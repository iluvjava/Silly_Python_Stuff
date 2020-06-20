"""
    This is a class that contains basic stuff for modeling 2d points,
    it's used for visualizing graphs on the 2d plane.

"""
import math
import random as rnd

class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if other.x != self.x:
            return False
        return other.y == self.y

    def __repr__(self):
        return f"({self.x}, {self.y})"

    def __hash__(self):
        return super().__hash__()


def rand_points(topLeft, bottomRight, n):
    """

    :param topLeft:
    :param bottomRight:
    :param n:
    :return:
    """
    assert topLeft[0] < bottomRight[0] and topLeft[1] > bottomRight[1]
    def randPointInSquare():
        x = rnd.random()*(bottomRight[0] - topLeft[0]) + topLeft[0]
        y = rnd.random()*(topLeft[1] - bottomRight[1]) + bottomRight[1]
        return  Point(x, y)
    return [randPointInSquare() for I in range(n)]

def unit_circle(n = 10, r = 1):
    """

    :return:
        Get points on a unit circle.
    """
    cos = math.cos
    sin = math.sin
    pi = math.pi
    circle = [Point(r*cos((2*pi/n)*i), r*sin((2*pi/n)*i)) for i in range(n)]
    return circle

def dis(a, b):
    """
        Euclidean distance between 2 points.
    :param a:
    :param b:
    :return:
    """
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)
