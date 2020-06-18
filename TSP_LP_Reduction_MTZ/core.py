#   Miller Tucker Zemlin's Linear Programming Reduction of the Traveling Salesman Problem.
import math as math
import random as rnd
from graph.simple_digraph import *
from typing import *



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


class FullGraph2D(SimpleDiGraph):

    def __iadd__(self, p: Type[Point]):
        n = self.size()
        super().__iadd__(p)
        for I in range(n):
            V = self[I]
            self.connect_by_idx(I, n, dis(self._V[I], self._V[n]))
            self.connect_by_idx(n, I, dis(self._V[I], self._V[n]))
        return self


class TravelingSalesManLP:
    """
        This reduction has a polynomial number of constraints applied to the system, and it's formulated by
        Miller Tucker Zemlin.
        Here are the variable:
            x_{i, j}: Going from city i to city j at some point during the tour.
                * Binary
                * Direction counts.
            u_i: The step city i has been visited.
                2 <= i <= n
                u_i = t, then it means that city i is visted at t step of the tour.
                0 <= u_i <= n - 1

    """

    def __init__(self):
        pass

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

def dis(a, b):
    """
        Euclidean distance between 2 points.
    :param a:
    :param b:
    :return:
    """
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def main():
    RandPoints = rand_points([0, 10], [10, 0], 10)
    FullG = FullGraph2D()
    for P in RandPoints:
        FullG += P
    print(FullG)
    pass

if __name__ == "__main__":
    main()