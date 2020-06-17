#   Some of the core stuff for the LP reduction of the MST problem.

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


class MinimumSpanningTreeLP:
    """
        n: |V|
        e_{i, j}: 1 <= i < j <= n, if the edge is in the MST
        x_{i}: 1 <= i <= n, if the vertex is in the MST
        c(i, j): Cost function, all edges are assumed to be positive.

        \sum_{i = 1}^n (x_i) = n // All index get used in the MST.

        // an edge is in MST if and only if both edge of its vertices are in MST too.
        for all 1 <= i < j <= n
            x_i + x_j - e_{i, j} <= 1

        // sum of the edges in MST:
        \sum_{1 \leq i < j \leq n} e_{i, j} = n - 1

        // Objective:
        min sum_{1 \leq i < j \leq n} c(i, j)e_{i, j}
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