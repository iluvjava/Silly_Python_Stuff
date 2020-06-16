### Here we define the problem of the TSP.
###   * TSP problem consists of list of points in 2d, and distance between 2 points is the Euclidean
###   * distance between the 2 points.

import math as math
import random as rnd
from pulp import *
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


class TSPPoints:
    """
        A collection of points for the TSP in 2d for the TSP problem.

        !!!!!!!!!!!!!! INDEX STARTS WITH 1
    """
    def __init__(self):
        self.points = []
        self.edgeCosts = {}

    def size(self):
        """
            The number of points that are in the TSP problem.
        :return:
        """
        return len(self.points)

    def __iadd__(self, other):
        """
            Mutable!
        :param other:
            an instance of another point.
        :return:
            it self, but with the new point added.
        """
        isinstance(other, Point)
        if other in self.points:
            return
        n = len(self.points) + 1
        for I in range(0, n - 1):
            d = dis(self.points[I], other)
            self.edgeCosts[I + 1, n] = self.edgeCosts[n, I + 1] = d
        self.points.append(other)
        return self

    def __getitem__(self, item):
        """
            Get points by giving the index of it.
            e.g:
                [a, b], gives
        :param item:
        :return:
        """
        if type(item) == tuple:
            assert len(item) == 2, "Must index with 2 integers "
            return dis(self.points[item[0] - 1], self.points[item[1] - 1])
        return self.points[item - 1]

    def get_edge_costs(self):
        """
        :return:
            A map mapping edge to the weight of the edge.
        """
        return self.edgeCosts


class TravelingSalesmanLP:
    """
        Find the shortest path with m vertices in the graph
        * Cost of all edges are assumed to be positive.
        * Store a lost variables with their string representation and the actual variable in the python.

        Decisions Variables:
            x_i_j: at point j at step i.

            1 <= j <= n
            1 <= i <= m

            e_a_b: the edge a, b has been included into the solution.

        LP Constraints:

            at each step we visit exactly one of the m vertex.
            \sum_{j = 1}^n = 1   (x_{i, j})
                for 1 <= i <= m

            for all vertex, we visit it in only one time.
            \sum_{i = 1}^m <= 1  (x_{i, j})
                for 1 <= j <= n

            We select edge e_{a, b} into our solution if and only if we vistied 'a'
            vertex at step 'i' and b at step 'i + 1'
            e_{a, b} >= x_{i, a} + x_{i + 1, b} - e_{a, b}
                for 1<= i <= m - 1

        :param points:
            An instance of the class TSPPoints.
        :param m:
            The number of vertices you want to visit in the plane.
    """
    def __init__(self):
        # all x_{i, j} variables:
        self.points = TSPPoints()

    def get_lp(self, m = None):
        """

        :param m:
            The number of vertice want to travel within the path.
        :return:
            The LP.
        """
        n = self.points.size()
        m = m if m is not None else n
        assert 1 < m and m <= n, "Invalid TSP problem"

        self.P = LpProblem(name="TSP2dPoints", sense=LpMinimize)
        self.x = LpVariable.dict("x", (range(1, m + 1), range(1, n + 1)), cat=LpBinary)
        # Self, double direction edges.
        self.e = LpVariable.dict("e", (range(1, n + 1), range(1, n + 1)), cat=LpBinary)

        for I in range(m):
            self.P += lpSum([self.x[I + 1, J + 1] for J in range(n)]) == 1
        for J in range(n):
            self.P += lpSum(self.x[I + 1,J + 1] for I in range(m)) <= 1
        for A in range(n):
            for B in range(n):
                for I in range(m - 1):
                    self.P += lpSum((self.x[I + 1, A + 1] + self.x[I + 2, B + 1] - self.e[A+ 1, B + 1])) <= 1

        ObjIdx = []
        for A in range(n):
            for B in range(n):
                if B == A:
                    continue
                ObjIdx.append((A,B))

        self.P += lpSum([self.e[A + 1, B + 1]*self.points.get_edge_costs()[A + 1, B + 1]\
                         for A, B in ObjIdx])
        return self.P

    def __iadd__(self, other: Type[Point]):
        self.points += other
        return self


def dis(a, b):
    """
        Euclidean distance between 2 points.
    :param a:
    :param b:
    :return:
    """
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


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


def main():
    print("Checking... ")
    tp = TSPPoints()
    tp += Point(1, 1)
    tp += Point(1, 2)
    tp += Point(1, 3)
    tp += Point(1, 4)
    for p in rand_points([0, 1], [1, 0], 10):
        tp += p

    print(tp.edgeCosts)
    print(tp.points)


if __name__ == "__main__":
    main()