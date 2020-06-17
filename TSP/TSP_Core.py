### Here we define the problem of the TSP.
###   * TSP problem consists of list of points in 2d, and distance between 2 points is the Euclidean
###   * distance between the 2 points.
### Using the Cplex Solver:
### https://stackoverflow.com/questions/47985247/time-limit-for-mixed-integer-programming-with-python-pulp

### Solver Arguments:
### def __init__(self, path = None, keepFiles = 0, mip = 1,
###            msg = 0, cuts = None, presolve = None, dual = None,
###            strong = None, options = [],
###            fracGap = None, maxSeconds = None, threads = None, mip_start=False):

import math as math
import random as rnd
from pulp import *
from typing import *
import matplotlib.pyplot as pyplt

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


class FullGraph2DPoints:
    """
        A collection of points for the TSP in 2d for the TSP problem.
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
        n = len(self.points)

        for I in range(n):
            d = dis(self.points[I], other)
            self.edgeCosts[I, n] = self.edgeCosts[n, I] = d
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
            return dis(self.points[item[0]], self.points[item[1]])
        return self.points[item]

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
        self.points = FullGraph2DPoints()
        self.path = None
        self.last_m = None # also tells if the problem has be solved or not.

    def get_lp(self, m = None):
        """
            Variables are going to be indexed from 0.
            Get an unsolved LP of the TSP problem.
        :param m:
            The number of vertice want to travel within the path.
        :return:
            The LP.
        """
        n = self.points.size()
        m = m if m is not None else n
        assert 1 < m and m <= n, "Invalid TSP problem"

        self.P = LpProblem(name="TSP2dPoints", sense=LpMinimize)
        self.x = LpVariable.dict("x", (range(m), range(n)), cat=LpBinary)
        # Self, double direction edges.
        self.e = LpVariable.dict("e", (range(n), range(n)), cat=LpBinary)

        for I in range(m):
            self.P += lpSum([self.x[I, J] for J in range(n)]) == 1
        for J in range(n):
            self.P += lpSum(self.x[I,J] for I in range(m)) <= 1
        for A in range(n):
            for B in [b for b in range(n) if b != A]: # eliminate self-edge.
                for I in range(m - 1):
                    self.P += lpSum((self.x[I, A] + self.x[I + 1, B] - self.e[A, B])) <= 1

        ObjIdx = []
        for A in range(n):
            for B in range(n):
                if B != A:
                    ObjIdx.append((A,B))

        self.P += lpSum([self.e[A, B]*self.points.get_edge_costs()[A, B]\
                         for A, B in ObjIdx])
        print("Lp has been seted up, time to call the solver.")
        return self.P

    def solve_get_path(self, m=None, mode = 0):
        """
            Solve the LP and get the path back.

            A path is a collection of tuples, where each tuples corresponds to indices of the vertex
            in the full graph.
        :param m:
            The number of vertices involved in the path.
        :param mode:
            The mode you want for the solver.
            mode == 0:
                default solver no parameters.
            mode == 1:
                Fancy approximations for big inputs and shit.
        :return:
        """
        # Lazy Evaluation.
        m = self.points.size() if m is None else m # Assume full path here.

        P = self.get_lp(m)
        if self.last_m == m:
            return
        else:
            self.last_m = m

        # fancy solve
        if mode == 1:
            print("Solver Type: " ,LpSolverDefault)
            P.solve(PULP_CBC_CMD(msg=True, maxSeconds = 300, fracGap=0.2))
        # not fancy solve.
        P.solve()
        self.status = P.status
        return self.__get_path()

    def plot_show(self):
        pyplt.clf()
        self.__plot_all_points()
        Path = self.path
        for Edge in Path:
            P1, P2 = self.points[Edge[0]], self.points[Edge[1]]
            pyplt.plot([P1.x, P2.x],[P1.y, P2.y])
        pyplt.show()


    def __get_path(self):
        assert self.status == 1, f"Status: {LpStatus[self.status]}, status Code: {self.status} "
        EdgesInPath = []
        for e, v in self.e.items():
            if v.varValue == 1:
                EdgesInPath.append(e)
        self.path = EdgesInPath
        return EdgesInPath

    def __plot_all_points(self):
        X, Y = [], []
        for Point in self.points:
            X.append(Point.x)
            Y.append(Point.y)
        pyplt.scatter(X, Y)

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


def unit_circle(n = 10):
    """

    :return:
        Get points on a unit circle.
    """
    cos = math.cos
    sin = math.sin
    pi = math.pi
    circle = [Point(cos((2*pi/n)*i), sin((2*pi/n)*i)) for i in range(n)]
    return circle

def city_grid_points(width = 1, height = 1):
    pass

def main():
    Cities = unit_circle(8) + rand_points([0, 1], [1, 0], 8)
    TSPProblem = TravelingSalesmanLP()

    for Point in Cities:
        TSPProblem += Point

    print(TSPProblem.solve_get_path(mode=1))
    TSPProblem.plot_show()


if __name__ == "__main__":
    main()