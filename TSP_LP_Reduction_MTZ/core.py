#   Miller Tucker Zemlin's Linear Programming Reduction of the Traveling Salesman Problem.
import math as math
import random as rnd
from graph.simple_digraph import *
from typing import *
from pulp import *
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


class TravelingSalesManLP(FullGraph2D):
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

        Additional Features:
        1. Greedy Coloring For forst solve
        2. Warm start for solving after changes has been made...
            * if the path is there, then each time a new added vertex will be included into
            previous best solution.
            * next time upon evalution, model will provides with this path for mid-starting the algorithm.
    """
    def __init__(self):
        """

        :param Granulerization:
            A boolean option, if this is set to false, then the algorithm will use the shortest
            distance between any 2 pair of points to measure distance to all other pairs,

            Granularizing the distance between vertices might speed up???????

        """
        self._changes = False # True means that the problem has been changed.
        self._solved = False # true means it has solved it at the first time.
        self._granular = False # APPROXIMATE/PRECISE distances between the vertices.
        self._path = [] # Previously solved path, None if previouly unsolve.
        super().__init__()

    def formulate_lp(self):

        n = self.size()
        assert n >= 3, "The problem is too small."
        self.P = LpProblem(sense=LpMinimize)
        self.u = LpVariable.dict("u", range(1, n), cat=LpInteger, lowBound=0, upBound=n - 1)
        EdgeIndexList = [(I, J) for I in range(n) for J in range(n) if I != J]

        self.x = LpVariable.dict("x", (range(n), range(n)), cat=LpBinary)
        # number of Incoming edges in path is exactly 1.
        for J in range(n):
            self.P += lpSum([self.x[I, J] for I in range(n) if I != J]) == 1
        # Outcoming edges in path is exactly 1
        for I in range(n):
            self.P += lpSum([self.x[I, J] for J in range(n) if J != I]) == 1
        # Excluding one vertex, the path must be simple!
        for I in range(1, n):
            for J in [J for J in range(1, n) if J != I]:
                self.P += lpSum(self.u[I] - self.u[J] + n*self.x[I, J]) <= n - 1
        # setting Object function:
        self.P += lpSum([self.c(I,J)*self.x[I, J] for I, J in EdgeIndexList])

    def c(self, I, J):
        V1 = self[I]
        V2 = self[J]
        if self._granular:
            MinDis = min(self._E.values())
            return dis(V1, V2)//MinDis
        return dis(V1, V2)

    def solve_path(self):
        n = self.size()
        assert n >= 3, "The problem is too small."
        def warm_start(): # after formulating the LP
            while self._path[0] != 0:
                self._path.append(self._path.pop(0))
            for I in range(1, n):
                self.u[self._path[I]].setInitialValue(I - 1)
            FeasiblePath = [(V1, V2) for V1, V2 in zip(self._path, self._path[1:] + [self._path[0]])]
            for I, J in [(I, J) for I in range(n) for J in range(n) if I != J]:
                self.x[I, J].setInitialValue(1 if (I, J) in FeasiblePath else 0)

        # Lazy Evalution address changes.
        if (not self._changes) and (self._solved):
            return self._path
        self._changes = False # The formulation will address the changes.
        self._solved = True

        self.formulate_lp()
        warm_start()
        status = self.P.solve(solver=PULP_CBC_CMD(msg=True, fracGap=0.05, maxSeconds=300, mip_start=True))
        assert status == 1, f"LP status not good: {LpStatus[status]}"
        # Interpret solution, which is a path.
        Path = [0] # all vertex must be in the solution

        while len(Path) != n:
            for J in range(n):
                I = Path[-1]
                if self.x[I, J].varValue == 1:
                    Path.append(J)
                    break # optional.
        self._path = Path
        return Path

    def plot_path(self):
        pyplt.clf()
        Path = self.solve_path()
        for V1, V2 in zip(Path[:-1], Path[1:]):
            pyplt.scatter(self[V1].x, self[V1].y)
            pyplt.scatter(self[V2].x, self[V2].y)
            pyplt.plot([self[V1].x, self[V2].x], [self[V1].y, self[V2].y])
        V_n, V0 = Path[0], Path[-1]
        pyplt.plot([self[V_n].x, self[V0].x], [self[V_n].y, self[V0].y], '--')
        pyplt.show()

    def __iadd__(self, other):
        n = self.size()
        super().__iadd__(other)
        if n <= 2:
            self._path.append(n)
        else:
            Closiest = float("+inf")
            Iclose = -1
            for I, V in enumerate(self._path):
                if self.c(I, n) < Closiest:
                    Closiest = self.c(I, n)
                    Iclose = I
            self._path.insert(Iclose, n)
        self._changes = True
        return self

    def granular_on(self):
        self._granular = True
        self._changes = True

    def granular_off(self):
        self._granular = False
        self._changes = True


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

def main():
    RandPoints = rand_points([0, 10], [10, 0], 10)
    FullG = TravelingSalesManLP()
    for P in RandPoints:
        FullG += P
        print(FullG._path)

    print(FullG.solve_path())
    FullG.plot_path()

    for I in range(10):
        for P in rand_points([0, 10], [10, 0], 3):
            FullG += P
        print(FullG.solve_path())
        FullG.plot_path()

    pass

if __name__ == "__main__":
    main()