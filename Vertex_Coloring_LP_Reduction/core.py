"""
    In this file we are going to implement the POP2 Hybrid LP Reduction for the vertex
    coloring problems.

    https://arxiv.org/pdf/1706.10191.pdf

    This is the research paper containing the POP2 Hybrid reduction model.

    Features:
        * for this implementation we will try to use a greedy strategy to
        maintain the previous optimal solution, and in that case, we need to keep track of the
        changes of the problem size.

"""

from pulp import *

from graph.simple_digraph import *
from graph.point import *

import random as rnd

class VertexColoring(SimpleDiGraph):
    """
    The vertex coloring problem.
    * Each vertex is on the uni circle

    """

    def __init__(self):
        super().__init__()
        self._C = {} # Integer representing the vertex |-> Color assigned to the vertex.
        self.__colorAssignment = None
        pass

    def formulate_lp(self):
        """
            Formulation of the LP:
        :return:
            The formulated LP problem.
        """
        def maxdeg():
            return max(len(L) for L in self._AdjLst.values())

        n = self.size()
        H = n + 1
        self.P = LpProblem("Vertex_coloring_POP2Hybrid", sense=LpMinimize)
        self.y = LpVariable.dict("y", (range(H), range(n)), cat=LpBinary)
        self.z = LpVariable.dict("z", (range(H), range(n)), cat=LpBinary)
        self.x = LpVariable.dict("x", (range(H), range(n)), cat=LpBinary)

        # Objective fxn
        # Minimizes the color rank of the 0th vertex.
        self.P += lpSum([self.y[C, 0] for C in range(H)])
        for V in range(n):
            self.P += self.z[0, V] == 0
            self.P += self.y[H - 1, V] == 0
            for C in range(H):
                self.P += self.x[C, V] + self.y[C, V] + self.z[C, V] == 1
            for C in range(H - 1):
                self.P += self.y[C, V] - self.y[C + 1, V] >= 0
                self.P += self.y[C, V] + self.z[C + 1, V] == 1
                self.P += self.y[C, 0] - self.y[C, V] >= 0
        for C in range(H):
            for U, V in self._E.keys():
                self.P += self.x[C, V] + self.x[C, U] <= 1
        return self.P

    def solve_color(self):
        LP = self.formulate_lp()
        status = LP.solve()
        assert status == 1, f"Status failed as: {LpStatus[status]}"
        y = self.y
        z = self.z

        pass

    def plot(self):
        pass



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


def randomG(n, p):
    """
        Method creates a random graph with n vertices and an edge density of p.
        The meta information associated with the vertices are 2D Points.
    :param n:
        The number of vertices
    :param p:
        The edge density of the vertex graph.
        0 <= p < = 1
    :return:
        The constructed random graph.
    """
    Gcoloring = VertexColoring()
    Points = unit_circle(n, 10)
    for P in Points:
        Gcoloring += P

    for I in range(len(Points)):
        for J in range(len(Points)):
            if I == J:
                continue # No self edge.
            if rnd.random() < p:
                Gcoloring.connect_by_idx(J, I)
                Gcoloring.connect_by_idx(I, J)
    return Gcoloring

def main():
    print("Testing out the random graph generation: ")
    RndG = randomG(10, 0.5)
    print(RndG)
    RndG.solve_color()

    pass

if __name__ == "__main__":
    main()