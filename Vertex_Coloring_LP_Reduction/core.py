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
import matplotlib.pyplot as pyplt
import random as rnd

class VertexColoring(SimpleDiGraph):
    """
    The vertex coloring problem.
    * Great visualization!
    * You can associated the vertex with anything the helps with the
    problems that are trying to solve.
    * It keeps a greedy coloring solution, which serves as a warm-start
    for the simplex.

    ! When connecting vertices to the graph, please do it for both
    direction.

    """

    def __init__(self):
        super().__init__()
        # Integer representing the vertex |-> Color assigned to the vertex.
        self.__colorAssignment = None
        self.__solved = False
        self.__changes = False
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
        assert n <= 100, "Problem is kinda huge; another model is needed. "
        H = maxdeg() + 1 # Can be optimized better.
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
        """
            Get a color assignment plan for all the vertices.

        :return:
        """
        def warm_start():
            """
                Interpret the color assignment and then initialize the variables.
            :return:
            """
            pass

        if self.__solved and not self.__changes:
            return
        self.__solved = True
        self.__changes = False

        n = self.size()
        LP = self.formulate_lp()
        status = LP.solve(PULP_CBC_CMD(msg = True, maxSeconds = 300, fracGap = 0.05))
        assert status == 1, f"Status failed as: {LpStatus[status]}"
        ColorAssignment = {}
        for V in range(n):
            for C in range(n):
                if self.x[C, V].varValue == 1:
                    ColorAssignment[V] = C
                    break
        self.__colorAssignment = ColorAssignment
        return ColorAssignment

    def plot(self):
        """
            Visualize the plot graph, flattened on the unit disk and make
            mark all vertices with the integer assignment of the color.
            * The visualization depends on the number of points we have on the graph.
        :return:
            None
        """
        self.solve_color()
        pyplt.clf()
        PointsX, PointsY = [], []

        def match_vertices_to_points():
            n = self.size()
            M = {}
            for I, P in enumerate(unit_circle(n)):
                M[I] = P
            return M
        M = match_vertices_to_points()
        for U, V in self._E.keys():
            V1 = M[U]
            V2 = M[V]
            pyplt.plot([V1.x, V2.x], [V1.y, V2.y], "g", linewidth = 0.25)
        for I in range(self.size()):
            V = M[I]
            PointsX.append(V.x)
            PointsY.append(V.y)
            pyplt.annotate(f"{self.__colorAssignment[I]}", (V.x, V.y), color="r")
        pyplt.scatter(PointsX, PointsY, color="b")
        pyplt.show()

    def __repr__(self):
        res = super().__repr__()
        res += "Color Assignment: \n"
        if self.__colorAssignment is None:
            res += "Color hasn't been assigned yet. "
            return res
        for V in range(self.size()):
            res += f"{V}: {self.__colorAssignment[V]}\n"
        return res

    def __iadd__(self, other):
        """
            Tries to maintains the optimal solution.
            * By default, the 0th vertex has no color, represented by a color value of -1
            * and the nth vertex has color n - 1
        :param other:
        :return:
        """
        n = self.size()
        if n == 0:
            self.__colorAssignment = {}
            self.__colorAssignment[0] = -1
        else:
            self.__colorAssignment[n] = n # New added color has index of n.
        return super(VertexColoring, self).__iadd__(other)

    def connect_by_idx(self, v1, v2, edge=None):
        """
            This function will call the super class 2 times, so that it adds a undirected graph.
        :param v1:
            vertex 1
        :param v2:
            vertex 2
        :param edge:
            The generic edge value to associate with.
        :return:
            Nothing
        """
        super(VertexColoring, self).connect_by_idx(v1, v2, edge=None)
        super(VertexColoring, self).connect_by_idx(v2, v1, edge=None)
        # figure out the new coloring strategy...




def unit_circle(n = 10, r = 1, offset = 0):
    """
        Points on the Circle of Unity
    :return:
        Get points on a unit circle.
    """
    cos = math.cos
    sin = math.sin
    pi = math.pi
    alpha = 2*pi*offset
    circle = [Point(r*cos(offset + (2*pi/n)*i), r*sin(offset + (2*pi/n)*i)) for i in range(n)]
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
    return Gcoloring


def main():
    print("Testing out the random graph generation: ")
    RndG = randomG(12, 0.2)
    RndG.solve_color()
    RndG.plot()


if __name__ == "__main__":
    main()