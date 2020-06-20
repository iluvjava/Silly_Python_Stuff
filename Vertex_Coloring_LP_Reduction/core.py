"""
    In this file we are going to implement the POP2 Hybrid LP Reduction for the vertex
    coloring problems.

    https://arxiv.org/pdf/1706.10191.pdf

    This is the research paper containing the POP2 Hybrid reduction model.


"""

from graph import simple_digraph
from pulp import *

from graph.simple_digraph import *


class VertexColoring(SimpleDiGraph):

    def __init__(self):
        self.__colorAssignment = None
        pass

    def formulate_lp(self):
        """
            Formulation of the LP:
        :return:

        """
        def maxdeg():
            return max(len(L) for L in self._AdjLst.values())

        H = maxdeg() + 1
        n = self.size()
        self.P = LpProblem("Vertex_coloring_POP2Hybrid", sense=LpMinimize)
        self.y = LpVariable.dict("y", (range(H), range(n)), cat=LpBinary)
        self.z = LpVariable.dict("z", (range(H), range(n)), cat=LpBinary)
        self.x = LpVariable.dict("x", (range(H), range(n)), cat=LpBinary)

        # prune self edge:
        for I in range(n):
            del self.y[I, I], self.x[I, I], self.z[I, I]
        # Objective fxn
        self.P += lpSum([self.y[C, 0] for C in range(H)]) # minimize rank of the color on 0th vertex.
        for V in range(n):
            self.P += self.z[1, V] == 0
            self.P += self.y[H, V] == 0
            for C in range(H):
                self.P += self.x[C, V] + self.y[C, V] + self.z[C, V] == 1
            for C in range(H - 1):
                self.P += self.y[C, V] - self.y[C + 1, V] >= 0
                self.P += self.y[C, V] + self.y[C + 1, V] == 1
                self.P += self.y[C, 0] - self.y[C, V] >= 0
        for C in range(H):
            for U, V in self._E.keys():
                self.P += self.x[C, V] + self.x[C, U] <= 1

        return self.P



def main():
    pass

if __name__ == "__main__":
    main()