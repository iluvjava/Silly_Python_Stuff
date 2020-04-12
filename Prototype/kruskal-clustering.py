"""
    Investiagating the size of the cluster when running Kruskal
    algorithm for 2 normal distributions of special points.

    * Points are in at least dimension of 3.
    * 2 normal distributions are resonably apart.
"""

import numpy as np
from disjoint_set import DisjointSet
import math
from typing import *

class FullGraph:

    def __init__(self, points:List):
        points = points.copy()

        self.V = dict(zip(len(points), points)) # index to spacial points.
        self.E = [] # (i, j, cost)

        for I in range(len(points)):
            for J in range(I + 1, len(points)):
                self.E.append((I, J, dis(self.V[I], self.V[J])))

        # Sort by edge cost
        self.E = sorted(self.E, key=lambda x: x[2])






def dis(x, y):
    return math.sqrt(sum((a-b)**2 for a,b in zip(x, y)))

def SpacialPointGenerate(center, sigma, samplesize):
    XcoordList = np.random.normal(center[0], sigma, samplesize)
    YcoordList = np.random.normal(center[1], sigma, samplesize)
    ZcoordList = np.random.normal(center[2], sigma, samplesize)
    points = [(x, y, z) for x, y, z in zip(XcoordList, YcoordList, ZcoordList)]
    return points


def main():
    points = SpacialPointGenerate((0,0,0), 100, 30)

    g = FullGraph([(0, 0), (1.1, 0), (2.2, 0), (3.3, 0), (5.5, 0)])

    pass


if __name__ == "__main__":
    main()
