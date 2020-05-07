"""
    Investiagating the size of the cluster when running Kruskal
    algorithm for 2 normal distributions of special points.

    * Points are in at least dimension of 3.
    * 2 normal distributions are reasonably apart.
"""

__all__ = ["NdimSpacialpointRnorm", "FullGraph"]

import numpy as np
from disjoint_set import DisjointSet
import math
from typing import *

class FullGraph:

    def __init__(self, points:List):
        points = points.copy()
        Vcount = len(points)

        self.Vcount = Vcount

        # index |-> (x,y)
        self.V = dict(zip(range(Vcount), points))
        self.E = [] # (i, j, cost)

        for I in range(Vcount):
            for J in range(I + 1, Vcount):
                self.E.append((I, J, dis(self.V[I], self.V[J])))

        # Sort by edge cost
        self.E = sorted(self.E, key=lambda x: x[2])
        self.EChoose, self.MaxPartitionEvolve = self.__KruChoose()

    def __KruChoose(self):
        PartitionSize = dict(zip(range(self.Vcount), [1]*self.Vcount))
        ListofMaxPartitionSizes = []
        Echoose = []
        # Components = self.Vcount
        ds = DisjointSet()
        for I, J, W in self.E:
            if ds.find(I) != ds.find(J):
                MergedSize = PartitionSize[ds.find(I)] + PartitionSize[ds.find(J)]
                ds.union(I, J)
                # PartitionSize[I] = 0
                PartitionSize[ds.find(J)] = MergedSize
                ListofMaxPartitionSizes.append(max(PartitionSize.values()))
                Echoose.append(True)
                # Components -= 1
                # if Components == 1:
                #     break
                continue
            Echoose.append(False)
        return Echoose, ListofMaxPartitionSizes


    def __repr__(self):
        s = f"This is the Vertex Dictionary: \n {self.V} \n"
        s += f"This is the edge list, sorted by weight: \n {self.E} \n"
        s += f"This is the list of edges chose by Kruskal: \n"
        s += f"{self.EChoose}\n"
        s += f"Kruskal Max Evolve: {self.MaxPartitionEvolve} \n"
        return s


def dis(x, y):
    return math.sqrt(sum((a-b)**2 for a,b in zip(x, y)))


def SpacialPointGenerate(center, sigma, samplesize):
    XcoordList = np.random.normal(center[0], sigma, samplesize)
    YcoordList = np.random.normal(center[1], sigma, samplesize)
    ZcoordList = np.random.normal(center[2], sigma, samplesize)
    points = [(x, y, z) for x, y, z in zip(XcoordList, YcoordList, ZcoordList)]
    return points


def NdimSpacialpointRnorm(paramList, sampleSize:int):
    """
        inputs is a list of params, each element is a list of param for normal distributions of a particular
        dimension.
        [[mean, sig],[mean, sig]...]

    :return:
        list of tuples as spacial points.
    """
    def NormalDisOneDim(mean, sig, n):
        return np.random.normal(mean, sig, n)
    L = [NormalDisOneDim(mu, sig, sampleSize) for mu, sig in paramList]
    Res = [tuple([L[J][I] for J in range(len(paramList))]) for I in range(sampleSize)]
    return Res


def main():
    points = SpacialPointGenerate((0,0,0), 100, 30)
    g = FullGraph([(0, 0), (1, 0), (2, 0)])
    print(g)
    g = FullGraph([(0, 0), (1, 0), (2, 0), (2, 2), (3, 0)])
    print(g)
    # so far it's correct

    g = FullGraph(points)
    print(g.MaxPartitionEvolve)

    points = SpacialPointGenerate((0, 0, 0), 100, 30) + SpacialPointGenerate((300, 300, 300), 100, 30)
    g = FullGraph(points)
    print(g.MaxPartitionEvolve)

    points = SpacialPointGenerate((0, 0, 0), 100, 30) + SpacialPointGenerate((200, 200, 200), 100, 30)
    g = FullGraph(points)
    print(g.MaxPartitionEvolve)

    points = SpacialPointGenerate((0, 0, 0), 100, 30) + SpacialPointGenerate((100, 100, 100), 100, 30)
    g = FullGraph(points)
    print(g.MaxPartitionEvolve)


if __name__ == "__main__":
    main()
