
__all__ = ["SimpleDiGraph"]
from typing import *

class SimpleDiGraph:

    def __init__(self):
        self._V = {}  # Integer Index |-> Generic Vertex
        self._VInvert = {}  # Generic |-> Integer Index
        self._AdjLst = {}  # Integer Index of vertices |-> Other Integer Index of vertices
        self._E = {}  # Integer index Tuple |-> Generic Edge association.

    def __iadd__(self, vertex):
        """

        :param vertex:
            A generic Vertex.
        :return:
            The instance itself.
        """
        n = self.size()
        if vertex in self._V.values():
            return self
        self._V[n] = vertex
        self._VInvert[vertex] = n
        self._AdjLst[n] = set()
        return self

    def size(self):
        """
        :return:
            Number of vertex in the graph.
        """
        return len(self._V.keys())

    def get_edge(self, item):
        """
            Get generic edge value by index tuple.
        :param item:
            An index tuple.
        :return:
            The generic value of the edge.
        """
        return self._E[item]

    def __getitem__(self, item):
        """
            Transform index to vertex and vice versa,
            always query the keys for index first.
        :param item:
            Vertex of an integer index of a vertex.
        :return:
            the neighbours of
        """
        if item in self._V.keys():
            return self._V[item]
        if item in self._VInvert.keys():
            return self._VInvert[item]
        if item is Type[Tuple[int]]:
            return self._E[item]
        raise Exception("Key Error. ")

    def connect(self, v1, v2, edge=None):
        """
            Add a neighbour for a vertex, new vertex could be completely new and that is ok.
        :param v1:
            A vertex
        :param v2:
            A neighbour.
        :return:
        """
        assert v1 in self._V.values() and v2 in self._V.values()
        self.connect_by_idx(self._VInvert[v1], self._VInvert[v2], edge)

    def connect_by_idx(self, v1, v2, edge=None):
        assert v1 in self._V.keys() and v2 in self._V.keys()
        self._AdjLst[v1].add(v2)
        self._E[v1, v2] = edge

    def __repr__(self):
        res = "Graph \n"
        for I in range(self.size()):
            res += f"{I}: {self._AdjLst[I]} \n"
        res += "E:\n"
        for EdgeTuple, Edge in self._E.items():
            res += f"{EdgeTuple}: {Edge}\n"
        res += "V:\n"
        for VertexIdx, Vertex in self._V.items():
            res += f"{VertexIdx}: {Vertex}\n"
        return res

