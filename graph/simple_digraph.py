
__all__ = ["SimpleDiGraph"]
from typing import *


class SimpleDiGraph:
    """
        A Generic simple digraph

        you can only add edges and vertices to this graph.
    """
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

            !! always query the keys for index first.

            1. Given a tuple, it will return the meta information for the edge.
            2. Given a vertex, or the integers representing the vertex, it will return the
            vertex for the integer, or integer  for the vertex.

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
            Connect a directed edge going from v1 to v2.
            v1 and v2 must already be added to the graph!
        :param v1:
            A vertex
        :param v2:
            A neighbour.
        :return:
        """
        assert v1 in self._V.values() and v2 in self._V.values()
        self.connect_by_idx(self._VInvert[v1], self._VInvert[v2], edge)

    def connect_by_idx(self, v1, v2, edge=None):
        """
            Connect a directed edge going frog v1 to v2.
            v1, v2 must be already presented in the graph.
        :param v1:
            A integer representation of the vertex.
        :param v2:
            A integer representation of the vertex.
        :param edge:
            The meta information you want to associate the edge with.
        :return:
            the graph itself.
        """
        assert v1 in self._V.keys() and v2 in self._V.keys()
        if v2 not in self._AdjLst[v1]:
            self._AdjLst[v1].add(v2)
            self._E[v1, v2] = edge

    def adj_vertices(self, Vidx):
        """
            Return the neighbouring vertices
        :param Vidx:
            The integer index representing the vertex.
        :return:
            A list of integers representing its neighbours.
        """
        return self._AdjLst[Vidx].copy()

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

