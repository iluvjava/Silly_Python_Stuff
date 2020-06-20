"""
    In this file, we are interesting in answering the following problem:

    Given a set of integers,
        for all subset that, how many unique values their sum can take?
        For each unique sum, how many subset can sum up to that number?

    Read README.MD file for more.

    Given a set of integers variables each with a range:
        For all subsets of variable where each of them take some kind of value
        in that range, how many unique sum it can take?

"""

from typing import *
import matplotlib.pyplot as pyplt
import numpy as np
import random as rnd

def main():

    for I in range(10, 100, 10):
        [L, U], R, T = unique_subset_sum([rnd.randint(-100, 100) for I in range(I)])
        print(T)
        print(f"L = {L}")
        print(f"U = {U}")
        print(f"R = {R}")
        plot_it(T)


def unique_subset_sum(S: List[int]):
    """
        Input is a list of integers, empty set is allowed.

        Implementation below is not memory friendly.
    :param S:
    :return:
    [L, U], {...}
    """
    L, U = sum(I for I in S if I < 0), sum(I for I in S if I > 0)
    T = {}
    for K in range(L, U + 1):
        T[0, K] = 1 if S[0] == K else 0
    for J in range(len(S)):
        T[J, 0] = 1 # There exists the empty set that sum up to zero!!!
        # Not summing anything is summing up to zero.
    for J, K in [(J, K) for K in range(L, U + 1) for J in range(1, len(S))]:
        T[J, K] = T[J - 1, K - S[J]] + T[J - 1, K] if ((J, K - S[J]) in T) else T[J - 1, K]
    Res = [K for K in range(L, U + 1) if T[len(S) - 1, K] != 0]
    T = dict([(K, T[len(S) - 1, K]) for K in range(L, U + 1) if T[len(S) - 1, K] != 0])
    T[0] = T[0] - 1
    return [L, U], Res, T

def plot_it(T):
    pyplt.clf()
    X, Y = list(T.keys()), list(T.values())
    pyplt.yscale("log")
    pyplt.plot(X, Y, 'o')
    pyplt.show()


if __name__ == "__main__":
    main()