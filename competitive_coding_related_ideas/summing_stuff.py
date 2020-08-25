
from typing import *

def array_prefix_sum_immuttable(arr):
    """
        Immutable Array Prefix_sum
     >>> array_prefix_sum_mutable([1,1,1,1])
     [1, 2, 3, 4]

    :param m:
    :return:
    """
    M = len(arr)
    Res = [0 for _ in range(M)]
    Res[0] = arr[0]
    for I in range(1, M):
        Res[I] = Res[I - 1] + arr[I]
    return Res


def array_prefix_sum_mutable(arr):
    """
    >>> array_prefix_sum_mutable([1,1,1,1])
    [1, 2, 3, 4]

    :param arr:
    :return:
    """
    for I in range(1, len(arr)):
        arr[I] = arr[I] + arr[I - 1]
    return arr


def array_prefix_sum_pythonista_style_immutable(arr):
    """
    >>> array_prefix_sum_pythonista_style_immutable([1 for _ in range(4)])
    [1, 2, 3, 4]

    :param arr:
    :return:
    """
    from itertools import accumulate
    return list(accumulate(arr, lambda a, b: a + b))


def matrix_prefix_sum_immutable(m):
    M, N = len(m), len(m[0])
    SumMatrix = [[0 for _ in range(N)] for _ in range(M)]
    for I in range(N):
        SumMatrix[I][0] = m[I][0]
    for I in range(M):
        for J in range(1, N):
            SumMatrix[I][J] = m[I][J] + SumMatrix[I][J - 1]
    for J in range(N):
        for I in range(1, M):
            SumMatrix[I][J] = SumMatrix[I][J] + SumMatrix[I - 1][J]
    return SumMatrix


def matrix_prefix_sum_muttable(m):
    M, N = len(m), len(m[0])
    for I in range(1, M):
        m[I][0] = m[I - 1][0] + m[I][0]
    for J in range(1, N):
        m[0][J] = m[0][J - 1] + m[0][J]
    for I in range(1, M):
        for J in range(1, N):
            m[I][J] = m[I][J - 1] + m[I - 1][J] -m[I - 1][J - 1]
    return m


def main():
    print(matrix_prefix_sum_muttable([[1 for _ in range(3)] for __ in range(3)]))
    pass

if __name__ == "__main__":
    main()