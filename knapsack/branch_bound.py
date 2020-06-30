"""
    We will use an approximation algorithm and the branch-and-bound algorithm
    to solve the napsack problem.

    Dual simplex + Gomory cut doesn't seem efficient at all, for this problem.

    So we will use Dynamic Programming and an Approximation algorithm.
"""
__all__ = ["knapsack_dp", "knapsack_approx"]
from typing import *


def knapsack_dp(profits:List[Union[float, int]], weights: List[int], maxWeight:int):
    assert len(profits) == len(weights), "weights and length must be in the same length. "
    assert min(profits) >= 0 and min(weights) >= 0,\
        "item profits and weight must be non-negative. "
    assert sum([1 for W in weights if W > maxWeight]) == 0,\
        "All items must be weights less than maxWeight to reduce redundancies"
    T = [0]*maxWeight
    T[weights[0]] = profits[0]
    Soln = [[] for W in range(maxWeight)] # Store the indices of item that sum up to that exact weight.
    Soln[weights[0]] = [1]
    for I in range(1, len(profits)):
        newT = [float("nan")]*len(T)
        for W in range(maxWeight):
            NewProfits = T[W - weights[I]] + profits[I] if  W - weights[I] >= 0 else float("-inf")
            if NewProfits > T[W]:
                Soln[W].append(I)
                newT[W] = NewProfits
            else:
                newT[W] = T[W]
        T = newT
    P_star = max(T)
    return Soln[T.index(P_star)], P_star

Reals = Union[float, int]
RealsLst = List[Reals]


def knapsack_approx(
        profits: RealsLst,
        weights: RealsLst,
        maxWeight: Reals,
        epsilon=0.1,
        **kwargs):
    """
    :param profits:
    :param weights:
    :param maxWeight:
    :param epsilon:
        Sensitivity, run-time is inversely proportional to this value. The smaller the value, the better the
        approximation.
    :param kwargs:
        roundUp: set the value to any thing so the algorithm will produce a solution that is
        an upper bound, by default it will give a solution that only a lower bound.
    :return:
    """
    assert len(profits) == len(weights), "weights and length must be in the same length. "
    assert min(profits) >= 0 and min(weights) >= 0, \
        "item profits and weight must be non-negative. "
    assert sum([1 for W in weights if W > maxWeight]) == 0, \
        "All items must be weights less than maxWeight to reduce redundancies"
    Multiplier = len(profits)/(epsilon*max(weights))
    maxWeight = int(Multiplier*maxWeight)
    weights = [(int(W*Multiplier + 1) if "roundUp" not in kwargs.keys() else int(W*Multiplier))\
               for W in weights]
    return knapsack_dp(profits, weights, maxWeight)


def main():
    soln, OptimalVal = knapsack_dp(profits=[2, 3, 2, 1], weights=[6, 7, 4, 1], maxWeight=9)
    print(soln)
    soln, OptimalVal = knapsack_approx(profits=[2, 3, 2, 1], weights=[6.01, 7.001, 4.02, 1.006], maxWeight=9)
    print(soln)
    pass


if __name__ == "__main__":
    main()