"""
    We will use an approximation algorithm and the branch-and-bound algorithm
    to solve the napsack problem.

    Dual simplex + Gomory cut doesn't seem efficient at all, for this problem.

    So we will use Dynamic Programming and an Approximation algorithm.
"""
__all__ = ["knapsack_dp", "knapsack_approx"]
from typing import *
import math


def knapsack_dp(
        profits: List[Union[float, int]],
        weights: List[int],
        Budget: int):
    assert len(profits) == len(weights), "weights and length must be in the same length. "
    assert min(profits) >= 0 and min(weights) >= 0,\
        "item profits and weight must be non-negative. "
    assert sum([1 for W in weights if W > Budget]) == 0,\
        "All items must be weights less than maxWeight to reduce redundancies"
    T = [0] * (Budget + 1)
    Soln = [[] for W in range(Budget + 1)] # Store the indices of item that sum up to that exact weight.
    for I in range(len(profits)):
        newT = [float("nan")]*len(T)
        for W in range(Budget + 1):
            IncludeItem = T[W - weights[I]] + profits[I] if W - weights[I] >= 0 else float("-inf")
            if IncludeItem > T[W]:
                Soln[W] = Soln[W - weights[I]] + [I]
                newT[W] = IncludeItem
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
        epsilon=0.5,
        **kwargs):
    """
    :param profits:
    :param weights:
    :param maxWeight:
    :param epsilon:
        Sensitivity, run-time is inversely proportional to this value. The smaller the value, the better the
        approximation.
    :param kwargs:
        upperBound: set the value to any thing so the algorithm will produce a solution that is
        an upper bound, by default it will give a solution that is an upper bound.
        * SOLUTION IS GOING TO BE INFEASIBLE MOST OF THE TIME.
    :return:
    """
    assert len(profits) == len(weights), "weights and length must be in the same length. "
    assert min(profits) >= 0 and min(weights) >= 0, \
        "item profits and weight must be non-negative. "
    assert sum([1 for W in weights if W > maxWeight]) == 0, \
        "All items must be weights less than maxWeight to reduce redundancies"
    WeightMax = max(weights)
    Multiplier = math.log2(WeightMax)/(WeightMax*epsilon) # This scales all weights
    MaxWeightModified = int(Multiplier*maxWeight)
    ScaledWeights = [(int(W*Multiplier) + 1 if "upperBound" not in kwargs.keys() else int(W*Multiplier))\
               for W in weights]
    Soln, OptProfits = knapsack_dp(profits, ScaledWeights, MaxWeightModified)
    return Soln, OptProfits


def main():
    soln, OptimalVal = knapsack_dp(profits=[2, 3, 2, 1], weights=[6, 7, 4, 1], Budget=9)
    print(soln)
    soln, OptimalVal = knapsack_approx(profits=[2, 3, 2, 1], weights=[6.01, 7.001, 4.02, 1.006], maxWeight=9)
    print(soln)
    p = [0.6030021407961512,
         0.10068952967138334,
         0.4939995363233398,
         0.5262149986510344,
         0.5236617089819712,
         0.7709652095560733,
         0.997521961746892,
         0.4301888262004909,
         0.2992361367251165,
         0.23567188609798284]
    w = [21, 21, 16, 21, 6, 16, 1, 21, 16, 6]
    b = 90
    soln, OptimalVal = knapsack_dp(p, w, b)

    pass


if __name__ == "__main__":
    main()