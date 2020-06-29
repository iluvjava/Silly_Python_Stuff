"""
    We will use an approximation algorithm and the branch-and-bound algorithm
    to solve the napsack problem.

    Dual simplex + Gomory cut doesn't seem efficient at all, for this problem.

    So we will use Dynamic Programming and an Approximation algorithm.
"""

from typing import *


def napsack_dp(profits:List[int], weights: List[int], maxWeight:int) -> List[int]:
    assert min(profits) >= 0 and min(maxWeight) >= 0
    T = [(w if w == weights[0] else float["-inf"]) for w in range(sum(maxWeight))]

    pass
