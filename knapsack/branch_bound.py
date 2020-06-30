"""
    We will use an approximation algorithm and the branch-and-bound algorithm
    to solve the napsack problem.

    Dual simplex + Gomory cut doesn't seem efficient at all, for this problem.

    So we will use Dynamic Programming and an Approximation algorithm.
"""

from typing import *


def knapsack_dp(profits:List[int], weights: List[int], maxWeight:int) -> List[int]:
    assert len(profits) == len(weights), "weights and length must be in the same length. "
    assert min(profits) >= 0 and min(weights) >= 0,\
        "item profits and weight must be non-negative. "
    assert sum([1 for W in weights if W > maxWeight]) == 0,\
        "All items must be weights less than maxWeight to reduce redundancies"
    T = [0]*maxWeight
    T[weights[0]] = profits[0]
    Soln = [[0]*maxWeight for W in range(maxWeight)] # Store the characteristic vectors representing each solution.
    Soln[weights[0]] = [1]
    for I in range(1, len(profits)):
        newT = [float("nan")]*len(T)
        for W in range(maxWeight):
            NewProfits = T[W - weights[I]] +  profits[I] if  W - weights[I] >= 0 else float("-inf")
            if NewProfits > T[W]:
                Soln[W].append(1)
                newT[W] = NewProfits
            else:
                Soln[W].append(0)
                newT[W] = T[W]
        T = newT
    return Soln[T.index(max(T))]


def main():
    soln = knapsack_dp(profits=[2, 3, 2, 1], weights=[6, 7, 4, 1], maxWeight=9)
    print(soln)
    pass


if __name__ == "__main__":
    main()