"""
    We are going to test some of the algorithm in python.
    * Subject of investigation:
        * Comparing the approximation algorithm with the exact dynamic programming solution.
            * Optimality
            * run-time.
    * Generation of the problem:
        * item's profits are random in range 0 < p < 1000 000
        * item's weight 0 < w < W/2; where W = n
"""

from random import random as rnd
from knapsack.branch_bound import *
import statistics as stat
from time import time


def main():

    def rand_problem(N:int):
        weights = [int(rnd()*(N/2)) for I in range(N)]
        profits = [rnd() for I in range(N)]
        MaxWeights = int(0.7*(sum(weights)))
        return weights, profits, MaxWeights

    def bench_mark(N: int, n):
        optimals = []
        Times = []
        for I in range(N):
            weights, profits, MaxWeights = rand_problem(n)
            Tm = time()
            _, OptExact = knapsack_dp(profits, weights, MaxWeights)
            Tp1 = time() - Tm
            Tm = time()
            _, OptApprox = knapsack_approx(profits, weights, MaxWeights) # Lower bound approx.
            Tp2 = time() - Tm
            optimals.append((OptExact, OptApprox))
            Times.append((Tp1, Tp2))
        return optimals, Times

    # printing out the reports.
    trials, n = 20, 100
    Results, Times = bench_mark(trials, n)
    OptimalExact, OptimalApprox = [I[0] for I in Results], [I[1] for I in Results]
    ExactTime, ApproxTime = [I[0] for I in Times], [I[1] for I in Times]
    print("Stats for Optimal Exact: ")
    print(f"mean: {stat.mean(OptimalExact)}")
    print(f"stdev: {stat.stdev(OptimalExact)}")
    print()
    print("Stats for Time Exact: ")
    print(f"mean: {stat.mean(ExactTime)}")
    print(f"stdev: {stat.stdev(ExactTime)}")
    print()
    print("Stats for Optimal Approx: ")
    print(f"mean: {stat.mean(OptimalApprox)}")
    print(f"stdev: {stat.stdev(OptimalApprox)}")
    print()
    print("Stats for Approx time: ")
    print(f"mean: {stat.mean(ApproxTime)}")
    print(f"stdev: {stat.stdev(ApproxTime)}")
    print()
    print(f"Number of trials is: {trials}")


if __name__ == "__main__":
    main()