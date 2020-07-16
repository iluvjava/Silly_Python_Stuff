

from knapsack.core import *
import random as rnd
import statistics as stat

def rand_problem_dense(N: int):
    weights = [int(rnd() * N) for I in range(N)]
    profits = [rnd() for I in range(N)]
    MaxWeights = int(0.5 * (sum(weights)))
    return profits, weights, MaxWeights


def rand_problem_sparse(N: int):
    weights = [int(rnd() * N) + N for I in range(N)]
    profits = [rnd() for I in range(N)]
    MaxWeights = int(2 * stat.median(weights))
    return profits, weights, MaxWeights

def main():
    def bb_vs_dp(N:int):

        pass

    pass

if __name__ == "__main__":
    main()