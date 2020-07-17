

from knapsack.core import *
from random import random as rnd
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


    def bb_vs_dp(N:int, n:int):
        ProblemList = [rand_problem_dense(n) for I in range(N)]
        for P, W, B in ProblemList:
            S1, Opt1 = branch_and_bound(P, W, B)
            S2, Opt2 = knapsack_dp_primal(P, W, B)
            S1.sort(), S2.sort()
            print(S1, S2)
            assert str(S1) == str(S2), f"{S1}={sum(P[I] for I in S1)}, {S2}={sum(P[I] for I in S2)}" + \
                f"Details: P = {P}\n  W={W}\n B={B}"


    def verify_upperbound():
        ProblemList = [rand_problem_sparse(5) for I in range(100)]
        for P, W, B in ProblemList:
            ks = Knapsack(P, W, B)
            _, Opt = knapsack_dp_primal(P, W, B)
            Upper = ks.tight_upper_bound()
            assert Upper >= Opt, f"{Upper} < {Opt}\n P={P}, W={W}, B={B}"

    bb_vs_dp(1000, 4)
    bb_vs_dp(1000, 20)
    verify_upperbound()
    pass

if __name__ == "__main__":
    main()