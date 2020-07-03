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
from knapsack.scratch_paper_works import *
import statistics as stat
from time import time
from knapsack.core import *
import csv

def main():

    def rand_problem_dense(N:int):
        weights = [int(rnd()*N) for I in range(N)]
        profits = [rnd() for I in range(N)]
        MaxWeights = int(0.5*(sum(weights)))
        return profits, weights, MaxWeights

    def rand_problem_sparse(N:int):
        weights = [int(rnd()*N) + N for I in range(N)]
        profits = [rnd() for I in range(N)]
        MaxWeights = int(2*stat.median(weights))
        return profits, weights, MaxWeights

    def bench_mark(N: int, n):
        optimals = []
        Times = []
        for I in range(N):
            weights, profits, MaxWeights = rand_problem_dense(n)
            Tm = time()
            _, OptExact = knapsack_dp(profits, weights, MaxWeights)
            Tp1 = time() - Tm
            Tm = time()
            _, OptApprox = knapsack_approx(profits, weights, MaxWeights) # Lower bound approx.
            Tp2 = time() - Tm
            optimals.append((OptExact, OptApprox))
            Times.append((Tp1, Tp2))
        return optimals, Times

    def bench_mark_by(problems, knapsackrunner:callable):
        """

        :param problems:
            A list of problems, each problem is tuple of 3 numbers.
            [(P1, W1, B1), (P2, W2, B2),...]
        :param knapsackrunner:
            A testing function to run on the list of inputs.
        :return:
            A tuple, both are list. List of optimal values and time for each
            instance in the list of problem.
        """
        Optimals, Times = [], []
        for P, W, B in problems:
            Tm = time()
            _, Opt = knapsackrunner(P, W, B)
            Tp1 = time() - Tm
            Times.append(Tp1)
            Optimals.append(Opt)
        return Optimals, Times

    def test1():
        # printing out the reports.
        trials, n = 20, 200
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

    def test_knapsack_class(problemFxn:callable, fileName:str):

        def approx_fast(p, w, m):
            Instance = Knapsack(p, w, m)
            return Instance.approx_fastest()

        def approx_dual(p, w, m):
            Instance = Knapsack(p, w, m)
            return Instance.dual_approx()

        def exact(p, w, m):
            return knapsack_dp_primal(p, w, m)

        N, n = 10, 60

        ProblemsList = []
        for I in range(N):
            ProblemsList.append(problemFxn(n))
        Optimal1, Times1 = bench_mark_by(ProblemsList, approx_fast)
        Optimal2, Times2 = bench_mark_by(ProblemsList, approx_dual)
        Optimal3, Times3 = bench_mark_by(ProblemsList, exact)

        CsvData = [None]*N
        for I in range(N):
            Row = {}
            Row["approx_fast"], Row["approx_dual"], Row["exact"] = Optimal1[I], Optimal2[I], Optimal3[I]
            Row["appox_fast_time"], Row["approx_dual_time"], Row["exact_time"] = Times1[I], Times2[I], Times3[I]
            CsvData[I] = Row

        with open(fileName, mode="w") as CsvDataFile:
            writer = csv.DictWriter(CsvDataFile, fieldnames=list(CsvData[0].keys()))
            writer.writeheader()
            writer.writerows(CsvData)

    test_knapsack_class(rand_problem_dense, fileName="test_data_dense_knapsac.csv")
    test_knapsack_class(rand_problem_sparse, fileName="test_data_sparse_knapsac.csv")


if __name__ == "__main__":
    main()