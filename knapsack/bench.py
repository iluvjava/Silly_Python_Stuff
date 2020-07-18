
from random import random as rnd
from time import time
from knapsack.core import *

def rand_problem_ints(range_:int, itemsCount:int, sparseness):
    assert 0 < sparseness < 1
    weights = [int(rnd() * range_) for I in range(itemsCount)]
    profits = [int(rnd() * range_) for I in range(itemsCount)]
    MaxWeights = int(sparseness * (sum(weights)))
    return profits, weights, MaxWeights

def run_solve_on(problemList, solver:callable):
    Time, Opt = [], []
    for P, W, B in problemList:
        Start = time()
        Value = solver(P,W,B)
        Time.append(time() - Start)
        print(f"Timer: {Time[-1]}")
        Opt.append(Value)
    return Time, Opt

from quick_csv import core

def bench_bb_with_dp(trials:int):

    def dp_solve(P, W, B):
        _, Opt = knapsack_dp_dual(P,W,B)
        return Opt

    def bb_solve(P, W, B):
        _, Opt = branch_and_bound(P, W, B)
        return Opt

    ItemCount = 400
    ItemProfitsWeightsRange = 1000
    KnapSackSparseness = 0.1
    ProblemList = [rand_problem_ints(ItemProfitsWeightsRange, ItemCount, KnapSackSparseness) for P in range(trials)]


    bb_time, bb_opt = run_solve_on(ProblemList, bb_solve)
    dp_time, dp_opt = run_solve_on(ProblemList, dp_solve)
    CSVHeader = ["bb_time", "dp_time", "bb_opt", "dp_opt"]
    CSVCols = [bb_time, dp_time, bb_opt, dp_opt]
    core.csv_col_save("bb, dp bench.csv", colHeader= CSVHeader, cols=CSVCols)

    print("Tests Detailed: ")
    print(f"The same set of problem is run on both BB and DP, time and optimal value is recored. ")
    print("The optimal value for both solver should be the same and the time cost is the interests. ")
    print(f"Item Count: {ItemProfitsWeightsRange}, Item's Profits and Weight range: (0, {ItemProfitsWeightsRange}), "
          f"Knapsack Sparseness: {KnapSackSparseness}")

def main():
    bench_bb_with_dp(10)
    pass

if __name__ == "__main__":
    main()