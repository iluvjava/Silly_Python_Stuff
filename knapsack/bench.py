from random import random as rnd
from numpy import random as np_rnd
from time import time
from knapsack.core import *
import sys; import os
from quick_csv import core
from quick_json import quick_json


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


def rand_problem_exponential(scale: int, itemsCount: int, satruration):
    assert 0 < satruration < 1
    Profits, Weights = np_rnd.exponential(scale=scale, size=(2, itemsCount))
    Profits, Weights = map(lambda x: int(x), Profits), map(lambda x:int(x), Weights)
    Profits, Weights = list(Profits), list(Weights)
    Budget = int(satruration*sum(W for W in Weights))
    return Profits, Weights, Budget


def bench_bb_with_dp(trials:int):

    def dp_solve(P, W, B):
        _, Opt = knapsack_dp_dual(P,W,B)
        return Opt

    def bb_solve(P, W, B):
        _, Opt = branch_and_bound(P, W, B)
        return Opt

    ItemCount = 20
    ItemProfitsWeightsRange = 1000
    KnapSackSparseness = 0.1
    ProblemList = [rand_problem_ints(ItemProfitsWeightsRange, ItemCount, KnapSackSparseness) for P in range(trials)]
    ProblemList += [rand_problem_exponential(ItemProfitsWeightsRange, ItemCount, KnapSackSparseness) for P in range(trials)]
    bb_time, bb_opt = run_solve_on(ProblemList, bb_solve)
    dp_time, dp_opt = run_solve_on(ProblemList, dp_solve)
    CSVHeader = ["bb_time", "dp_time", "bb_opt", "dp_opt"]
    CSVCols = [bb_time, dp_time, bb_opt, dp_opt]
    core.csv_col_save("bb, dp bench.csv", colHeader= CSVHeader, cols=CSVCols)
    quick_json.json_encode([CSVHeader, CSVCols], filename="bb, dp bench.json")


    print("Tests Detailed: ")
    print(f"The same set of problem is run on both BB and DP, time and optimal value is recored. ")
    print("The optimal value for both solver should be the same and the time cost is the interests. ")
    print(f"Item Count: {ItemProfitsWeightsRange}, Item's Profits and Weight range: (0, {ItemProfitsWeightsRange}), "
          f"Knapsack Sparseness: {KnapSackSparseness}")


def main():
    bench_bb_with_dp(10)
    pass


def test():
    print(rand_problem_exponential(10000, 5, 0.5))


if __name__ == "__main__":
    print(f"swd: {os.getcwd()}")
    print(sys.argv)
    if len(sys.argv) != 1:
        test()
    else:
        main()
