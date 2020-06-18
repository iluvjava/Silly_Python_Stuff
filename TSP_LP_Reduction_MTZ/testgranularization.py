"""
    Investigate if granularization has impact on the run-time of the
    TSP LP.
"""
from TSP_LP_Reduction_MTZ.core import TravelingSalesManLP, rand_points

from time import time

def main():
    X1, X2 = [], []
    ObjectiveDiff = []
    N = 30

    for I in range(N):
        SolverInstance = TravelingSalesManLP()
        for P in rand_points([0, 10], [10, 0], 10):
            SolverInstance += P

        SolverInstance.granular_off()
        TStart = time()
        SolverInstance.solve_path()
        X1.append(time() - TStart)
        ObjectiveVal = SolverInstance.P.objective
        SolverInstance.plot_path()

        SolverInstance.granular_on()
        TStart = time()
        SolverInstance.solve_path()
        X2.append(time() - TStart)
        ObjectiveDiff.append(SolverInstance.P.objective - ObjectiveVal)
        SolverInstance.plot_path()


    print(f"X1: {X1}")
    print(f"X2: {X2}")
    print(f"Avg of X1 - X2: {sum([x1 - x2 for x1, x2 in zip(X1, X2)])/N}")
    print(f"Avg on pair-wise objective Diff: {sum(ObjectiveDiff)/N}")
    pass


if __name__ == "__main__":
    main()
