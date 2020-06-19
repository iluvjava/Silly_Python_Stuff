"""
    Investigate if granularization has impact on the run-time of the
    TSP LP.
"""
from TSP_LP_Reduction_MTZ.core import TravelingSalesManLP, rand_points, unit_circle
from statistics import stdev
from time import time

def main():
    X1, X2 = [], []
    N = 50

    for I in range(N):
        SolverInstance = TravelingSalesManLP()
        for P in unit_circle(r=3):
            SolverInstance += P
        for P in unit_circle(n=5):
            SolverInstance += P

        SolverInstance.granular_off()
        TStart = time()
        SolverInstance.solve_path()
        X1.append(time() - TStart)
        SolverInstance.plot_path()

        SolverInstance.granular_on()
        TStart = time()
        SolverInstance.solve_path()
        X2.append(time() - TStart)
        SolverInstance.plot_path()

    X3 = [x1 - x2 for x1, x2 in zip(X1, X2)]
    print(f"Avg of X1 - X2: {sum(X3)/N}")
    print(f"stdv of X1 - X2: {stdev(X3)}")
    # print(f"Avg on pair-wise objective Diff: {sum(ObjectiveDiff)/N}")

    print("Differences of runtime of (Non-Granular - Granular)")
    for I in X3:
        print(I)
    #print("Differences of objective val of (Non-Granular - Granular)")
    #for I in ObjectiveDiff:
    #    print(I)




if __name__ == "__main__":
    main()
