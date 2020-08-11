"""
    In this file we are going to visualize some of the inputs and outputs of the extended knapsack problem,
    so poeple and appreciated the algorithm in a visual way.


    * Items with different profits and weights are circles on the 2d planes.
    * The area of the circle is proportional to the number of available items in the inputs of the knapsack problem.

    * The solution it's filling in the circles with black color, so it's concentric circles I believe,
        ** Plot label a number besides it will be very helpful.

    * I don't know how to visulize the budget tho.
        ** Just plot a line, representing the ratio of the available budget and the total weights of all items.
"""
from matplotlib import pyplot as plt
from knapsack import extended_napsack_core3 as eks
import math as math


def visualize_inputs(profits, weights, counts, budget, maxMarkerSize = 150):
    # normalize supposed sizes of markers ------------------------------------------------------------------------------
    MaxCount = max(C for C in counts)
    MarkerSizes = [((C/MaxCount)**2)*maxMarkerSize for C in counts]
    # plot it ----------------------------------------------------------------------------------------------------------
    FigHandle, ax = plt.subplots(dpi=400)
    for P, W, S in zip(profits, weights, MarkerSizes):
        ax.scatter([P], [W], s=S, facecolors="none", edgecolor="b")
    ax.set_xlabel("Profits")
    ax.set_ylabel("Weights")
    ax.set_title("Extended Knapsack, Markersize: Number of availability of that item ")
    # Try include info about the budgets -------------------------------------------------------------------------------
    Density = budget/math.fsum(C*W for C, W in zip(counts, weights))
    ax.axhline(y=Density, color="r", linestyle='-', label=f"Budget~={round(budget, 4)}")
    legend = ax.legend(loc='best', shadow=True, fontsize='small')
    return FigHandle, ax


def visualize_inputs_with_soln(profits, weights, counts, budget, soln, maxMarkerSize=150):
    """
        Plot the inputs together with the solution. 
    :param profits:
        list of item's profits.
    :param weights:
        List of item's weights
    :param counts:
        List of integers.
    :param soln:
        Map, representing the solution to the problem.
    :param maxMarkerSize:
        the maximum marker size for plotting the graph.
    :return: 
    """
    FigHandle, ax = visualize_inputs(profits, weights, counts, budget, maxMarkerSize)
    # Markersize Multiplier --------------------------------------------------------------------------------------------
    Multiplier = maxMarkerSize/max(counts)
    for I, V in soln.items():
        ax.scatter(profits[I], weights[I], s=((V/counts[I])**2)*(counts[I]*Multiplier), edgecolor="none")
        ax.annotate(f"|{V}/{counts[I]}|", (profits[I], weights[I]), color="orange", size=4)
    return FigHandle, ax


def main():

    def GenerateRandomProblemAndPlot():
        N = 3
        for I in range(N):
            PWC, B = eks.make_extended_knapsack_problem(size=60, density=1*((I+1)/(N+1)), itemsCounts=30, significance=9)
            P, W, C = list(map(list, zip(*PWC)))

            Solve = eks.EknapsackGreedy(P, W, C, B)
            Solved, Obj = Solve.solve()

            Fighanle, ax = visualize_inputs_with_soln(P, W, C, B, Solved)
            Fighanle.show()

    def Test():
        PWC, B = eks.make_extended_knapsack_problem(size=8, density=0.1, itemsCounts=100, significance=9)
        P, W, C = list(map(list, zip(*PWC)))
        Solve = eks.EknapsackGreedy(P, W, C, B)
        Solved, Obj = Solve.solve()
        print("P, W, C, B: ")
        print(P, W, C, B)
        print(Solved)
        Fighanle, ax = visualize_inputs_with_soln(P, W, C, B, Solved)
        Fighanle.show()


    GenerateRandomProblemAndPlot()
    # Test()




if __name__ == "__main__": 
    main()