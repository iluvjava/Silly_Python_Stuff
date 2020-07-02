"""
    Pre pare a class that encapsulate all the things need for
    the branch and bound algorithm.
    Together with some good helper methods to aid things a bit.

"""

from typing import *
RealNumber = Union[float, int]


def knapsack_dp_dual(
        profits: List[int],
        weights: List[RealNumber],
        Budget: int):
    """
            The dual formulation of the knapsack problem, minimizing the weights used for
            a certain profits.
    :param profits:
        All profits of items are non-negative, and must be an integers.
    :param weights:
        All weights of items are non negative, can be real numbers.
    :param Budget:
        The maximum amount of budget allowed for the item's weight.
    :return:
        The set of indices representing the solution.
    """
    assert len(profits) == len(weights), "weights and length must be in the same length. "
    assert min(profits) >= 0 and min(weights) >= 0, \
        "item profits and weight must be non-negative. "
    assert sum([1 for W in weights if W > Budget]) == 0, \
        "All items must be weights less than maxWeight to reduce redundancies"

    TotalProfits = sum(profits)
    T = [float("+inf")] * (TotalProfits + 1)
    T[0] = 0
    Soln = [[] for W in range(TotalProfits + 1)] # Store the indices of item that sum up to that exact profits
    for I in range(len(profits)):
        newT = [float("nan")]*len(T)
        for P in range(TotalProfits):
            addW = T[P - profits[I]] + weights[I] if P - profits[I] >= 0 else float("inf")
            if addW < T[P]:
                Soln[P].append(I)
                newT[P] = addW
            else:
                newT[P] = T[P]
        T = newT
    Res = [P for P in range(len(T)) if T[P] <= Budget][-1] # Index of the highest feasible profits.
    return Soln[Res]


def knapsack_dp_primal(
        profits: List[Union[float, int]],
        weights: List[int],
        Budget: int):
    """
        A primal formulation of the knapsack problem.
    :param profits:
        Non-negative real numbers.
    :param weights:
        Non-negative Integers.
    :param Budget:
        Non-negative real number, larger than all item's weight.
    :return:
        The optimal solution (list of indices of the items. )
    """
    assert len(profits) == len(weights), "weights and length must be in the same length. "
    assert min(profits) >= 0 and min(weights) >= 0,\
        "item profits and weight must be non-negative. "
    assert sum([1 for W in weights if W > Budget]) == 0,\
        "All items must be weights less than maxWeight to reduce redundancies"
    T = [0] * (Budget + 1)
    Soln = [[] for W in range(Budget + 1)] # Store the indices of item that sum up to that exact weight.
    for I in range(len(profits)):
        newT = [float("nan")]*len(T)
        for W in range(Budget + 1):
            IncludeItem = T[W - weights[I]] + profits[I] if W - weights[I] >= 0 else float("-inf")
            if IncludeItem > T[W]:
                Soln[W] = Soln[W - weights[I]] + [I]
                newT[W] = IncludeItem
            else:
                newT[W] = T[W]
        T = newT
    P_star = max(T)
    return Soln[T.index(P_star)]

class Knapsack:
    """
        This class is designed to serve for branch and bound.

        Defines the sub problems of knapsack:
        * Heuristic Upper Bound.
        * Feasible lower bound approximation.
        * Exact dynamic programming for integral profits on items.
    """

    def __init__(self, itemProfits: List[RealNumber], itemWeight: List[RealNumber], budget: RealNumber):
        self.__preconditions(itemProfits, itemWeight, budget)
        self.__epsilon = 0.1
        self.__p, self.__w, self.__b = itemProfits.copy(), itemWeight.copy(), budget

    def __preconditions(self, p, w, b):
        assert len(p) == len(w), "Length of list of weight must equal to the length of list of profits. "
        assert sum(1 for W in w if W > b), "All items must have weight less than the budget allowed, no redundancies. "
        assert min(w) >= 0 and min(p) >= 0, "All weights and profits must be positive. "

    def fractional_approx(self):
        """
            Allowing fractional item, estimate the upper bound for the problem.
        :return:
            The optimal value as the upper bound.
        """

        pass

    def integral_profits_approx(self):
        """
            Scale the profits and make them into integers.
            * Optimal >= (1-epsilon)OPT; where OPT is the true optimal value with non-integer profits.
        :return:
        """
        pass

    @property
    def epsilon(self):
        return self.epsilon

    @epsilon.setter
    def epsilon(self, eps):
        assert 0 < eps < 1, "Epislon out of range. "
        self.epsilon = eps

def main():
    print(knapsack_dp_dual([2, 3, 2, 1], [6, 7, 4, 1], 9))
    pass


if __name__ == "__main__":
    main()