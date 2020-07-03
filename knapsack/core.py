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
        budget: int):
    """
            The dual formulation of the knapsack problem, minimizing the weights used for
            a certain profits.
    :param profits:
        All profits of items are non-negative, and must be an integers.
    :param weights:
        All weights of items are non negative, can be real numbers.
    :param budget:
        The maximum amount of budget allowed for the item's weight.
    :return:
        The set of indices representing the solution, and the optimal value of the solution.
    """
    assert len(profits) == len(weights), "weights and length must be in the same length. "
    assert min(profits) >= 0 and min(weights) >= 0, \
        "item profits and weight must be non-negative. "
    assert sum([1 for W in weights if W > budget]) == 0, \
        "All items must be weights less than maxWeight to reduce redundancies"

    TotalProfits, TotalWeights = sum(profits), sum(weights)
    T = [float("+inf")] * (TotalProfits + 1)
    T[0] = 0
    Soln = [[] for W in range(TotalWeights + 1)] # Store the indices of item that sum up to that exact profits
    for I in range(len(profits)):
        newT = [float("nan")]*len(T)
        for P in range(len(T)):
            NewWeight = T[P - profits[I]] + weights[I] if P - profits[I] >= 0 else float("inf")
            if NewWeight < T[P]:
                Soln[NewWeight] += Soln[NewWeight - weights[I]] + [I]
                newT[P] = NewWeight
            else:
                newT[P] = T[P]
        T = newT
    Res = Soln[[T[I] for I in range(len(T)) if T[I] <= budget][-1]]  # Index of the highest feasible profits.
    return Res, sum(profits[I] for I in Res)


def knapsack_dp_primal(
        profits: List[RealNumber],
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
            NewProfit = T[W - weights[I]] + profits[I] if W - weights[I] >= 0 else float("-inf")
            if NewProfit > T[W]:
                Soln[W] = Soln[W - weights[I]] + [I]
                newT[W] = NewProfit
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
        assert sum(1 for W in w if W > b) == 0, "All items must have weight less than the budget allowed, no redundancies. "
        assert min(w) >= 0 and min(p) >= 0, "All weights and profits must be positive. "

    def fractional_approx(self):
        """
            Allowing fractional item, estimate the upper bound for the problem.

            * The solution will have a loser bound than integral dual approx, however, I didn't prove if it's always the
            case, it depend on epsilon and the inputs.
        :return:
            The optimal value as the upper bound, and the optimal value of the solution, in the format of:
            Solution, TotalProfits.
        """
        P, W, B = self.__p, self.__w, self.__b  # Profits, weights, and budget.
        # tuples, first element is the value, second element is the index of the item.
        M = [(P[I]/W[I], I) for I in range(len(P))]
        M.sort(key=(lambda x: x[0]), reverse=True)
        Solution, TotalProfits, RemainingBudget = {}, 0, B # solution: index |-> (0, 1]
        for _, Index in M:
            ItemW, ItemP = W[Index], P[Index]
            if RemainingBudget - ItemW < 0:
                Solution[Index] = RemainingBudget/ItemW
                TotalProfits += ItemP; RemainingBudget = 0
                break
            RemainingBudget -= ItemW; TotalProfits += ItemP
            Solution[Index] = 1
        return Solution, TotalProfits

    def dual_approx(self):
        """
            * Gives an integral solution that is feasible, together with an estimated upperbound for the true optimal
            using this set of items.

            Scale the profits and make them into integers.
            * Solution is feasible
            * Optimal >= (1-epsilon)OPT; where OPT is the true optimal value with non-integer profits.
            * Polynomial run-time.
        :return:
            A integral solution that marks a lowerbound, and a number representing the upper bound.
        """
        P, W, B, eps = self.__p, self.__w, self.__b, self.__epsilon # Profits, weights, and budget.
        Scale = len(P)/(eps*max(P))
        ScaledProfits = [int(Profit*Scale) for Profit in P]
        Soln, Opt = knapsack_dp_dual(ScaledProfits, W, B)
        return Soln, Opt/(1 - eps)

    def primal_approx_upper(self):
        """
            Get a integral solution that may or may not be feasible, if it's infeasible, then it's an upper bound, else it's the
            optimal solution.

        :return:
            solution, optimal value.
        """
        return

    def __primal_approx(self, mode):
        """
            Internal use,
        :param mode:
            1: Over estimation; 2: Under estimation.
        :return:
            The integral solution of the approx.
        """
        return

    def primal_approx_lower(self):
        """
            Get an integral solution really fast, it's feasible.

            The approx solution can be arbitrarily bad for pathological inputs.
        :return:
            Solution, optimal value.
        """
        return

    @property
    def epsilon(self):
        return self.epsilon

    @epsilon.setter
    def epsilon(self, eps):
        assert 0 < eps < 1, "Epsilon out of range. "
        self.epsilon = eps

def main():
    print(knapsack_dp_dual([2, 3, 2, 1], [6, 7, 4, 1], 9))
    print(knapsack_dp_primal([2, 3, 2, 1], [6, 7, 4, 1], 9))

    def test_frac_approx():
        K = Knapsack([2, 3, 2, 1], [6, 7, 4, 1], 9)
        print(K.fractional_approx())
    test_frac_approx()




if __name__ == "__main__":
    main()