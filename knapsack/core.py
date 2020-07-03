"""
    Pre pare a class that encapsulate all the things need for
    the branch and bound algorithm.
    Together with some good helper methods to aid things a bit.

"""

__all__ = ["knapsack_dp_primal", "knapsack_dp_dual", "Knapsack"]
from typing import *
RealNumber = Union[float, int]
import math


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
    :param profitsUpper:
        Don't mess with this variable, it's for the knapsack class.
    :return:
        The set of indices representing the solution, and the optimal value of the solution.
    """
    assert len(profits) == len(weights), "weights and length must be in the same length. "
    assert min(profits) >= 0 and min(weights) >= 0, \
        "item profits and weight must be non-negative. "
    assert sum([1 for W in weights if W > budget]) == 0, \
        "All items must be weights less than maxWeight to reduce redundancies"

    TotalProfits = int(knapsack_greedy(profits, weights, budget)) + 1
    T = [float("+inf")] * (TotalProfits + 1)
    T[0] = 0
    Soln = [[] for P in range(TotalProfits + 1)]  # Store the indices of item that sum up to that exact profits
    for I in range(len(profits)):
        newT = [float("nan")]*len(T)
        SolnCopied = [S.copy() for S in Soln]
        for P in range(len(T)):
            NewWeight = T[P - profits[I]] + weights[I] if P - profits[I] >= 0 else float("inf")
            if NewWeight < T[P]:
                SolnCopied[P] = Soln[P - profits[I]] + [I]
                newT[P] = NewWeight
            else:
                newT[P] = T[P]
        Soln = SolnCopied
        T = newT
    BestFeasibleProfit = [I for I in range(len(T)) if T[I] <= budget][-1]
    Res = Soln[BestFeasibleProfit]  # Index of the highest feasible profits.
    return Res, BestFeasibleProfit


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
    return Soln[T.index(P_star)], P_star


def knapsack_greedy(profits, weights, budget):
    assert len(profits) == len(weights), "weights and length must be in the same length. "
    assert min(profits) >= 0 and min(weights) >= 0, \
        "item profits and weight must be non-negative. "
    assert sum([1 for W in weights if W > budget]) == 0, \
        "All items must be weights less than maxWeight to reduce redundancies"

    M = [(profits[I] / weights[I], I) if weights[I] > 0 else (float("+inf"), I) for I in range(len(profits))]
    M.sort(key=(lambda x: x[0]), reverse=True)
    Solution, TotalProfits, RemainingBudget = {}, 0, budget  # solution: index |-> (0, 1]
    for _, Index in M:
        ItemW, ItemP = weights[Index], profits[Index]
        if RemainingBudget - ItemW <= 0:
            Solution[Index] = RemainingBudget / ItemW
            TotalProfits += ItemP * (RemainingBudget / ItemW)
            RemainingBudget = 0
            break
        RemainingBudget -= ItemW;
        TotalProfits += ItemP
        Solution[Index] = 1
    return TotalProfits

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

    def fractional_approx(self, moreInfo=False):
        """
            Allowing fractional item, estimate the upper bound for the problem (Greedy Algorithm)

            * The solution will have a loser bound than integral dual approx, however, I didn't prove if it's always the
            case, it depend on epsilon and the inputs.

            * Returns the amount of profits from the fractional item. This will indicate how lose te upper bound is.
            It's going to be between 0, and 1/2, 0 means the solution is integral, and it's the global optimal, and if
            it's 1/2, then it means the upper bound is really lose.
        :param moreInfo:
            True: then it will return one additional parameter telling how good the upper bound is.
            False: It won't return the fractional item's profits.
        :return:
            The optimal value as the upper bound, and the optimal value of the solution, in the format of:
            Solution, TotalProfits.

            The solution is a map, mapping the index of the item to the fractional value of that item which is being
            taken.
        """
        P, W, B = self.__p, self.__w, self.__b  # Profits, weights, and budget.
        # tuples, first element is the value, second element is the index of the item.
        M = [(P[I]/W[I], I) if W[I] > 0 else (float("+inf"), I) for I in range(len(P))]
        M.sort(key=(lambda x: x[0]), reverse=True)
        Solution, TotalProfits, RemainingBudget = {}, 0, B # solution: index |-> (0, 1]
        FractionalProfits = None
        for _, Index in M:
            ItemW, ItemP = W[Index], P[Index]
            if RemainingBudget - ItemW <= 0:
                Solution[Index] = RemainingBudget/ItemW
                FractionalProfits = ItemP*(RemainingBudget/ItemW)
                TotalProfits += FractionalProfits; RemainingBudget = 0
                break
            RemainingBudget -= ItemW; TotalProfits += ItemP
            Solution[Index] = 1
        return (Solution, TotalProfits) if not moreInfo else (Solution, TotalProfits, FractionalProfits/TotalProfits)

    def dual_approx(self, superFast=True):
        """
            * Gives an integral solution that is feasible, together with an estimated upperbound for the true optimal
            using this set of items.

            Scale the profits and make them into integers. It will try to minimize the scaling factor, it will seek
            for an scale such that each scaled profits are at least 1 away from each, (So all different profits lies
            in their own integer slots. )

            * Polynomial run-time.

        :param superFast:
            This make things runs super fast, but it can't give you an upper bound for the optimal solution.
            * The solution it returns can get arbitrarily bad, but most of the time it's OK.
        :return:
            An Integral solution that is definitely feasible, and its optimal value.
        """
        P, W, B, eps = self.__p, self.__w, self.__b, self.__epsilon # Profits, weights, and budget.
        N = len(P)
        OptUpperBound = self.fractional_approx()[1]
        def best_scaling():

        Scale = N/(eps*max(P)) if not superFast else OptUpperBound/max(P)
        ScaledProfits = [int(Profit*Scale) for Profit in P]
        Soln, _ = knapsack_dp_dual(ScaledProfits, W, B)
        Opt = sum(P[I] for I in Soln)
        return Soln, Opt

    def primal_approx_upper(self):
        """
            Get a integral solution that may or may not be feasible, if it's infeasible, then it's an upper bound, else
            it's the optimal solution.
        :return:
            solution, optimal value.
        """
        return self.__primal_approx(mode=1)

    def __primal_approx(self, mode):
        """
            Internal use.
        :param mode:
            1: Over estimation; 2: Under estimation.
        :return:
            The integral solution of the approx, the objective value of the solution.
        """
        weights, maxWeight, profits, epsilon = self.__w, self.__b, self.__p, self.__epsilon
        WeightMax = max(weights)
        Multiplier = math.log2(WeightMax) / (WeightMax * epsilon)  # This scales all weights
        MaxWeightModified = int(Multiplier * maxWeight)
        ScaledWeights = [(int(W * Multiplier) + 1 if mode == 0 else int(W * Multiplier))\
                         for W in weights]
        Soln, objectiveValue = knapsack_dp_primal(profits, ScaledWeights, MaxWeightModified)
        return Soln, objectiveValue

    def primal_approx_lower(self):
        """
            Get an integral solution really fast, it's feasible.

            The approx solution can be arbitrarily bad for pathological inputs.
        :return:
            Solution, optimal value.
        """
        return self.__primal_approx(mode = 2)

    def upper_bound_tightiest(self):
        """
            Gives the tightest upper bound without bruteforcing.

            The sense of optimality is given by the value of epsilon.
        :return:
            The upper bound, no solution will be returned.
        """

        pass

    def approx_fastest(self, moreInfo=False):
        """
            Analyze the problem intelligently, and then according to the problem, tailor a feasible solution
            in the faster way possible.

            Optimality can be arbitarily bad.

        :param moreInfo:
            If this is set to true, then it will return "True" to indicate the the error is confidently within
            epsilon, if "False" is returned, then it will indicate that the this solution has optimal value
            less than (1-epsilon)*P_star, where P_sar is the absolute optimal.
        :return:
            The optimal solution and it's optimal value.
        """
        Soln1, Opt1, FracSlack = self.fractional_approx(moreInfo=True)
        OptLowerThreshold = Opt1*(1 - self.__epsilon)
        Opt1 = (1-FracSlack)*Opt1
        Soln2, Opt2 = self.dual_approx()  # super fast

        if Opt1 > Opt2:
            if moreInfo:
                return [I for I, V in Soln1.items() if Soln1[I] == 1], Opt1, Opt1 > OptLowerThreshold
            return [I for I, V in Soln1.items() if Soln1[I] == 1], Opt1
        if moreInfo:
            return Soln2, Opt2, Opt2 > OptLowerThreshold
        return Soln2, Opt2

    def approx_best(self):
        """
            Give the best feasible solution that definitely is at least (1 - epsilon)*P_star where P_star is the
            absolute optimal of the solution.
        :return:
            The optimal solution and its optimal value.
        """
        Soln, Opt, Confidence = self.approx_fastest(moreInfo=True)
        if not Confidence:
            Soln, Opt = self.dual_approx(superFast=False)
        return Soln, Opt



    @property
    def epsilon(self):
        return self.epsilon

    @epsilon.setter
    def epsilon(self, eps):
        assert 0 < eps < 1, "Epsilon out of range. "
        self.__epsilon = eps


def Branch_and_bound():
    pass


def main():

    print(knapsack_dp_dual([2, 3, 2, 1], [6, 7, 4, 1], 9))
    print(knapsack_dp_primal([2, 3, 2, 1], [6, 7, 4, 1], 9))

    def test_frac_approx():
        K = Knapsack([2, 3, 2, 1], [6, 7, 4, 1], 9)
        print(K.fractional_approx())
        print(K.dual_approx())
    test_frac_approx()


if __name__ == "__main__":
    main()