"""
    Pre pare a class that encapsulate all the things need for
    the branch and bound algorithm.
    Together with some good helper methods to aid things a bit.

"""

__all__ = ["knapsack_dp_primal", "knapsack_dp_dual", "Knapsack", "branch_and_bound"]
from typing import *
RealNumber = Union[float, int]
import math
from collections import namedtuple
from typing import *


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
        RemainingBudget -= ItemW
        TotalProfits += ItemP
        Solution[Index] = 1
    return TotalProfits


class Knapsack:
    """
        This class is designed to serve for branch and bound.
        This class takes in arrays of item's profits, item's weights, and item's budget.

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
        assert min(w) >= 0 and min(p) >= 0, "All weights and profits must be positive. "

    def greedy_approx(self, moreInfo=False):
        """
            Allowing fractional item, estimate the upper bound for the problem (Greedy Algorithm)

            * The solution will have a loser bound than integral dual approx, however, I didn't prove if it's always the
            case, it depend on epsilon and the inputs.

            * Returns the amount of profits from the fractional item. This will indicate how lose te upper bound is.
            It's going to be between 0, and 1/2, 0 means the solution is integral, and it's the global optimal, and if
            it's 1/2, then it means the upper bound is really lose.
        :param moreInfo:
            True: Returns the percentage of the profit that the fractional item contributes to the total greedy
            estimation.
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
        FractionalProfits = 0
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

    def dual_approx(self, superFast=True, moreInfo=False):
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
        :param moreInf:
            Gets an additional value representing the upperbound for the problem a bound that the optimal is
            not going to escape.
        :return:
            An Integral solution that is definitely feasible, and its optimal value.

        """
        P, W, B, eps = self.__p, self.__w, self.__b, self.__epsilon # Profits, weights, and budget.
        N = len(P)
        OptUpperBound = self.greedy_approx()[1]
        Scale = self.__dual_scale_best() if not superFast else OptUpperBound / max(P)
        ScaledRoundedProfits = [int(Profit*Scale) for Profit in P]
        Soln, _ = knapsack_dp_dual(ScaledRoundedProfits, W, B)
        Opt = sum(P[I] for I in Soln)
        if moreInfo:
            UpperBound = sum(ScaledRoundedProfits[I] for I in Soln)/(Scale*(1 - eps))
            return Soln, Opt, UpperBound
        return Soln, Opt

    def __dual_scale_best(self):
        """
            Try to find a best scale that make the optimal solution constrained by epsilon.
            * Epsilon Scale:
                * Precision is promised by it can be an over kill when profits are spread out.
            * MinDiffScale:
                * The solution is likely to be very close, but it depends on the items with the closest profits to
                each other.
        :return:
            The scaling factor, a float value.
        """
        N, P, eps = len(self.__p), self.__p, self.__epsilon
        EpsilonScale = N / (eps * max(P))
        Psorted = P.copy()
        Psorted.sort()
        MinDiff = min(I1 - I0 for I0, I1 in zip(Psorted[:-1], Psorted[1:]) if I0 - I1 != 0)
        MinDiffScale = 2 / MinDiff  # This assures the rounding won't change the relative size of all profits.
        return min(EpsilonScale, MinDiffScale)

    def dual_eps_upperbound(self):
        """
            Get an error bound from the rounding and scaling algorithm fro the dual approximation.
        :return:
            An error bound is is definitely bounded, and the error is inversely proportional to 1-epsilon.
        """
        return self.dual_approx(superFast=False, moreInfo=True)[-1]

    # def __primal_scale_exact(self):
    #     """
    #         Return the scaling factor such that, when it's applied to the primal approx,
    #         it will return the optimal solution that is the exact solution.
    #
    #         * The scaling factor assures that all weights of items are at least 2 apart from each other.
    #     :return:
    #         A positive float value, in float.
    #     """
    #     W = self.__w.copy()
    #     Wsorted = W.sort()
    #     MinDiff = min(I1 - I0 for I0, I1 in zip(Wsorted[:-1], Wsorted[1:]) if I0 - I1 != 0)
    #     return 2/MinDiff
    #
    #
    # def primal_approx_upper(self):
    #     """
    #         Get a integral solution that may or may not be feasible, if it's infeasible, then it's an upper bound, else
    #         it's the optimal solution.
    #     :return:
    #         solution, optimal value.
    #     """
    #     return self.__primal_approx(mode=1)
    #
    # def __primal_approx(self, mode, forceMultiplier = None):
    #     """
    #         Internal use.
    #     :param mode:
    #         1: Over estimation; 2: Under estimation.
    #     :return:
    #         The integral solution of the approx, the objective value of the solution.
    #     """
    #     weights, maxWeight, profits, epsilon = self.__w, self.__b, self.__p, self.__epsilon
    #     WeightMax = max(weights)
    #     Multiplier =len(weights) / (WeightMax * epsilon) if forceMultiplier is None else forceMultiplier
    #     MaxWeightModified = int(Multiplier * maxWeight)
    #     ScaledWeights = [(int(W * Multiplier) + 1 if mode == 0 else int(W * Multiplier))\
    #                      for W in weights]
    #     Soln, objectiveValue = knapsack_dp_primal(profits, ScaledWeights, MaxWeightModified)
    #     return Soln, objectiveValue
    #
    # def primal_approx_lower(self):
    #     """
    #         Get an integral solution really fast, it's feasible.
    #
    #         The approx solution can be arbitrarily bad for pathological inputs.
    #     :return:
    #         Solution, optimal value.
    #     """
    #     return self.__primal_approx(mode = 2)
    #
    # def upper_bound_tightiest(self):
    #     """
    #         Gives the tightest upper bound without bruteforcing.
    #
    #         The sense of optimality is given by the value of epsilon.
    #     :return:
    #         The upper bound, no solution will be returned.
    #     """
    #
    #     pass

    def approx_fastest(self, moreInfo=False):
        """
            Analyze the problem intelligently, and then according to the problem, tailor a feasible solution
            in the faster way possible.

            * Uses super fast dual approx
            * Uses greedy approx.

            Optimality can be arbitarily bad.
        :param moreInfo:
            If this is set to true, then it will return "True" to indicate the the error is confidently within
            epsilon, if "False" is returned, then it will indicate that the this solution has optimal value
            less than (1-epsilon)*P_star, where P_sar is the absolute optimal.
        :return:
            A approximated integral solution and it's optimal value.

            Moreinfo:
                If the optimal solution (That is returned)
                is higher than (1-epsilon) portion of the greedy, non-integral solution.
        """
        Soln1, Opt1, FracSlack = self.greedy_approx(moreInfo=True)
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

            * If not confidence on the fast approximation, it will try to compare primal scaling factor and dual
            scaling factor.
                ** if primal is faster, then it will see if primal's solution is good enough,
                    ** if not, it will run dual's approximation.
                ** If primal is not faster, then it will just run dual's approximation.
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

    @property
    def profits(self):
        """
            No, you can't modify it.
        :return:
            The copied list of tiems profits for the items. s
        """
        return self.__p.copy()

    def __len__(self):
        return len(self.__p)


class Problem:

    def __init__(this, itemsIncluded, remainingItems, weights, profits, budget):
        assert budget >= 0, "No negative budget is allowed. "
        this.ItemsIncluded = itemsIncluded
        this.RemainingItems = remainingItems
        this.Weights = weights
        this.Profits = profits
        this.Budget = budget

    def greedy_subset(this):
        """
            Solve a sub problem with a given set of indices.
        :param indices:
        :return:
            Fractional item's index, Integral item's index, optimal value.
        """
        indices = this.RemainingItems
        if this.Budget == 0:
            return None, [I for I in indices if this.Weights[I] == 0], \
                   sum(this.Profits[I] for I in indices if this.Weights[I] == 0)
            # bruh...
        # prune, for preconditions:
        indices = [I for I in indices if this.Weights[I] <= this.Budget]
        SolutionIndexRemap = [None]*len(indices)
        P, W = [None]*len(indices), [None]*len(indices)  # profits, weights for sub-problem
        for I, Index in enumerate(indices):
            SolutionIndexRemap[I] = Index
            P[I] = this.Profits[Index]
            W[I] = this.Weights[Index]
        SubProblem = Knapsack(P, W, this.Budget)
        S, _ = SubProblem.greedy_approx()
        _, Opt = SubProblem.approx_best()
        FractionalItemIndex = None
        for K, V in S.items():
            if V != 1:
                FractionalItemIndex = K
                break
        FractionalItemIndex = None if FractionalItemIndex is None else SolutionIndexRemap[FractionalItemIndex]
        return FractionalItemIndex, [SolutionIndexRemap[K] for K, V in S.items() if V == 1], Opt


def branch_and_bound(profits, weights, budgets):
    """
        Iterative implementation of the branch and bound algorithm on Knapscak problem.
    :param solution:
        This is the solution that is assumed to be the best and will be improve upon.
    :param objectiveValue:
        This is the objective value given by the best solution.
    :return:
        The optimal solution.
    """
    """
        ItemsIncluded: 
            List of indices that are defined by the super problems. 
        RemainingItems: 
            List of indices that the sub-problems are going to investigate. 
        !! If an item's index are in neither, then it's excluded by the algorithm. 
    """

    def InitialProblem():
        return Problem([], [I for I in range(len(profits))], weights, profits, budgets)

    def SpawnProblems(problem, U_star, S_star):
        """
            Problem will produce a 2 sets of indices S1, S2, and an optimal value, opt.

            S1: The integral item's indices by greedy algorithm.
            S2: The index for the item that's fractional.

            Cases:
            1. No spawn no update. (Greedy worst than best Integral opt)
            2. Spawn and update.
            3. No spawn but update. (Register Current Integral Solution)
        :param P:
            Problem type named tuples.
        :param UpperBound:
            The best upperbound giver by the integral solution.
        :return:
            tuple， first integer representing different cases, handled by the caller.
        """
        W, P, B = problem.Weights, problem.Profits, problem.Budget
        FracSolnIdx, IntSolnIdx, Opt = problem.greedy_subset()
        U_tilde = sum(P[I] for I in problem.ItemsIncluded) + Opt
        # Things to return.
        P1, P2 = None, None
        # Branch
        if (U_star is None ) or (U_tilde > U_star):
            if FracSolnIdx is not None:
                ItemsIncludedCopy = problem.ItemsIncluded.copy()
                RemainingItemsCopy = problem.RemainingItems.copy()
                ItemsIncludedCopy += [FracSolnIdx]  # only includes the fractional item
                RemainingItemsCopy.remove(FracSolnIdx)
                Budget = B - sum(W[I] for I in ItemsIncludedCopy)
                if Budget >= 0:
                    P1 = Problem(ItemsIncludedCopy, RemainingItemsCopy, W, P, Budget)
                ItemsIncludedCopy = problem.ItemsIncluded.copy()
                P2 = Problem(ItemsIncludedCopy, RemainingItemsCopy, W, P, B - sum(P[I] for I in ItemsIncludedCopy))
            else:
                U_star = U_tilde  # Update, integral...
        # check if update S_star
        if (S_star is None) or \
                (sum(P[I] for I in problem.ItemsIncluded) + sum(P[I] for I in IntSolnIdx) > sum(P[I] for I in S_star)):
            S_star = problem.ItemsIncluded + IntSolnIdx
        return U_star, S_star, P1, P2

    Stack = [InitialProblem()]
    U_star, S_star = None, None # Best upper-bound and Best feasible solution that gives the upper bound.
    while len(Stack) > 0:
        U_star, S_star, P1, P2 = SpawnProblems(Stack.pop(), U_star, S_star)
        if P1 is not None:
            Stack.append(P1)
        if P2 is not None:
            Stack.append(P2)
    return U_star, S_star




def main():
    print(knapsack_dp_dual([2, 3, 2, 1], [6, 7, 4, 1], 9))
    print(knapsack_dp_primal([2, 3, 2, 1], [6, 7, 4, 1], 9))
    def test_frac_approx():
        K = Knapsack([2, 3, 2, 1], [6, 7, 4, 1], 9)
        print(K.greedy_approx())
        print(K.dual_approx())
    test_frac_approx()
    def test_BB():
        print(branch_and_bound([2, 3, 2, 1], [6, 7, 4, 1], 9))
        print(branch_and_bound([1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 6], 10))
    test_BB()


if __name__ == "__main__":
    main()