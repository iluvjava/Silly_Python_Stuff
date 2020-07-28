"""
    The extended knapsack problem:
        Just items but with the extra attributes of "Counts"

    This problem is really an Linear programming problem where the
    simplex algorithm is simplified.
"""
from typing import *

RealNumberList = List[Union[float, int]]

class EknapsackProblem:
    """
        * problem Description:
            ** The item's weight are positive real numbers.
            ** The item's profits are positive real numbers.
            ** The item's counts are positive real numbers.
            ** The total budge allowed.
        This represents an extendned knapsack problem.
        * It will use as little resources as possible so it's efficient for the branch and bound algorithm.

        * The whole problem is encapsulated in an instance of the class,
        to solve a sub-problem on the BB tree, simply passes in the partial solution and an index set
        indicating the items and the sub-problem can USE for ITS SOLUTION.

        **
            If I in indices: Then, I can be added to the knapsack.
            If I not in indices:
                If I is in the partial solution, then it's constrained by <= floor(x_i) y the parent nodes in the BB
                tree.
                If I is not in the partial solution, then something is wrong, because the variable is free, but
                    it's not investigated any this problem and any of the sub-problems of this sub-problem.

    """
    def __init__(self,
                 profits: RealNumberList,
                 weights: RealNumberList,
                 counts: RealNumberList,
                 budget: RealNumberList):
        """

        :param profits:
        :param weights:
        :param counts:
        :param budget:
        """
        assert all(I >= 0 for I in counts), "Item's counts cannot be negative."
        assert all(len(I) == self.Size for I in [profits, weights, counts, budget])
        self._P, self._W, self._C, self._B = profits, weights, counts, budget

    def greedy_solve(self, AlreadyDecidedSoln):
        """
            This function serves as heuristics for the BB, and it returns the list of
            all integral soltution, the fractional item, and the objective value
            of the greedy approximated solution.

            ** It will evaluate "AlreadyDecideSoln" first and then to decide on the
            further solution.
        :param AlreadyDecidedSoln:
            The constraints accumulated via the BB algorithm, limiting the options for choosing
            further solution.
            {
                "item's Idx": "How much of this is already in knapsack",
                ...
            }
        :return:
            {
                "item's Idx": "How much of this item is in the knapsack"
                ...
            }
            ,
            ("Item's Idx", The fraction amount of item taken)
            ,
            Objective value of the solution.
        """
        def SubSlicing(self, toSlice):
            # Copied.
            return [toSlice for I in self.Indices]

        def SubSolving(P, W, C, B):
            Values = [((P[I]/W[I], I) if W[I] != 0 else float("+inf")) for I in len(P)]
            Values.sort(key=(lambda x: x[0]), reverse=True)
            Values = [I for _, I in Values]
            FractionalItems, IntegralItems, ObjectiveValue = {}, {}, 0
            RemainingBudge = B
            for Idx in Values:
                if W[Idx] == 0:
                    ObjectiveValue += P[Idx]
                    IntegralItems[Idx] = C[Idx]
                else:
                    if RemainingBudge == 0:
                        break
                    ToTake = min(RemainingBudge/W[Idx], C[Idx])
                    if int(ToTake) == ToTake:
                        IntegralItems[Idx] = ToTake
                    else:
                        FractionalItems[Idx] = ToTake
                    RemainingBudge -= ToTake*W[Idx]
                    ObjectiveValue += ToTake*P[Idx]
            return IntegralItems, FractionalItems, ObjectiveValue
        # Problem Digest------------------------------------------------------------------------------------------------
        C = self._C.copy()
        for K, V in AlreadyDecidedSoln:
            # Take partial solution into account.
            C[K] -= V
        P, W, C, B = SubSlicing(self._P),  SubSlicing(self.W),  SubSlicing(C),  SubSlicing(self._B)
        # The index in the sub problem is remapped to indices in the original problem.
        IdxInverseMap = [-1]*self.Size
        for I, V in self.Indices:
            IdxInverseMap[I] = V
        # End ----------------------------------------------------------------------------------------------------------
        # Merge the solution -------------------------------------------------------------------------------------------
        IntegralItems, FractionalItems, ObjVal = SubSolving(P, W, C, B)


        pass

    @property
    def Indices(self):
        return self.Indices.copy()

    @Indices.setter
    def Indices(self, indices: List[int]):
        self.Indices = indices.copy()

    @property
    def Size(self):
        return self(self._P)

