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
                *** If I is in the partial solution, then it's constrained by <= floor(x_i) y the parent nodes in the BB
                tree.
                *** If I is not in the partial solution, then something is wrong, because the variable is free, but
                    it's not investigated any this problem and any of the sub-problems of this sub-problem.

    """
    def __init__(self,
                 profits: RealNumberList,
                 weights: RealNumberList,
                 counts: RealNumberList,
                 budget: RealNumberList):
        """
            Construct the problem with all these elements, this represent a root problem
            for the knapsack problem.
        :param profits:
            Float list, all positive.
        :param weights:
            Weights, all positive.
        :param counts:
            The limit of number an item can be taken.
        :param budget:

        """
        assert all(I >= 0 for I in counts), "Item's counts cannot be negative."
        assert all(len(I) == len(profits) for I in [profits, weights, counts])
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
            Soln, ObjectiveValue = {}, 0
            RemainingBudge = B
            for Idx in Values:
                if W[Idx] == 0:
                    ObjectiveValue += P[Idx]
                    Soln[Idx] = C[Idx]
                else:
                    if RemainingBudge <= 0:
                        break
                    ToTake = min(RemainingBudge/W[Idx], C[Idx])
                    Soln[Idx] = ToTake
                    RemainingBudge -= ToTake*W[Idx]
                    ObjectiveValue += ToTake*P[Idx]
            return Soln, ObjectiveValue

        # Problem Digest------------------------------------------------------------------------------------------------
        C = self._C.copy()
        for K, V in AlreadyDecidedSoln.items():
            # Take partial solution into account.
            C[K] -= V
        B = self._B - sum(self._W[K]*V for K, V in AlreadyDecidedSoln.items())
        P, W, C = SubSlicing(self._P), SubSlicing(self.W), SubSlicing(C)

        # Index in sub |==> Index in full problem.
        IdxInverseMap = [-1]*self.Size
        for I, V in enumerate(self.Indices):
            IdxInverseMap[I] = V
        # End ----------------------------------------------------------------------------------------------------------

        # solve and merge the solution ---------------------------------------------------------------------------------
        Soln, ObjVal = SubSolving(P, W, C, B)
        SolnRemapped = {}
        for K, V in Soln.items():
            SolnRemapped[IdxInverseMap[K]] = V
        for K, V in SolnRemapped.items():
            AlreadyDecidedSoln[K] += V
        # End ----------------------------------------------------------------------------------------------------------
        return AlreadyDecidedSoln

    @property
    def Indices(self):
        return self.Indices.copy()

    @Indices.setter
    def Indices(self, indices: List[int]):
        self.Indices = indices.copy()

    @property
    def Size(self):
        return len(self._P)


def main():

    def TestKnapSack():
        P, W, B, C = [2, 3, 2, 1], [6, 7, 4, 1], 9, [1]*4
        EKnapSack = EknapsackProblem(P, W, C, B)
        print(f"Instance: {EKnapSack}")
        pass
    TestKnapSack()


if __name__ == "__main__":
    main()