"""
    The extended knapsack problem:
        Just items but with the extra attributes of "Counts"

    This problem is really an Linear programming problem where the
    simplex algorithm is simplified.

    ! Warning Lack of Equivalence compare to the BB algorithm for LP:
        In BB, The braching asserts new constraint such as x_i < floor(x_i_tilde)
        But this is not the equivalent of setting the variable to that value and then exluding it
        from further recursion.
    ! Yes, for standard knapsack, if we lower the counts of item that has been selected for the greedy solution,
    then that item will still be selected for the number of counts in future recursion.
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

        ** This class is going to handle the branching of the BB also, so it makes the codes for the BB Very
        high level.

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

        # Initialize as an root problem for bb
        self._PartialSoln = dict(zip(range(self.Size), [0]*self.Size))
        self._Indices = set(range(self.Size))

    def greedy_solve(self):
        """
            This function serves as heuristics for the BB, and it returns the list of
            all integral soltution, the fractional item, and the objective value
            of the greedy approximated solution.

            ** It will evaluate "AlreadyDecideSoln" first and then to decide on the
            further solution.

            ** Function will modify the partial solution that for this class, a reference will be returned.
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
        def SubSlicing(toSlice, Indices):
            # Copied.
            return [toSlice[I] for I in Indices]

        def SubSolving(P, W, C, B):
            Values = [((P[I]/W[I], I) if W[I] != 0 else float("+inf")) for I in range(len(P))]
            Values.sort(key=(lambda x: x[0]), reverse=True)
            Values = [I for _, I in Values]
            Soln = {}
            RemainingBudge = B
            for Idx in Values:
                if W[Idx] == 0:
                    Soln[Idx] = C[Idx]
                else:
                    if RemainingBudge <= 0:
                        break
                    ToTake = min(RemainingBudge/W[Idx], C[Idx])
                    Soln[Idx] = ToTake
                    RemainingBudge -= ToTake*W[Idx]
            return Soln

        # Problem Digest------------------------------------------------------------------------------------------------
        C = self._C.copy()
        AlreadyDecidedSoln = self._PartialSoln
        for K, V in AlreadyDecidedSoln.items():
            # Take partial solution into account.
            C[K] -= V
        B = self._B - sum(self._W[K]*V for K, V in AlreadyDecidedSoln.items())
        P, W, C = SubSlicing(self._P, self.Indices), SubSlicing(self._W, self.Indices), SubSlicing(C, self.Indices)

        # Index in sub |==> Index in full problem.
        IdxInverseMap = [-1]*self.Size
        for I, V in enumerate(self.Indices):
            IdxInverseMap[I] = V
        # End ----------------------------------------------------------------------------------------------------------
        # solve and merge the solution ---------------------------------------------------------------------------------
        Soln = SubSolving(P, W, C, B)
        SolnRemapped, FracIdx= {}, -1
        for K, V in Soln.items():
            SolnRemapped[IdxInverseMap[K]] = V
        for K, V in SolnRemapped.items():
            if int(V) != V:
                FracIdx = K
            if K not in AlreadyDecidedSoln:
                AlreadyDecidedSoln[K] = V
            else:
                AlreadyDecidedSoln[K] += V
        # End ----------------------------------------------------------------------------------------------------------
        return AlreadyDecidedSoln, sum(self._P[K]*V for K, V in AlreadyDecidedSoln.items()), FracIdx

    def branch(self, globalIntegralValue):
        """
            Branch this current instance.
        :param BestIntegralValue:
            Provide a best integral value, it's a must for the Heuristic Brancher.
        :return:
            Sub-problems, and renewed objective value and solution for the global landscape.
        """
        Soln, ObjVal, FracIdx = self.greedy_solve()
        # To Return:----------------------------------------------------------------------------------------------------
        NewSoln, NewObjVal, SubP1, SubP2, = None, None, None, None
        # END ----------------------------------------------------------------------------------------------------------
        IsIntegral = FracIdx != -1
        OptimalitySatisfied = ObjVal > globalIntegralValue
        # Pruned by optimality of the integral solution ----------------------------------------------------------------
        if IsIntegral and OptimalitySatisfied:
            NewSoln, NewObjVal = Soln, ObjVal
            return NewSoln, NewObjVal, SubP1, SubP2
        # Fractional, and it should branch -----------------------------------------------------------------------------
        if OptimalitySatisfied:
            # p1, Bound from above -------------------------------------------------------------------------------------
            PartialSoln = Soln.Copy()
            NewIndices = self.Indices
            NewIndices.remove(FracIdx)
            if int(PartialSoln[FracIdx]) != 0:
                PartialSoln[FracIdx] = int(PartialSoln[FracIdx])
            else:
                del PartialSoln[FracIdx]
            SubP1 = EknapsackProblem(self._P, self._W, self._C, self._B)
            SubP1.Indices = NewIndices
            SubP1.PartialSoln = PartialSoln
            # p2, Bound from below -------------------------------------------------------------------------------------
            PartialSoln = Soln.Copy()
            NewIndices = self.Indices
            PartialSoln[FracIdx] = int(PartialSoln[FracIdx]) + 1
            SubP2 = EknapsackProblem(self._P, self._W, self._C, self._B)
            SubP2.Indices = NewIndices
            SubP2.PartialSoln = PartialSoln

        return NewSoln, NewObjVal, SubP1, SubP2


    @property
    def Indices(self):
        return self._Indices.copy()

    @Indices.setter
    def Indices(self, indices: List[int]):
        self._Indices = indices.copy()

    @property
    def Size(self):
        return len(self._P)

    @property
    def PartialSoln(self):
        """
            If key is not specified, then the default value for x_i is going to be zero.

            The item it's taking is always gonna be integers.
        :return:
        """
        return self._PartialSoln

    @PartialSoln.setter
    def PartialSoln(self, item):
        self._PartialSoln = item

    @staticmethod
    def BB():
        """
            A static method for evaluation the whole Eknapsack problem

        :return:
            solution.
        """


        pass


def main():

    def TestKnapSack():
        P, W, B, C = [2, 3, 2, 1], [6, 7, 4, 1], 9, [1]*4
        EKnapSack = EknapsackProblem(P, W, C, B)
        print(f"Instance: {EKnapSack}")
        print(EKnapSack.greedy_solve())

        print("exclude item at index 1 (set x_1 = 0)")
        EKnapSack.Indices = {0, 2, 3}
        del EKnapSack.PartialSoln[1]
        print(EKnapSack.greedy_solve())
        print("Including item at index 1 (set x_1 = 1)")
        EKnapSack.Indices = {0, 1, 2, 3}
        EKnapSack.PartialSoln = {1: 1}
        print(EKnapSack.greedy_solve())

        pass
    TestKnapSack()


if __name__ == "__main__":
    main()