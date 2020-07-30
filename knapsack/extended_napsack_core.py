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

    TODO: Check the correctness of the algorithm with the Pul LP solver.
"""
from typing import *
import pulp as lp
import random as rnd

RealNumberList = List[Union[float, int]]

def make_extended_knapsack_problem(size:int, density:float):
    """
        Generate an extended knapsack problem where the integer solution of the problem is unqiue.
    :param size:
        The number of items that are going to be in the problem .
    :param density:
        The budget divides the total weight of all items.
    :return:
        [(p, w, c), ...], budget
    """
    assert 0 < density < 1, "density must be a quantity that is between 0 and 1. "
    ToReturn = [(rnd.random(),rnd.random(),int(rnd.random()*size)) for I in range(size)]





class EknapsackProblem:
    """
        * Preconditions:
            ** The item's weight are positive real numbers.
            ** The item's profits are positive real numbers.
            ** The item's counts are positive Integers.
            ** The total budge allowed.
        This represents an extended knapsack problem.
        * It will use as little resources as possible so it's efficient for the branch and bound algorithm.

        * The whole problem is encapsulated in an instance of the class,
        to solve a sub-problem on the BB tree, simply passes in the partial solution and an index set
        indicating the items and the sub-problem can USE for ITS SOLUTION.

        * The solution is must be unique, this solver cannot find all linear combinations of the  optimal solution.

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
        # check if the problem will produce unique solution ------------------------------------------------------------
        Values = [((self._P[I] / self._W[I], I) if self._W[I] != 0 else float("+inf")) for I in range(self.Size)]
        Values.sort(key=(lambda x: x[0]), reverse=True)
        Values = [I for _, I in Values]
        self._SolutionUnique =  sum((1 if Values[I] == Values[I+1] else 0) for I in range(self.Size - 1)) <= 0  # ------

        # Initialize as an root problem for bb -------------------------------------------------------------------------
        self._PartialSoln = {}
        self._Indices = set(range(self.Size))  # -----------------------------------------------------------------------

        # The instance is only going to solve itself ONCE. -------------------------------------------------------------
        self.__GreedySoln, self._ObjVal, self._FractIndx = None, None, None  # -----------------------------------------


    def greedy_solve(self):
        """
            This function serves as heuristics for the BB, and it returns the list of
            all integral soltution, the fractional item, and the objective value
            of the greedy approximated solution.

            ** It will evaluate "AlreadyDecideSoln" first and then to decide on the
            further solution.

            ** Function will modify the partial solution that for this class, a reference will be returned.

            ** If the objective value return is negative infinity, then bingo, the passed in partial solution
            is not feasible.
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
        if self._ObjVal is not None:
            return self.__GreedySoln, self._ObjVal, self._FractIndx

        def SubSlicing(toSlice, Indices):
            # Copied.
            return [toSlice[I] for I in Indices]

        def SubSolving(P, W, C, B):
            Values = [((P[I]/W[I], I) if W[I] != 0 else float("+inf")) for I in range(len(P))]
            Values.sort(key=(lambda x: x[0]), reverse=True)  # Sort by item's values -----------------------------------
            Soln = {}
            RemainingBudge = B
            for _, Idx in Values:
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
        AlreadyDecidedSoln = self._PartialSoln.copy()
        for K, V in AlreadyDecidedSoln.items():
            # Take partial solution into account.
            C[K] -= V
        B = self._B - sum(self._W[K]*V for K, V in AlreadyDecidedSoln.items())
        # Infeasible, Pruned -------------------------------------------------------------------------------------------
        if B < 0:
            self._ObjVal = float("-inf")
            return None, float("-inf"), None
        P, W, C = SubSlicing(self._P, self.Indices), SubSlicing(self._W, self.Indices), SubSlicing(C, self.Indices)
        IdxInverseMap = [-1]*self.Size   # Index in sub |==> Index in full problem.
        for I, V in enumerate(self.Indices):
            IdxInverseMap[I] = V
        # solve and merge the solution ---------------------------------------------------------------------------------
        Soln = SubSolving(P, W, C, B)
        SolnRemapped, FracIdx = {}, -1
        for K, V in Soln.items():
            SolnRemapped[IdxInverseMap[K]] = V
        for K, V in SolnRemapped.items():
            if int(V) != V:
                FracIdx = K
            if K not in AlreadyDecidedSoln:
                AlreadyDecidedSoln[K] = V
            else:
                AlreadyDecidedSoln[K] += V
        self.__GreedySoln = AlreadyDecidedSoln
        self._ObjVal = sum(self._P[K]*V for K, V in AlreadyDecidedSoln.items())
        self._FractIndx = FracIdx
        # Returning the solution ---------------------------------------------------------------------------------------
        return self.__GreedySoln, self._ObjVal, self._FractIndx

    def branch(self, globalIntegralValue, NewSoln=None):
        """
            Branch this current instance.
        :param BestIntegralValue:
            Provide a best integral value, it's a must for the Heuristic brancher.
        :param NewSoln:
            The best global integral solution, if not current integral solutions are found or it's not as good as
            the given one, it will just return this parameter.
        :return:
            Sub-problems, and renewed objective value and solution for the global landscape.
        """
        Soln, ObjVal, FracIdx = self.greedy_solve()
        # To Return:----------------------------------------------------------------------------------------------------
        NewObjVal, SubP1, SubP2, = globalIntegralValue, None, None
        IsIntegral = FracIdx == -1
        OptimalitySatisfied = ObjVal > globalIntegralValue
        # Pruned by optimality of the integral solution ----------------------------------------------------------------
        if IsIntegral and OptimalitySatisfied:
            NewSoln, NewObjVal = Soln, ObjVal
            return NewSoln, NewObjVal, SubP1, SubP2
        # Fractional, and it should branch -----------------------------------------------------------------------------
        if OptimalitySatisfied:
            # p1, Bound from above -------------------------------------------------------------------------------------
            PartialSoln = self.PartialSoln.copy()
            NewIndices = self.Indices
            NewIndices.remove(FracIdx)
            if int(Soln[FracIdx]) != 0:
                PartialSoln[FracIdx] = int(Soln[FracIdx])
            SubP1 = EknapsackProblem(self._P, self._W, self._C, self._B)
            SubP1.Indices = NewIndices
            SubP1.PartialSoln = PartialSoln
            # p2, Bound from below -------------------------------------------------------------------------------------
            PartialSoln = self.PartialSoln.copy()
            NewIndices = self.Indices
            PartialSoln[FracIdx] = int(Soln[FracIdx]) + 1
            SubP2 = EknapsackProblem(self._P, self._W, self._C, self._B)
            SubP2.Indices = NewIndices
            SubP2.PartialSoln = PartialSoln

        return NewSoln, NewObjVal, SubP1, SubP2

    def __repr__(self):
        s = "\n" + "-"*20 + "\n"
        s += "EKnapSack Instance: \n"
        s += f"Size: {self.Size}\n"
        s += f"Non-Fixed Variables: {self.Indices}\n"
        s += f"Partial Solution: {self.PartialSoln}\n"
        s += f"Greedy Soluion: {self.__GreedySoln}\n"
        s += f"Upperbound (Objective Value from Greedy algo): {self._ObjVal}\n"
        return s

    def __getitem__(self, I):
        """
            Quick tuple indexing to get desirable information regarding a certain item.
        :param item:
        :return:
        """
        return [self._P, self._W, self._C][I[0]][I[1]]
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
    def BB(rootProblem, verbose=False):
        """
            A static method for evaluation the whole Eknapsack problem
        :param rootProblem:
            An instance of an root problem, representing the initial problem for the initial knapsack.
        :return:
            optimal solution.
        """
        def Initialization():
            S, _, _ = rootProblem.greedy_solve()
            S = dict([(I, V) for I, V in S.items() if int(V) == V])
            ObjVal = sum(rootProblem[0, I]*V for I, V in S.items())
            if verbose:
                print(f"BB executing with war start solution and objective value: ")
                print(S)
                print(ObjVal)
            return S, ObjVal
        GIntSoln, GObjVal = Initialization()
        Stack = [rootProblem]
        while len(Stack) != 0:
            P = Stack.pop()
            P.greedy_solve()
            if verbose:
                print(P)
            GIntSoln, GObjVal, SubP1, SubP2 = P.branch(GObjVal, GIntSoln)
            if SubP1 is not None:
                Stack.append(SubP1)
            if SubP2 is not None:
                Stack.append(SubP2)
        return GIntSoln, GObjVal




class EknapsackSimplex:
    """
        This class will reduce the problem to LP, so it can be compare with the
        BB native python implementations for correctness and efficiency.

    """

    def __init__(self,
                 profits: RealNumberList,
                 weights: RealNumberList,
                 counts: RealNumberList,
                 budget: RealNumberList):
        self._P, self._W, self._C, self._B = profits, weights, counts, budget
        L = len(self._P)
        assert all(len(I) == L for I in [self._W, self._C])
        self._LP = None
        self._X = None

    def formulate_lp(self):
        n = len(self._P)
        Problem = lp.LpProblem(name="EknapSack", sense=lp.LpMaximize)
        X = lp.LpVariable.dict(name='x', indexs=range(n), lowBound=0, cat=lp.LpInteger)
        Problem += lp.lpSum(X[I]*self._P[I] for I in range(n))  # Objective.
        Problem += lp.lpSum(X[I]*self._W[I] for I in range(n)) <= self._B
        for I, V in enumerate(self._C):
            Problem += X[I] <= V
        self._LP, self._X = Problem, X
        return Problem

    def solve(self):
        """
            Solve the formultated LP problem and then return the results as a map.
        :return:
        """
        Soln = {}  # To return -----------------------------------------------------------------------------------------
        Problem = self.formulate_lp()
        Problem.solve()
        for I, Var in self._X.items():
            if Var.varValue != 0:
                Soln[I] = Var.varValue
        return Soln, sum(V*self._P[I] for I, V in Soln.items())

    @property
    def LpProblem(self):
        if self._LP is None:
            self.formulate_lp()
        return self._LP


def main():

    def TestKnapSack():
        P, W, B, C = [2, 3, 2, 1], [6, 7, 4, 1], 9, [1]*4
        EKnapSack = EknapsackProblem(P, W, C, B)

        print(EKnapSack.greedy_solve())
        EKnapSack.Indices = {0, 2, 3}
        del EKnapSack.PartialSoln[1]

        print(EKnapSack)
        print(EKnapSack.greedy_solve())

        EKnapSack.Indices = {0, 1, 2, 3}
        EKnapSack.PartialSoln = {1: 1}
        print(EKnapSack)
        print(EKnapSack.greedy_solve())

        pass

    def TestKnapSakBranchingMechanics():
        P, W, B, C = [2, 3, 2, 1], [6, 7, 4, 1], 9, [1] * 4
        EKnapSack = EknapsackProblem(P, W, C, B)
        print(EKnapSack)
        _, _, Sub1, Sub2 = EKnapSack.branch(float("-inf"))
        print(f"Sub1 {Sub1}")
        print(f"Sub2 {Sub2}")

    def RunBB():
        P, W, B, C = [2, 3, 2, 1], [6, 7, 4, 1], 9, [1] * 4
        EKnapSack = EknapsackProblem(P, W, C, B)
        print(EknapsackProblem.BB(EKnapSack))
        print("="*20)
        P, W, B, C = [2, 3, 2, 1], [6, 7, 4, 1], 9, [1, 1, 1, 2]
        EKnapSack = EknapsackProblem(P, W, C, B)
        print(EknapsackProblem.BB(EKnapSack))
        print("="*20)
        P, W, B, C = [1, 2, 3, 4], [4, 3, 2, 1], 8, [4, 2, 1, 1]
        EKnapSack = EknapsackProblem(P, W, C, B)
        print(EknapsackProblem.BB(EKnapSack))
        P, W, B, C = [1, 2, 3, 4], [4, 3, 2, 1], 8, [4, 2, 1, 4]
        EKnapSack = EknapsackProblem(P, W, C, B)
        print(EknapsackProblem.BB(EKnapSack))

    def LPFormulation():
        P, W, B, C = [2, 3, 2, 1], [6, 7, 4, 1], 9, [1] * 4
        EKnapSack = EknapsackSimplex(P, W, C, B)
        print(EKnapSack.formulate_lp())
        print(EKnapSack.solve())

        P, W, B, C = [2, 3, 2, 1], [6, 7, 4, 1], 9, [1] * 4
        EKnapSack = EknapsackProblem(P, W, C, B)
        print(EknapsackProblem.BB(EKnapSack))

    def CheckAgainstLP():
        pass



    RunBB()
    LPFormulation()


if __name__ == "__main__":
    main()