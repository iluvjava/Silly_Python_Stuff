"""
    The extended knapsack problem:
        Just items but with the extra attributes of "Counts"

    This problem is really an Linear programming problem where the
    simplex algorithm is simplified.

    ! Warning Lack of Equivalence compare to the BB algorithm for LP:
        In BB, The braching asserts new constraint such as x_i < floor(x_i_tilde)
        But this is not the equivalent of setting the variable to that value and then exluding it
        from further recursion.
    ! No, it's correct immediately after one level of branching, but it won't work if in future branching, because it's
    possible that, in future branching, that constraint x_i < floor(x_i_tilde) ceases to be a tight one.

    ====================================================================================================================
    TODO: Investiage this if possible:
    -- For this instance, the CBC pulp solver produced an sub-optimal solution to the problem.
        Failed on inputs: ([(0.14, 0.02, 2), (0.33, 0.9, 3), (0.65, 0.82, 2), (0.95, 0.28, 1), (0.64, 0.78, 2)], 1.866),
        obj is like: (2.37, 1.88)
        Solutions are like: ({4: 2, 3: 1, 0: 1}, {0: 2.0, 2: 1.0, 3: 1.0})

        Failed on inputs: ([(0.434, 0.5501, 2), (0.4285, 0.7953, 3), (0.0088, 0.3501, 2), (0.715, 0.6947, 3)],
        1.8811200000000001),
        obj is like: (1.583, 1.4387999999999999)
        Solutions are like: ({0: 2, 3: 1}, {2: 1.0, 3: 2.0})

    -- For this instance, my algorithm produced sub-optimal solution:
        Failed on inputs: ([(0.79, 0.38, 2), (0.35, 0.19, 4), (0.23, 0.01, 5), (0.89, 0.6, 4), (0.9, 0.63, 1)], 1.38),
        obj is like: (3.55, 3.7800000000000002)
        Solutions are like: ({1: 3, 0: 2, 2: 4}, {0: 2.0, 1: 3.0, 2: 5.0})
        
        Failed on inputs: ([(0.7109, 0.4548, 3), (0.1049, 0.8713, 1), (0.5754, 0.1237, 1), (0.7012, 0.6726, 1)],
        0.9095999999999999),
        obj is like: (1.2863, 1.4218)
        Solutions are like: ({2: 1, 0: 1}, {0: 2.0})


    ** Might be related the item's with weights that are incredibly close to zero.

    Conclusion:
        * The problem is not related to the numerical stability of inversing the constraint matrix.
        * Numerial instability is introduced by integral solution with a extremely small slack, which renders an
        infeasible solution feasible or vice versa.

    TODO: Fix the problem of numerical instability:
    -- Limited Scope Rational Computing.
        * Control it in the inner scope of the greedy solving method, that is the place where most of the
        computations happened.

"""
from typing import *
import pulp as lp
import random as rnd
import fractions as frac
import numpy as np
from kahan_summation import core as ksum

RealNumberList = List[Union[float, int]]


def make_extended_knapsack_problem(size: int, density: float, itemsCounts=5):
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
    ToReturn = [(rnd.random(), rnd.random(),  int(rnd.random()*itemsCounts) + 1) for _ in range(size)]
    ToReturn = list(map(lambda x: (round(x[0], 4), round(x[1], 4), x[2]), ToReturn))
    Budget = sum(W*C for _, W, C in ToReturn)*density
    return ToReturn, Budget


class EknapsackGreedy:
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
    Verbose = False
    def __init__(self,
                 profits: RealNumberList,
                 weights: RealNumberList,
                 counts: RealNumberList,
                 budget,
                 branchingIdentifier: str= ""):
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
            The constraight.
        :param branchingIdentifier:
            0, left branching, <= floor(x_tilde_i),
            1, right branching, >= ceiling(x_tilde_i)
        """
        assert all(I >= 0 for I in counts), "Item's counts cannot be negative."
        assert all(len(I) == len(profits) for I in [profits, weights, counts])
        self._P, self._W, self._C, self._B = profits, weights, counts, budget
        # check if the problem will produce unique solution ------------------------------------------------------------
        Values = [((self._P[I] / self._W[I], I) if self._W[I] != 0 else (float("+inf"), I)) for I in range(self.Size)]
        Values.sort(key=(lambda x: x[0]), reverse=True)
        Values = [I for _, I in Values]
        self._SolutionUnique =  sum((1 if Values[I] == Values[I+1] else 0) for I in range(self.Size - 1)) <= 0  # ------

        # Initialize as an root problem for bb -------------------------------------------------------------------------
        self._PartialSoln = {}
        self._Indices = set(range(self.Size))  # -----------------------------------------------------------------------

        # The instance is only going to solve itself ONCE. -------------------------------------------------------------
        self.__GreedySoln, self._ObjVal, self._FractIndx = None, None, None  # -----------------------------------------
        self._BranchingIdentifier = branchingIdentifier  # This is for debugging ---------------------------------------



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
            Values = [((P[I]/W[I], I) if W[I] != 0 else (float("+inf"), I)) for I in range(len(P))]
            Values.sort(key=(lambda x: x[0]), reverse=True)  # Sort by item's values -----------------------------------
            Soln = {}
            RemainingBudget = ksum.KahanRunningSum(B)
            for _, Idx in Values:
                if W[Idx] == 0:
                    Soln[Idx] = C[Idx]
                else:
                    if RemainingBudget.Sum <= 0:
                        break
                    ToTake = min(RemainingBudget/W[Idx], C[Idx])
                    Soln[Idx] = ToTake
                    RemainingBudget -= ToTake*W[Idx]
            return Soln

        # Problem Digest------------------------------------------------------------------------------------------------
        C = self._C.copy()
        AlreadyDecidedSoln = self._PartialSoln.copy()
        for K, V in AlreadyDecidedSoln.items():
            # Take partial solution into account.
            C[K] -= V
        B = ksum.kahan_sum([(-self._W[K]*V) for K, V in AlreadyDecidedSoln.items()] + [self._B])
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
        self._ObjVal = ksum.kahan_sum(self._P[K]*V for K, V in AlreadyDecidedSoln.items())
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
        # define preliminary stuff -------------------------------------------------------------------------------------
        Indent = len(self._BranchingIdentifier) * "  "
        Soln, ObjVal, FracIdx = self.greedy_solve()
        # To Return:----------------------------------------------------------------------------------------------------
        NewObjVal, SubP1, SubP2, = globalIntegralValue, None, None
        IsIntegral = FracIdx == -1
        OptimalitySatisfied = ObjVal > globalIntegralValue
        # Pruned by optimality of the integral solution ----------------------------------------------------------------
        if IsIntegral and OptimalitySatisfied:
            NewSoln, NewObjVal = Soln, ObjVal
            EknapsackGreedy.log(Indent + "[!] This Node has found an integral solution and updated the optimal")
            return NewSoln, NewObjVal, SubP1, SubP2
        # Fractional, and it should branch -----------------------------------------------------------------------------
        if OptimalitySatisfied:
            EknapsackGreedy.log(Indent + "[?] Heuristic points to increasing objective, this node branches.")
            # p1, Bound from above -------------------------------------------------------------------------------------
            PartialSoln = self.PartialSoln
            NewItemCounts = self._C
            NewIndices = self.Indices
            if int(Soln[FracIdx]) != 0:
                NewItemCounts = NewItemCounts.copy()
                NewItemCounts[FracIdx]= int(Soln[FracIdx])
            else:
                NewIndices.remove(FracIdx)
            SubP1 = EknapsackGreedy(self._P,
                                    self._W,
                                    NewItemCounts,
                                    self._B,
                                    branchingIdentifier=self._BranchingIdentifier + "0")
            SubP1._Indices = NewIndices
            SubP1._PartialSoln = PartialSoln
            # p2, Bound from below -------------------------------------------------------------------------------------
            PartialSoln = self.PartialSoln.copy()
            NewIndices = self.Indices
            PartialSoln[FracIdx] = int(Soln[FracIdx]) + 1
            SubP2 = EknapsackGreedy(self._P,
                                    self._W,
                                    self._C,
                                    self._B,
                                    branchingIdentifier=self._BranchingIdentifier + "1")
            SubP2._Indices = NewIndices
            SubP2.PartialSoln = PartialSoln
        else:
            EknapsackGreedy.log(f"{Indent}[~] Sub-Optimal; Branching is pruned. "
                                if ObjVal != float("-inf") else f"{Indent}[*] Pruned by Infeasibility." )

        return NewSoln, NewObjVal, SubP1, SubP2

    def solve(self, verbose=False):
        EknapsackGreedy.Verbose = verbose
        Soln, ObjVal = EknapsackGreedy.BB(self)
        return Soln, ObjVal

    def __repr__(self):
        Indent = len(self._BranchingIdentifier)*"  "
        s = "\n" + Indent + "-"*20 + "\n"
        s += Indent + "EKnapSack Instance: \n"
        s += Indent + f"Size: {self.Size}\n"
        s += Indent + f"Variables Aren't fixed to zero: {self.Indices}\n"
        s += Indent + f"Variable Lower Bound: {self.PartialSoln}\n"
        s += Indent + f"Variable Uppwer Bound: {self._C}\n"
        s += Indent + f"Greedy Soluion: {self.__GreedySoln}\n"
        s += Indent + f"Upperbound (Objective Value from Greedy algo): {self._ObjVal}\n"
        s += Indent + f"Branching Identifier: {self._BranchingIdentifier}\n"
        return s

    def __getitem__(self, I):
        """
            Quick tuple indexing to get desirable information regarding a certain item.
        :param item:
        :return:
        """
        return [self._P, self._W, self._C][I[0]][I[1]]

    def check_condition(self):
        """
            Attempt to access the numerical stability of the problem.

            * Get the simplex tableau of this thing and see the condition number of the matrix.
            * The matrix is the constraints matrix of the LP problem.
        :return:
            cond numb
        """
        ConstraintMatrix = np.eye(self.Size, self.Size, k = 0)  # n by n identity matrix -------------------------------
        ConstraintMatrix = np.vstack((np.array(self._W), ConstraintMatrix))
        ConstraintMatrix = np.hstack(
            (
                ConstraintMatrix,
                np.array([self._B] + self._C)[:, np.newaxis]
            )
        )

        return np.linalg.cond(ConstraintMatrix), ConstraintMatrix  # Condition number of the constraint matrix ---------



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

    @property
    def BranchingIdentifier(self):
        return self._BranchingIdentifier

    @BranchingIdentifier.setter
    def BranchingIdentigier(self, value):
        self._BranchingIdentifier = value

    @staticmethod
    def BB(rootProblem):
        """
            A static method for evaluation the whole Eknapsack problem
        :param rootProblem:
            An instance of an root problem, representing the initial problem for the initial knapsack.
        :return:
            optimal solution.
        """
        Class = EknapsackGreedy
        def Initialization():
            S, _, _ = rootProblem.greedy_solve()
            S = dict([(I, V) for I, V in S.items() if int(V) == V])
            ObjVal = ksum.kahan_sum(rootProblem[0, I]*V for I, V in S.items())
            Class.log(f"BB executing with warm start solution and objective value: ")
            Class.log(S)
            Class.log(ObjVal)
            return S, ObjVal
        GIntSoln, GObjVal = Initialization()
        Stack = [rootProblem]
        while len(Stack) != 0:
            P = Stack.pop()
            P.greedy_solve()
            Class.log(P)
            GIntSoln, GObjVal, SubP1, SubP2 = P.branch(GObjVal, GIntSoln)
            if SubP1 is not None:
                Stack.append(SubP1)
            if SubP2 is not None:
                Stack.append(SubP2)
        return GIntSoln, GObjVal

    @classmethod
    def log(cls, msg):
        if cls.Verbose:
            print(msg)


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
        Problem.solve(lp.PULP_CBC_CMD(msg = False))
        for I, Var in self._X.items():
            if Var.varValue != 0:
                Soln[I] = Var.varValue
        return Soln, ksum.kahan_sum(V*self._P[I] for I, V in Soln.items())

    @property
    def LpProblem(self):
        if self._LP is None:
            self.formulate_lp()
        return self._LP


def main():

    def TestKnapSack():
        P, W, B, C = [2, 3, 2, 1], [6, 7, 4, 1], 9, [1]*4
        EKnapSack = EknapsackGreedy(P, W, C, B)

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
        EKnapSack = EknapsackGreedy(P, W, C, B)
        print(EKnapSack)
        _, _, Sub1, Sub2 = EKnapSack.branch(float("-inf"))
        print(f"Sub1 {Sub1}")
        print(f"Sub2 {Sub2}")

    def RunBB():
        P, W, B, C = [2, 3, 2, 1], [6, 7, 4, 1], 9, [1] * 4
        EKnapSack = EknapsackGreedy(P, W, C, B)
        print(EknapsackGreedy.BB(EKnapSack))
        print("="*20)
        P, W, B, C = [2, 3, 2, 1], [6, 7, 4, 1], 9, [1, 1, 1, 2]
        EKnapSack = EknapsackGreedy(P, W, C, B)
        print(EknapsackGreedy.BB(EKnapSack))
        print("="*20)
        P, W, B, C = [1, 2, 3, 4], [4, 3, 2, 1], 8, [4, 2, 1, 1]
        EKnapSack = EknapsackGreedy(P, W, C, B)
        print(EknapsackGreedy.BB(EKnapSack))
        P, W, B, C = [1, 2, 3, 4], [4, 3, 2, 1], 8, [4, 2, 1, 4]
        EKnapSack = EknapsackGreedy(P, W, C, B)
        print(EknapsackGreedy.BB(EKnapSack))

    def LPFormulation():
        P, W, B, C = [2, 3, 2, 1], [6, 7, 4, 1], 9, [1] * 4
        EKnapSack = EknapsackSimplex(P, W, C, B)
        print(EKnapSack.formulate_lp())
        print(EKnapSack.solve())

        P, W, B, C = [2, 3, 2, 1], [6, 7, 4, 1], 9, [1] * 4
        EKnapSack = EknapsackGreedy(P, W, C, B)
        print(EknapsackGreedy.BB(EKnapSack))


    def CheckAgainstPulp():
        import time
        FailedTests = 0
        TotalTests = 500
        ConditionsNumberPassed, ConditionNumberFailed = [], []
        for _ in range(TotalTests):

            PWC, B = make_extended_knapsack_problem(4, 0.3)
            P, W, C = map(list, zip(*PWC))

            KnapsackInstance1 = EknapsackGreedy(P, W, C, B)
            KnapsackInstance2 = EknapsackSimplex(P, W, C, B)
            # ----------------------------------------------------------------------------------------------------------
            try:
                Soln1, Obj1 = KnapsackInstance1.solve()
                Soln2, Obj2= KnapsackInstance2.solve()
                # print(Soln1, Soln2)

                if abs(Obj1 - Obj2) > 1e-14:
                    FailedTests += 1
                    print(f"Failed on inputs: {PWC, B}, \n obj is like: {Obj1, Obj2}")
                    print(f"Solutions are like: {Soln1, Soln2}")
                    print(f"condition number of the constraint matrix is: {KnapsackInstance1.check_condition()[0]}")
                    ConditionNumberFailed.append(KnapsackInstance1.check_condition()[0])
                else:
                    print(P, W, C, B, " : passed")
                    ConditionsNumberPassed.append(KnapsackInstance1.check_condition()[0])
            except lp.apis.core.PulpSolverError:
                print(f"Pulp Solver error")
                print(f"Constraint matrix condition: {KnapsackInstance1.check_condition()[0]}")
                ConditionNumberFailed.append(KnapsackInstance1.check_condition()[0])

        print(f"Failed: {FailedTests}")
        print(f"Failed Tests condition numbers: ")
        print(ConditionNumberFailed)
        print(f"Passed Tests condition numbers: ")
        print(ConditionsNumberPassed)


    def UnstableInstance():
        PWC, B = ([(0.79, 0.38, 2), (0.35, 0.19, 4), (0.23, 0.01, 5), (0.89, 0.6, 4), (0.9, 0.63, 1)], 1.38)
        P, W, C = map(list, zip(*PWC))
        KnapscakInstance = EknapsackGreedy(P, W, C, B)
        return KnapscakInstance

    def InvestigateNumericalInstability():
        KnapsackInstance = UnstableInstance()
        KnapsackInstance.solve(verbose=True)
        pass


    # RunBB()
    # LPFormulation()
    CheckAgainstPulp()
    InvestigateNumericalInstability()


if __name__ == "__main__":
    main()