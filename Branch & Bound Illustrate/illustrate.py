# This is a print of all the procedures in the Branch and Bound algorithm
# The pulp module is needed for it to run.

from pulp import *

def mane():
    P = LpProblem(name="P", sense=LpMaximize)
    x1 = LpVariable("x1", lowBound=0)
    x2 = LpVariable("x2", 1 / 2)
    P += 3 * x1 + x2
    P += -2 * x1 + 3 * x2 <= 6
    P += 10 * x1 + 4 * x2 <= 27

    print_BB(P)


def print_BB(P):
    """
        This is only for maximization problem
    :param P:
        The Lp Problem.
    :return:
        none
    """
    def solve_print_p(p):
        """ Print and returns the objective value"""

        if (p.get_lp() == 1):
            print(f"{p.name} is solved: ")
            for I, V in enumerate(p.variables()):
                print(f"x{I + 1} = {V.varValue}")
            p.roundSolution()
            print(f"c^Tx^* = {value(p.objective)}")
            return value(p.objective)

        else:
            print(f"{p.name} cannot be solved.")
            return None

    def soln_Is_Integral(p):
        for V in p.variables():
            if int(V.varValue) != V.varValue:
                return False
        return True

    def get_branching_variable(p):
        for V in p.variables():
            if int(V.varValue) != V.varValue:
                print(f"Choose {V} to branch")
                return V
        raise AssertionError


    P_star = None
    stack = [P]
    IterationCount = 0
    while len(stack) != 0:
        IterationCount += 1
        print(f"Iteration: {IterationCount}---------------------------------------------------------------------------")
        P_ = stack.pop()
        ObjVal = solve_print_p(P_)
        if (ObjVal is None):
            print(f"{P_.name} Unsolvable")
            continue # Infeasible, unsolvable
        if ((P_star is not None) and (value(P_star.objective) >= ObjVal)):
            print(f"{P_.name} Bounded by optimal, pruned")
            continue # Bounded by optimal, pruned.
        if ((P_star is None) or ObjVal > value(P_star)) and (soln_Is_Integral(P_)):
            P_star = P_
            print(f"{P_.name} Is integral and set as new optimal")
            continue
        # branch on current polytope.
        xi = get_branching_variable(P_)
        P_1 = P_.copy(); P_1.name = P_.name + "1"; P_1 += xi <= int(xi.value())
        P_2 = P_.copy(); P_2.name = P_.name + "2"; P_2 += xi <= int(xi.value())
        stack.append(P_1)
        stack.append(P_2)


    print(f"\n--------------The Optimal Polytope is: {P_star}---------------------")
    for I,V in enumerate(P_star.variables()):
        print(f"x{I + 1} = {V.value()}")
    print(f"objective: {value(P_star.objective)}")


if __name__ == "__main__":
    mane()