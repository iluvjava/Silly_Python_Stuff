"""
    You about to witness some serious shit when you run this file. Some very high energy contestant is
    in our scene today.
"""
from knapsack import extended_napsack_core3 as ks


def get_problem_list(problemCount: int, problemSize: int, density: float, countsUpperBound = None, significance= 8):
    """
        Generate a set of parameters for the extended-knapsack problem.
    :param problemSize:
        number of items in the problem.
    :param density:
        ratio of the budget and the total weights of all items in the problem.
    :return:
        a list of problems, in this format:
        [
            [[], [], [], b],
            [[], [], [], b]
            ...
        ]
    """
    def ExpandProblem(P):
        Pack= list(map(list, zip(*P[0])))
        Pack.append(P[1])
        return Pack
    return [ExpandProblem(ks.make_extended_knapsack_problem(problemSize, density, countsUpperBound, significance))
            for _ in range(problemCount)]


def time_it_for(instance: callable):
    import time as t
    Start = t.time()
    Result= instance()  # No parameters here
    return t.time() - Start, Result


def benchmark_solver_for(solver, problems):
    Time, Objective = [], []
    for P in problems:
        def Runner():
            _, Obj = solver(P[0], P[1], P[2], P[3])
            return Obj
        Elapsed, Obj = time_it_for(Runner)
        Time.append(Elapsed)
        Objective.append(Obj)
    return Time, Objective


def main():

    def CompareSolversForProblemSize():
        problemSize = 80
        ProblemList = get_problem_list(30, problemSize, 0.2, 3)

        def PulpSolver(p, w, c, b):
            SolverInstance = ks.EknapsackSimplex(p, w, c, b)
            return SolverInstance.solve()

        def GreedyBBSolver(p, w, c, b):
            SolverInstance = ks.EknapsackGreedy(p, w, c, b)
            return SolverInstance.solve()

        ExecutionTimePulp, ObjectivePulp = benchmark_solver_for(PulpSolver, ProblemList)
        ExecutionTimeGreed, ObjectiveGreed = benchmark_solver_for(GreedyBBSolver, ProblemList)

        print(ExecutionTimePulp)
        print(ExecutionTimeGreed)
        print(ObjectivePulp)
        print(ObjectiveGreed)

        JsonData = {}
        JsonData["Problem_size"] = problemSize
        JsonData["PulpSolver"] = {}
        JsonData["GreedyBBSolver"] = {}
        JsonData["PulpSolver"]["Execution_Time"] = ExecutionTimePulp
        JsonData["PulpSolver"]["Objective_value"] = ObjectivePulp
        JsonData["GreedyBBSolver"]["Execution_Time"] = ExecutionTimeGreed
        JsonData["GreedyBBSolver"]["Objective_value"] = ObjectiveGreed
        from quick_json import quick_json as qj
        qj.json_encode(obj=JsonData, filename=f"Extended_knapsack_benchmark_results_problemsize{problemSize}.json")






if __name__ == "__main__":
    main()