"""
    You about to witness some serious shit when you run this file. Some very high energy contestant is
    in our scene today.
"""
from knapsack import extended_napsack_core3 as ks

def get_problem_list(problemCount: int, problemSize: int, density: float, countsUpperBound = None):
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

    return [ExpandProblem(ks.make_extended_knapsack_problem(problemSize, density, countsUpperBound))
            for _ in range(problemCount)]


def main():
    print(get_problem_list(3, 10, 0.2, 10))
    pass


if __name__ == "__main__":
    main()