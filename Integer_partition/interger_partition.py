"""
    Euler solved the integer partition, let's see
    if I can find the recurrences too, and summarize it in python.
"""

from typing import Dict, Tuple


def soln(a: int, b: int, R: Dict[Tuple, int] = {}):
    """
        This is a recurrence solution for the integer partition problem.
        The dict R stores all the values need for the recurrences.
        * Definition of R[a, b]
            - a <= b : This must be true.
            - R[a, b] denotes the total number of partitions of a with chunk size no more than b.
                -- e.g. R[2,2] is asking how many way 2 can be summed up with a list of sorted integers, and each
                    integers must be less than or equal to 2.

    :param a:
        integer.
    :param b:
        integer.
    :param R:
        The map for the recurrences.
    :return:
        The value of R[a, b]
    """
    # already there.
    if (a, b) in R.keys():
        return R[a, b]
    # base case.
    if b == 0 or b == 1:
        R[a, b] = 1
        return R[a, b]
    R[a, b] = sum(soln(a - c, min(a - c, c), R) for c in range(1, b + 1))
    return R[a, b]



def main():
    print(f"partition of 0: {soln(0, 0)}")
    print(f"partition of 3: {soln(3, 3)}")
    print(f"partition of 4: {soln(4, 4)}")
    print(f"partition of 5: {soln(5, 5)}")
    correct = [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77, 101, 135, 176, 231, 297, 385, 490, 627, 792, 1002, 1255,
               1575, 1958, 2436, 3010, 3718, 4565, 5604, 6842, 8349, 10143, 12310, 14883, 17977, 21637, 26015, 31185,
               37338, 44583, 53174, 63261, 75175, 89134, 105558, 124754, 147273, 173525]
    for I, J in enumerate(correct):
        S = soln(I, I)
        assert S == J, f"Partition for {I} is {S} which is not correct. "

    print(f"Test Passed, for the first {len(correct)} integers.")


if __name__ == "__main__":
    main()