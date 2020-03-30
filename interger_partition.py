"""
    Euler solved the integer partition, let's see
    if I can find the recurrences too, and summarize it in python.
"""

from types import *

def count_partition(N: int, R: Dict[Tuple, int] = None):
    assert N >= 1, "N must be > 1"
    # Tuple maps to count.
    R = {(1, 1): 1} if R is None else R
    for I in range(2, N + 1):
        Sum = 1
        for J in range(1, I):
            if (J - I) >= J:
                Sum += R[J, J]
            else:
                Sum += R[J, J - I]
        R[I, I] = Sum

    return R[N, N]


def soln(A: int, B: int, R: Dict[Tuple, int] = None):
    R = {(1, 1): 1} if R is none else R



def main():
    print(f"partition of 3: {count_partition(3)}")
    pass


if __name__ == "__main__":
    main()