"""

    Let's invesitage this via experiments and see how big of an issue it actually is.

    link: Python Floating point limitations
    https://docs.python.org/3/tutorial/floatingpoint.html

    link: Khan Summation algorithm
    https://en.wikipedia.org/wiki/Kahan_summation_algorithm
"""
from typing import *
import fractions as frac


def rand_list(size) -> List[float]:
    import random as rand
    return [(rand.random()*2e10 - 1e10 )for _ in range(size)]


def naive_sum(theList:List[float]) -> float:
    Accumulator = 0
    for I in theList:
        Accumulator += I
    return Accumulator


def python_sum(theList:List[float]) -> float:
    return sum(theList)


def rational_sum(theList: List[float]) -> float:
    return float(sum(map(lambda x: frac.Fraction(x), theList)))


def khan_sum(theList: List[float]) -> float:
    Sum, Compensator= 0, 0
    for I in theList:
        T = Sum + I
        if abs(Sum) >= abs(I):
            Compensator += (Sum - T) + I  # Sum - T exposes the numerical error. ---------------------------------------
        else:
            Compensator += (I - T) + Sum  # If summees are on the similar scale, this rarely happens.  -----------------
        Sum = T
    return Sum + Compensator


def main():
    def BenchMark1(sum1, sum2, Trials = 1000, ListSize = 20):
        print("Benchmarking... ")
        Wrong = 0
        for TheList in [rand_list(ListSize) for _ in range(Trials)]:
            S2 = sum1(TheList)
            S3 = sum2(TheList)
            if S2 != S3:
                print(f"{S2}!={S3}")
                Wrong += 1
        print(f"% of trials they disagree: {Wrong/Trials}, the size of the list is: {ListSize}")

    print(f"Bench marking Python Sum against Rational Sum: ")

    BenchMark1(python_sum, rational_sum)

    print(f"Bench Marking Khan-Sum with Rational Sum: ")

    BenchMark1(khan_sum, rational_sum, Trials=100, ListSize=1000)


if __name__ == "__main__":
    main()