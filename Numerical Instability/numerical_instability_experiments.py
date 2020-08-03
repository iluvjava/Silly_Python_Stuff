"""

    Let's invesitage this via experiments and see how big of an issue it actually is.

    link: Python Floating point limitations
    https://docs.python.org/3/tutorial/floatingpoint.html

    link: Khan Summation algorithm
    https://en.wikipedia.org/wiki/Kahan_summation_algorithm
"""
from typing import *
import fractions as frac
import numpy as np
import statistics as stats


def rand_list(size) -> List[float]:
    import random as rand
    return [(rand.random()*2e8 - 1e8)for _ in range(size)]


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


def numpy_sum(theList: List[float]) -> float:
    return np.sum(theList)

def main():

    def GetListofErrorsFor(sum1:callable, trials:int=1000, listSize:int=20):
        print("Benchmarking... ")
        Errors = []
        for TheList in [rand_list(listSize) for _ in range(trials)]:
            S2 = sum1(TheList)
            S3 = rational_sum(TheList)
            Errors.append(abs(S2 - S3))
        return Errors

    print(f"Bench marking Python Sum against Rational Sum: ")

    print(GetListofErrorsFor(python_sum, trials=100, listSize=1000))

    print(f"Bench Marking Khan-Sum with Rational Sum: ")

    print(GetListofErrorsFor(khan_sum, trials=100, listSize=1000))

    print(GetListofErrorsFor(numpy_sum, trials=100, listSize=1000))


if __name__ == "__main__":
    main()