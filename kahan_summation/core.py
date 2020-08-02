"""

    This is provably better than the summing algorithm implemented by python.
"""


__all__ = ["kahan_sum", "KahanRunningSum"]

from typing import *


def rand_list(size) -> List[float]:
    import random as rand
    return [round(rand.random()*20, 2) for _ in range(size)]


def rational_sum(theList: List[float]) -> float:
    import fractions as frac
    return float(sum(map(lambda x: frac.Fraction(x), theList)))


def kahan_sum(theList: List[float]) -> float:
    Sum, Compensator= 0, 0
    for I in theList:
        T = Sum + I
        if abs(Sum) >= abs(I):
            Compensator += (Sum - T) + I  # Sum - T exposes the numerical error. ---------------------------------------
        else:
            Compensator += (I - T) + Sum  # If summees are on the similar scale, this rarely happens.  -----------------
        Sum = T
    return Sum + Compensator


class KahanRunningSum:
    """
        Summing up floating points without the loos of precision.
    """

    def __init__(self, initialSum = 0):
        self.__Sum = initialSum
        self.__Compensator = 0

    @property
    def Sum(self):
        return round(self.__Sum + self.__Compensator, 15)

    def __iadd__(self, other):
        """
            Add a number to the sum.
        :param other:
            Float, ints, or whatever.
        :return:
        """
        T = self.__Sum + other
        if abs(self.__Sum) >= abs(other):
            self.__Compensator += (self.__Sum - T) + other
        else:
            self.__Compensator += (other - T) + self.__Sum
        self.__Sum = T
        return self

    def __isub__(self, other):
        self += -other
        return self

    def __mul__(self, other):
        return self.Sum * other

    def __truediv__(self, other):
        return self.Sum / other

    def __eq__(self, other):
        return self.Sum - other == 1e-16

    def __lt__(self, other):
        return self.Sum < other

    def __gt__(self, other):
        return self.Sum > other

    def __le__(self, other):
        return self.Sum <= other

    def __ge__ (self, other):
        return self.Sum >= other

    def __ne__ (self, other):
        return self.Sum - other != 1e-16




def main():
    RandSum = rand_list(99999)
    Sum1 = rational_sum(RandSum)
    KSum = KahanRunningSum()
    for S in RandSum:
        KSum += S
    print(KSum.Sum, Sum1)

if __name__ == "__main__":
    main()