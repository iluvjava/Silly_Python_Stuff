"""
    This class is going to provide fast and accuarate evaluations of polynomials and
    relevant functionalities.
"""

def nested_multiplications(a, alpha):
    """

    :param a:
        List of coefficients of polynomial, descending.
    :param alpha:
        p(alpha), the value we want to evaluate the polynomial at.
    :return:
        p(alpha)
    """
    B = [a[0]]
    I = 1
    L = len(a)
    while I < L:
        B.append(alpha * B[I - 1] + a[I])
        I += 1
    return B[-1]


def poly_val(a, alpha):
    """

    :param a:
        coefficient, descending order.
    :param alpha:
    :return:
    """
    from kahan_summation import core2
    Ksum = core2.KahanRunningSumMutable
    B = Ksum(a[0])
    I = 1
    L = len(a)
    while I < L:
        B.multiply(alpha)
        B.add(a[I])
        I += 1
    return B.Sum


def poly_val2(a, alpha):
    """

    :param a:
        polynomial coefficients in descending order.
    :param alpha:
    :return:
    """
    from kahan_summation import core2
    Ksum = core2.KahanRunningSumMutable(0)
    Xpow = 1
    for C in reversed(a):
        Ksum.add(C*Xpow)
        Xpow *= alpha
    return Ksum.Sum

def poly_val3(a, alpha):
    from math import fsum
    Xpow = 1
    ToSum = []
    for C in reversed(a):
        ToSum.append(C * Xpow)
        Xpow *= alpha
    return fsum(ToSum)

def poly_val4(a, alpha):
    import numpy as np
    Xpow = 1
    ToSum = []
    for C in reversed(a):
        ToSum.append(C * Xpow)
        Xpow *= alpha

    return np.sum(ToSum)

def get_row(r:int, j = None ):
    """
        Compute the jth element on the r th row.
    :param r:
        The row we are looking at.
    :return:
    """
    row = [None]*(r + 1)
    row[0] = row[-1] = 1
    for i in range(1, r//2 + 1):
        product = (row[i - 1]*(r - (i - 1)))//i
        row[i] = row[-i - 1] = product
    return row if j is None else row[j]


def main():
    power = 30
    a = get_row(power)
    x = -0.9
    print(a)
    Val = poly_val(a, x)
    ValCompare = nested_multiplications(a, x)
    print(Val)
    print(ValCompare)
    print(poly_val2(a, x))
    print(poly_val3(a, x))
    print(poly_val4(a, x))
    print((1 + x)**power)




if __name__ == "__main__":
    main()