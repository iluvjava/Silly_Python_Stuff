#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600


def poly_val_fsum(a, alpha):
    from math import fsum
    Xpow = 1
    ToSum = []
    for C in reversed(a):
        ToSum.append(C * Xpow)
        Xpow *= alpha
    ToSum.sort()
    return fsum(ToSum)


def get_row(r:int, j=None):
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


def poly_numpy(a, alpha):
    Xpow = 1
    ToSum = []
    for C in reversed(a):
        ToSum.append(C * Xpow)
        Xpow *= alpha
    return np.sum(ToSum)


def poly_val_numpy_poly(a, alapha):
    return np.polyval(a, alapha)


def poly_val_exp(a, alpha):
    import math as m
    Xpow = 1
    ToSum = []
    for C in reversed(a):
        ToSum.append(C * Xpow)
        Xpow *= alpha
    ToMul = []
    for S in ToSum:
        ToMul.append(m.e**S)
    Product = 1
    for S in ToMul:
        Product *= S
    return m.log(Product)


if __name__ == "__main__":
    def main():
        def f(x):
            return (x + 1)**40
        Coeffs = get_row(40)

        Xs = np.arange(-2, 0, 1e-3)
        print(poly_numpy(Coeffs, -0.9))

        Ys1 = [np.log(abs(poly_numpy(Coeffs, X)) - f(X)) for X in Xs]
        Ys2 = [np.log(abs(poly_val_fsum(Coeffs, X) - f(X))) for X in Xs]
        # Ys3 = [np.log(abs(poly_val_exp(Coeffs, X) - f(X))) for X in Xs]
        plt.scatter(Xs, Ys1, s=1, label="Numpy Polyval")
        plt.scatter(Xs, Ys2, color="red", s=1, alpha=0.2, label="Simple Fsum")
        # plt.scatter(Xs, Ys3, color="green", s=1, alpha=0.2, label="Exp Sum")
        plt.legend()
        plt.ylabel("log(Error)")
        plt.xlabel("x")

        plt.show()
    main()
    print(poly_val_exp([1, 2, 1], 2))





