"""
    This file explores Faray's sequence and it's applications.
"""
import fractions as frac

def faray_sequence(N):
    F0 = [(0, 1), (1,1)]
    for _ in range(N):
        FNext = []
        for I in range(len(F0) - 1):
            FNext.append(F0[I])
            if F0[I][1] + F0[I+1][1] <= N:
                FNext.append((F0[I][0] + F0[I+1][0], F0[I][1] + F0[I+1][1]))
        FNext.append(F0[-1])
        F0 = FNext
    return FNext


def print_farey_sequence(n: int, descending: bool = False) -> None:
    """
        Print the n'th Farey sequence. Allow for either ascending or descending.
        Copied from wikipedia.
    """
    (a, b, c, d) = (0, 1, 1, n)
    if descending:
        (a, c) = (1, n - 1)
    print("{0}/{1}".format(a,b))
    while (c <= n and not descending) or (a > 0 and descending):
        k = (n + b) // d
        (a, b, c, d) = (c, d, k * c - a, k * d - b)
        print("{0}/{1}".format(a,b))


def faray_approx(toApprox: float):
    """
        This function is not for any serious usage, it's probably
        the same implementations as what is inside of fractions.Fraction class, so...
        yeah.....
    :param toApprox:
        A float
    :return:
        A fraction that is closest to that floating point.
    """
    B = [(0, 1), (1, 2), (1, 1)]
    if toApprox == 0 or toApprox == 1:
        return frac.Fraction(1, 1) if toApprox == 1 else frac.Fraction(0, 1)
    while abs(frac.Fraction(*B[1]) - toApprox) > 1e-14:
        MiddleValue = frac.Fraction(*B[1])
        if toApprox > MiddleValue:
            NewMiddleValue = (B[1][0] + B[2][0], B[1][0] + B[2][1])
            B = [B[1], NewMiddleValue, B[2]]
        else:
            NewMiddleValue = (B[0][0] + B[1][0], B[0][1] + B[1][1])
            B = [B[0], NewMiddleValue, B[1]]
    return frac.Fraction(*B[1])


def main():
    print(print_farey_sequence(5))
    print(faray_sequence(5))
    print(faray_approx(0.19))

if __name__ == "__main__":
    main()