"""
    Let's focuses on the learning of 2D regression.
    In this file, we are focusing on using the sklearn.
"""
import numpy as np
from matplotlib import pyplot as pyplt

def generate_random_poly(queryPoints, roots, epsilon):
    def EvalAt(x):
        Terms = np.full((1, len(roots), x)) - np.array(roots)
        Res = Terms[0]
        for T in Terms[1:]:
            Res *= T
        return Res
    Res = np.zeros(len(queryPoints))
    Noise = epsilon*np.random.randn(len(queryPoints))
    if type(queryPoints) is not np.ndarray:
        queryPoints = np.array(queryPoints)
    for I, V in enumerate(queryPoints):
        Res[I] = EvalAt(V) + Noise[I]
    return Res


def main():
    XGrindPoints = np.linspace(0, 10, 100)
    Roots = np.linspace(0, 9, 3)
    YScatteredPoints = generate_random_poly(XGrindPoints, Roots, 0.1)
    pyplt(XGrindPoints, YScatteredPoints)
    pyplt.show()
    pass

if __name__ == "__main__":
    main()