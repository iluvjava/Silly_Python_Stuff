"""
    Let's focuses on the learning of 2D regression.
    In this file, we are focusing on using the sklearn.

    * One of the thing in Machine Fitting is the use of regularization.
        - By including the norm of the regression coefficients itself into the
        optimization problem we are able to fit the data and at the same time controls the
        complexity of the model.

"""
import numpy as np
from matplotlib import pyplot as pyplt
from matplotlib.pyplot import figure
figure(num=None, figsize=(4, 3), dpi=150, facecolor='w', edgecolor='k')
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
from typing import *
NpArray = Type[np.array]


def generate_random_poly(queryPoints: NpArray, roots:NpArray, epsilon):
    def EvalAt(x):
        Terms = np.full(len(roots), x) - roots
        Res = Terms[0]
        for T in Terms[1:]:
            Res *= T
        return Res
    Res = np.zeros(len(queryPoints))
    Noise = epsilon*np.random.randn(len(queryPoints))
    for I, V in enumerate(queryPoints):
        Res[I] = EvalAt(V) + Noise[I]
    return Res

class OrdinaryRegression:
    pass

def main():
    XGrindPoints = np.linspace(0, 10, 100)
    Roots = np.linspace(0, 9, 3)
    YScatteredPoints = generate_random_poly(XGrindPoints, Roots, 10)
    pyplt.scatter(XGrindPoints, YScatteredPoints)
    pyplt.show()

if __name__ == "__main__":
    main()