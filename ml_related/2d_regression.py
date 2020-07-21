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
figure(num=None, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
from typing import *
NpArray = Type[np.array]
from random import random as sysrnd

# Things for Ordinary Polynomial fitting:
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
# End


def generate_random_poly(queryPoints: NpArray, epsilon, **kwargs):
    """

    :param queryPoints:
    :param epsilon:
    :param kwargs:
        roots:
            given the roots of the polynomial as a NParray.
        fxn:
            manually provides a function that takes in one
            argument "x" and return the number as the output for the function.
    :return:
    """
    def EvalAt(x):
        Terms = np.full(len(roots), x) - roots
        Res = Terms[0]
        for T in Terms[1:]:
            Res *= T
        return Res
    roots = None
    # Set up the things:
    for K, V in kwargs.items():
        if K is "roots":
            assert type(V).__module__ == np.__name__ # The type comes from numpy module
            roots = V
        if K is "fxn":
            EvalAt = V

    Res = np.zeros(len(queryPoints))
    Noise = epsilon*np.random.randn(len(queryPoints))
    for I, V in enumerate(queryPoints):
        Res[I] = EvalAt(V) + Noise[I]
    return Res


def ordinary_regression_fit_2d(xPoints:NpArray, yPoints:NpArray, polyOrder:int):
    """
        Try to fit a set of (x_i, y_i) data points with a certain degree of
        for the polynomial model.

        * Measure of quality: MSE, SS_exp
    :param xPoints:
        Row Vector!
    :param yPoints:
        Row Vector!
    :param polyOrder:
        The order of the polynomial to fit the curve.
    :return:
    """
    assert len(xPoints) == len(yPoints), "X, Y dim mismatches"
    poly = PolynomialFeatures(degree=polyOrder)
    Vandermonde = poly.fit_transform(xPoints[:, np.newaxis]) # Get the Vandermonde matrix from sklearn
    LinModel = linear_model.LinearRegression()
    LinModel.fit(Vandermonde, yPoints)
    return LinModel


def regression_MSE_compute(linModel, predictors:NpArray, expectedPredictants: NpArray):
    """

    :param linModel:
    :param predictors:
    :param expectedPredictants:
    :return:
    """
    PredictedByModel = linear_model.predict(predictors)
    ActualY = expectedPredictants
    n = len(predictors) - len(linModel.coef_)
    Res = sum((PredictedByModel - ActualY)**2)
    Res /= n
    return Res


def rand_split_idx(l: int, ratio):
    """
        Get all the indices for the training data set.
    :param l:
        The size of the data set.
    :param ratio:
        The ration for the training data set.
    :return:
        All the indices that should be the tranning set.
    """
    assert ratio < 1 and ratio > 0, "Ratio between 1, 0"
    return [I for I in range(l) if sysrnd() < ratio]


class TrainOrdinary2DRegression:
    """
        Train a model, get the MSE, and see how the validation errors changes with DF of the model.
    """

    def __init__(self, xPoints:NpArray, yPoints:NpArray):
        """
        :param xPoints:
            1d Row VECTOR!
        :param yPoints:
            1d ROW VECTOR!
        """
        self._X, self._Y = np.copy(xPoints), np.copy(yPoints)

    def get_MSE_for(self, indices):
        """
            Get the MSE for a certain set of indices marking the training data set.
        :param indices:
            This is the indices marking the training data set.
        :return:
            The MSE for it.
        """


    @property
    def XDataPoints(self):
        return self._X

    @property
    def YdataPoints(self):
        return self._Y




def main():
    XGrindPoints = np.linspace(0, 10, 100)
    Roots = np.linspace(0, 9, 3)
    YScatteredPoints = generate_random_poly(XGrindPoints, roots = Roots, epsilon=10)
    pyplt.scatter(XGrindPoints, YScatteredPoints)
    pyplt.show()

    LinModel = ordinary_regression_fit_2d(XGrindPoints, YScatteredPoints, 3)
    print(f"The df of the model is: {len(LinModel.coef_)}")



if __name__ == "__main__":
    main()