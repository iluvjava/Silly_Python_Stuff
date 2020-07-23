"""
    Let's focuses on the learning of 2D regression.
    In this file, we are focusing on using the sklearn.

    * One of the thing in Machine Fitting is the use of regularization.
        - By including the norm of the regression coefficients itself into the
        optimization problem we are able to fit the data and at the same time controls the
        complexity of the model.

    * This file highlighted the meta-training of regression models in general. Which is important for generalizing
    the process.

    * For sparse data point, parametric regression is the best to choose

    * So in some cases there can be colinearity, say: x1~x2, and x1, x2 are both predictors for y, this reduces the
    amount of information if it's used for predicting with a linear regression model.

    say x1 = a + b*x2, then we have y = w1*x1 + w2*x2 = w1*(a + b*x2) + w2*x2, which is just...
    = w1*a + (w1*b + w2)*x2, and the parameters for one of the predictor doesn't have to exist at all.

"""
import numpy as np
from matplotlib import pyplot as pyplt
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
from typing import *
NpArray = Type[np.array]
# Basic functions from the system:
from random import random as sysrnd
import math
import statistics as sysstat
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
        The order of the polynomial to fit the curve
    :return:
        The linear model, the Vandermonde matrix for the points you given.
    """
    assert len(xPoints) == len(yPoints), "X, Y dim mismatches"
    poly = PolynomialFeatures(degree=polyOrder)
    Vandermonde = poly.fit_transform(xPoints[:, np.newaxis]) # Get the Vandermonde matrix from sklearn
    LinModel = linear_model.LinearRegression()
    LinModel.fit(Vandermonde, yPoints)
    return LinModel, Vandermonde


def regression_MSE_compute(linModel, predictors:NpArray, expectedPredictants: NpArray, vandermonde = None):
    """

    :param linModel:
        The Trained Linear Model.
    :param predictors:
        A 1d NP, Array, ROW VECTOR
    :param expectedPredictants:
        A 1d NP array, ROW VECTOR
    :param vandermonde:
        This is a vandermone matrix, if you already have it as a type of from Polynomial features
        from previous usage, then feed it in and speed things up.
    :return:
        The total MSE of the model, the residuals of the model too.
    """
    Vandermonde = None
    if vandermonde is None:
        Vandermonde = PolynomialFeatures(degree=len(linModel.coef_)-1)
        Vandermonde = Vandermonde.fit_transform(predictors[:, np.newaxis])
    else:
        Vandermonde = vandermonde
    PredictedByModel = linModel.predict(Vandermonde)
    ActualY = expectedPredictants
    n = len(predictors)
    Residual = PredictedByModel - ActualY
    MSE = np.sum((Residual)**2)
    MSE /= n
    return MSE, Residual


def rand_split_idx(l: int, ratio=0.5):
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


def simple_min_search(f, a, b, tol):
    """
        Find the minimum of a univariable smooth function using ternary search.
    :param f:
        function.
    :param a:
        lower bound.
    :param b:
        upper bound.
    :return:
        argmin, min.
    """
    m1, m2 = (b-a)*(1/3) + a, a + (b-a)*(2/3)
    while abs(m1 - m2) > tol:
        if f(m1) <= f(m2):
            b = m2
        else:
            a = m1
        m1, m2 = (b - a) * (1 / 3) + a, a + (b - a) * (2 / 3)
    return (m1+m2)/2, f((m1+m2)/2)


def golden_section_search(f:callable, a, b, tol):
    """
        Perform a golden section search on the given single variable function.
    :param f:
        function, preferably unimodal.
    :param a:
        left boundary.
    :param b:
        right boundary
    :param tol:
        The tolerance you want.
    :return:
    """
    assert a < b and tol > 0, "Something wrong with the input parameters. "
    gr = (math.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(c - d) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    return (b + a) / 2, f((b+a)/2)


class MyRegression:

    def train_model_for(self, indices, tweakingParam):
        """
            The list of indices are all the indices that should be linked to a training data set.
            All other indices thare are not indices are assumed to be the validation set.
        :param indices:
            A list of indices. Vallina array is ok. s
        :param tweakingParam:
            A parameters that can be tweaked by the trainer instance to obtain the best inputs.
        :return:
            Linear model, a number that is related to the quality of training (Usually MSE)
        """


    def query(self, x:NpArray):
        """
            Query the trained model with a list of data points.
        :param x:
            A set of query points for the model.
        :return:
            A all the predicted values for the model for that inputs.
        """
        pass


class MultiVarLassoRegression:
    """
        This is a class that takes in a list of features set and then use
        the LassoRegression to fit the data.

        * A lasso regression should be good for handling data set that has potential colinearity for
        the predictors and stuff.
    """
    @staticmethod
    def collin_3d_data_points():
        """
            x1~x2 with colinearity.
        :return:
            A NP array, 2d.
        """
        pass


class MyLittleMyRegression(MyRegression):
    """
        Train a model, get the MSE, and see how the validation errors changes with DF of the model.
        It's only for 2d.
    """

    def __init__(self, xPoints:NpArray, yPoints:NpArray):
        """
        :param xPoints:
            1d Row VECTOR!
        :param yPoints:
            1d ROW VECTOR!
        """
        self._X, self._Y = np.copy(xPoints), np.copy(yPoints)
        self._LinModel = None

    def train_model_for(self, indices, polyOrder):
        """
            Get the MSE for a certain set of indices marking the training data set.
        :param indices:
            This is the indices marking the training data set.
        :return:
            The MSE for it.
        """
        assert sum(1 for I in indices if (I < 0 or I >= len(self.XDataPoints))) == 0, \
            f"Invalid Trainning Indices: {indices}, len of data: {len(self.XDataPoints)}"
        X_train, X_test = np.array([self.XDataPoints[I] for I in indices]), \
                          np.array([self.XDataPoints[I] for I in range(len(self.XDataPoints)) if I not in indices])
        Y_train, Y_test = np.array([self.YDataPoints[I] for I in indices]), \
                          np.array([self.YDataPoints[I] for I in range(len(self.YDataPoints)) if I not in indices])
        LinModel, V = ordinary_regression_fit_2d(X_train, Y_train, polyOrder=polyOrder)
        self._LinModel = LinModel
        MSE, _ =  regression_MSE_compute(LinModel, X_test, Y_test)
        return LinModel, MSE


    def query_model(self, x:NpArray):
        """
            Query the model with a list of points, and it will return the prediction.
        :param x:
            ROW VECTOR or villina Array.
        :return:
            Vanilla flavor array for the predicted data points.
        """

        Vandermonde = PolynomialFeatures(degree=len(self._LinModel.coef_) - 1)
        Vandermonde = Vandermonde.fit_transform(x[:,np.newaxis])
        Predicted = self._LinModel.predict(Vandermonde)
        return Predicted

    @property
    def XDataPoints(self):
        return self._X

    @property
    def YDataPoints(self):
        return self._Y

    @property
    def LinModel(self):
        return self._LinModel


class MyLittleRegressionTrainer:
    """
        This class will take in a series of data, and then automatically determine the
        correct polynomial degree for the 2d regression model.
    """
    def __init__(self, maxPolyOrder:int):
        assert maxPolyOrder < 20 and maxPolyOrder >= 1, "The maxpoly order is ridiculous. "
        self._MaxPolyOrder = maxPolyOrder

    def train_it_on(self, xData, yData, N=1):
        """
            Trains on a certain set of data and figure out the best degree for the
            polynomial for the data.
        :param xData:
            Row NParray vector.
        :param yData:
            Row Nparray vector.
        :param N:
            The number of test and train instances for the model.
        :return:
            min deg, min MSE, Instance of Mylittle Regression.
        """
        Test_Indices = [rand_split_idx(len(xData)) for I in range(N)]
        Regression = MyLittleMyRegression(xData, yData)
        def mse_error(polyDegree):
            MSE_List = [None]*N
            for I, IdxList in enumerate(Test_Indices):
                _, MSE = Regression.train_model_for(IdxList, int(polyDegree))
                MSE_List[I] = MSE
            return sysstat.mean(MSE_List)
        Argmin, min = golden_section_search(mse_error, 1, self._MaxPolyOrder, 1)
        return Argmin, min, Regression


def main():

    def test1():
        XGrindPoints = np.linspace(0, 10, 100)
        Roots = np.linspace(0, 9, 3)
        YScatteredPoints = generate_random_poly(XGrindPoints, roots = Roots, epsilon=10)
        pyplt.scatter(XGrindPoints, YScatteredPoints)
        pyplt.show()

    def test2():
        X = np.random.uniform(0, 10, 1000) # 100 points.
        Y = generate_random_poly(X, epsilon=10, roots=np.array([1, 3, 8]))
        pyplt.scatter(X, Y)

        MyRegression = MyLittleMyRegression(X, Y)
        LinModel, MSE = MyRegression.train_model_for(rand_split_idx(len(X)), 3)
        X_GridPoints = np.linspace(0, 10, 100)
        Y_GridPointsPredicted = MyRegression.query_model(X_GridPoints)
        print(f"Y_GridPointsPredicted = {Y_GridPointsPredicted}")
        pyplt.plot(X_GridPoints, Y_GridPointsPredicted, color="r")
        pyplt.show()
        print(f"The MSE for the trained modle is: {MSE} ")

    def test3():
        print(simple_min_search(lambda x: (x-1)**2, -10, 10, 1e-5))
        pass

    def test4(testPoints):
        X = np.random.uniform(0, 10, testPoints)
        Y = generate_random_poly(X, epsilon=10, roots=np.array([5, 8]))
        Trainer = MyLittleRegressionTrainer(10)
        Deg, MinMSE, LittleRegression = Trainer.train_it_on(X, Y, N=10)
        print(f"test4: deg = {Deg}")
        X_GridPoints = np.linspace(0, 10, 100)
        Y_points = LittleRegression.query_model(X_GridPoints)
        pyplt.scatter(X, Y)
        pyplt.plot(X_GridPoints, Y_points, color="r")
        pyplt.show()

    test4(300)


if __name__ == "__main__":
    main()