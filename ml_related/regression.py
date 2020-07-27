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
        - For dence data points, we have to either use some down-sampling
        - Or we have to use non-parametric regression.

    * So in some cases there can be colinearity, say: x1~x2, and x1, x2 are both predictors for y, this reduces the
    amount of information if it's used for predicting with a linear regression model.

    say x1 = a + b*x2, then we have y = w1*x1 + w2*x2 = w1*(a + b*x2) + w2*x2, which is just...
    = w1*a + (w1*b + w2)*x2, and the parameters for one of the predictor doesn't have to exist at all.

    * On the other than the ridge regression is good at reducing the model complexity, if for some reasons, one has
    to use a model containing a lot of DF, then it's better to use the ridge regression as a way to reduce the complexity
    of the model.

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
from sklearn.preprocessing import PolynomialFeatures, scale
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

        * This kind of search will try to reuse on the point from that last iteration, in theory it
        will reduce the total number of evaluations involved for the function.
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
    phi = (math.sqrt(5) + 1) / 2
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    f_c, f_d = f(c), f(d)
    while abs(c - d) > tol:
        if f_c < f_d:
            b = d
            f_d = f_c
            f_c = f(b - (b - a) / phi)
        else:
            a = c
            f_c = f_d
            f_d = f(a + (b - a) / phi)
        # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / phi
        d = a + (b - a) / phi
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
        raise Exception("Empty Shell method. ")

    def query(self, x:NpArray):
        """
            Query the trained model with a list of data points.
        :param x:
            A set of query points for the model.
        :return:
            A all the predicted values for the model for that inputs.
        """
        raise Exception("Empty Shell method. ")

    def size(self):
        """

        :return:
            The number of data points in the model.
        """


class MultiVarRegularizedRegression(MyRegression):
    """
        Class designed to optimize the lambda param to produce the best model for a given data set.

        * This instance of the model will be trained with the alpha parameters.
        * Prove dynamic swappable static functions for instances of the class to define different training
        routine of the model.
        1. Lasso Regression:
            Best for eliminating colinearity for multi-variable models. However the alpha funciton against
            training errors for MSE is not convex.
        2. Ridge regression:
            Smooth out some of the parametesrs for a model with a lot of non-linearity to prevent over fitting.
        3. LassoLARS Regression:
            Uses specialized optimizer rather than the general coordinate descended optimizer.
    """

    def __init__(self, predictorsData:NpArray,
                 predictantData:NpArray,
                 deg=2,
                 modelTrainingFxn:callable=None):
        """
            initialize it with an instance of data points.
        :param predictorsData:
            This is the xData, must be a 1d row np vector.
        :param predictantData:
            This is the yData, must be a 1d row np vector.
        :param modelTrainingFxn:
            This is going to be one of the selections of static methods in this class
            that specifies the training style of the algorithm.

            * The meta parameters depends...
        """
        self._Predictors, self._Predictants = np.copy(predictorsData), np.copy(predictantData)
        if len(self._Predictors.shape) == 1:
            self._Predictors = self._Predictors[:, np.newaxis]
        self._Deg = deg
        self._LinModel = None
        self._TrainingStyle = MultiVarRegularizedRegression.train_model_ridge_style if modelTrainingFxn is None\
            else modelTrainingFxn

    @property
    def Predictors(self):
        return self._Predictors
    
    @property
    def Predictants(self):
        return self._Predictants

    @property
    def LinModel(self):
        return self._LinModel

    def train_model_for(self, indices, alpha):
        """
            The list of indices are all the indices that should be linked to a training data set.
            All other indices thare are not indices are assumed to be the validation set.

            * Each time it's trained, the model in the field will get updated.
        :param indices:
            A list of indices. Vallina array is ok. s
        :param alpha:
            A parameters that can be tweaked by the trainer instance to obtain the best inputs.
        :return:
            Linear model, a number that is related to the quality of training (Usually MSE)
        """
        Predictors_Training, Preditants_Training = self._Predictors[indices,...], self._Predictants[indices, ...]
        n = self._Predictors.shape[0]
        Predictors_Test, Preditants_Test = self._Predictors[[I for I in range(n) if I not in indices], ...],\
                                           self._Predictants[[I for I in range(n) if I not in indices], ...]
        self._LinModel = self._TrainingStyle(Predictors_Training, Preditants_Training, alpha=alpha, deg=self._Deg)
        Predicted = self.query(Predictors_Test)
        MSE = sum((Predicted - Preditants_Test)**2)
        MSE /= n
        return self._LinModel, MSE

    @staticmethod
    def train_model_ridge_style(predictors, predictants, alpha, deg):
        """
            This is dispatchable static method for an instance of the class. 
            Passing this parameter in for the constructor will allow the model to be trained with 
            certain regularization.
        :param predictors: 
        :param predictants: 
        :param alpha: 
        :param deg: 
        :return: 
        """
        Vandermonde = PolynomialFeatures(degree=deg)
        Vandermonde = Vandermonde.fit_transform(predictors)
        LinModel = linear_model.Ridge(alpha=alpha)
        LinModel = LinModel.fit(Vandermonde, predictants)
        return LinModel

    @staticmethod
    def train_model_lasso_style(predictors, predictants, alpha, deg):
        """
            Dispatachable methodd for an instance of the class that regularize the model with lasso regression。
        :param predictors:
        :param predictants:
        :param alpha:
        :param deg:
        :return:
        """
        Vandermonde = PolynomialFeatures(degree=deg)
        Vandermonde = Vandermonde.fit_transform(predictors)
        LinModel = linear_model.Lasso(alpha=alpha)
        LinModel = LinModel.fit(Vandermonde, predictants)
        return LinModel

    @staticmethod
    def train_model_lassoLARS_style(predictors, predictants, alpha, deg):
        Vandermonde = PolynomialFeatures(degree=deg)
        Vandermonde = Vandermonde.fit_transform(predictors)
        LinModel = linear_model.LassoLars(alpha=alpha)
        LinModel = LinModel.fit(Vandermonde, predictants)
        return LinModel

    def query(self, x:NpArray):
        """
            Query the trained model with a list of data points.
        :param x:
            A set of query points for the model.
        :return:
            A all the predicted values for the model for that inputs.
        """
        assert self._LinModel is not None, "You can't query the model when it's not trained yet. "
        PolyFeatures = PolynomialFeatures(self._Deg)
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        Vandermonde = PolyFeatures.fit_transform(x)
        Predicted = self._LinModel.predict(Vandermonde)
        return Predicted

    def size(self):
        return len(self._Predictors)



    @staticmethod
    def random_3d_regression_data(N:int):
        """
            Prepare all the rando data point for regression training,
            (x1, x2)~y
            * Predictors are going to be in a unit square of [-1, 1]X[-1, 1], normally distributed.
            * The varinace on the preditants are going to be normal too.
            * Second order and terms of interactions are involved into the random model.
        :return:
            x, y
        """
        Rnd_Predictors = np.random.normal(loc=0, scale=1, size=(N, 2))
        def LinearModel(predictors, w1, w2, w3, w4):
            Expression = lambda x, y: w1*x + w2*y + w3*x*y + w4*x**2
            Y = np.array([Expression(T[0], T[1]) + np.random.normal(loc = 0, scale = 0.3) for T in predictors])
            return Y
        return Rnd_Predictors, LinearModel(Rnd_Predictors, 1, 1, 0.5, 0.2)

class SGDRegression(MyRegression):
    """
        We uses the GSD regressor in the sklearn package to do some stuff, which gives more
        flexibility to our model under trainings.
    """
    def __init__(self):
        pass

class MyLittleRegression(MyRegression):
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
        polyOrder = int(polyOrder) # this might not be int if query from the trainer.
        X_train, X_test = np.array([self.XDataPoints[I] for I in indices]), \
                          np.array([self.XDataPoints[I] for I in range(len(self.XDataPoints)) if I not in indices])
        Y_train, Y_test = np.array([self.YDataPoints[I] for I in indices]), \
                          np.array([self.YDataPoints[I] for I in range(len(self.YDataPoints)) if I not in indices])
        LinModel, V = ordinary_regression_fit_2d(X_train, Y_train, polyOrder=polyOrder)
        self._LinModel = LinModel
        MSE, _ =  regression_MSE_compute(LinModel, X_test, Y_test)
        return LinModel, MSE

    def query(self, x:NpArray):
        """
            Query the model with a list of points, and it will return the prediction.
        :param x:
            ROW VECTOR or villina Array.
        :return:
            Vanilla flavor array for the predicted data points.
        """
        assert self._LinModel is not None, "You can't query the model before actually training the model. "
        Vandermonde = PolynomialFeatures(degree=len(self._LinModel.coef_) - 1)
        # create the vandermonde matrix base on the df of the model.
        Vandermonde = Vandermonde.fit_transform(x[:,np.newaxis])
        Predicted = self._LinModel.predict(Vandermonde)
        return Predicted

    def size(self):
        return len(self._X)

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

        * This trainer can only train on parameter at a time.
    """
    def __init__(self, metaParamLower, metaParamUpper, regressionTrainee):
        """

        :param metaParamUpper:
            This parameter is the upper bound for the meta parameter that gets optimized
            for the test error of the data set.
        :param metaParamLower:
            This is the lowerbound for the tunning of the meta parameter of the model.
        :param regressionTrainee:
            An instance of the regression class, possible represent a model together boundle with its data.
        """

        self._MetaParamUpper, self._MetaParamLower = metaParamUpper, metaParamLower
        self._Trainee = regressionTrainee

    def train_it_on(self, N=1, tol=1):
        """
            Trains on a certain set of data and figure out the best degree for the
            polynomial for the data.

        :param xData:
            Row NParray vector.
        :param yData:
            Row Nparray vector.
        :param N:
            The number of test and train instances for the model.
        :param tol:
            This is the delta vicinity you want for the the meta-prameter.
        :return:
            min meta-param, min MSE, Instance of Mylittle Regression.
        """
        Test_Indices = [rand_split_idx(self._Trainee.size()) for I in range(N)]
        Regression = self._Trainee
        def mse_error(metaParam):
            MSE_List = [None]*N
            print(f"Trainner checking metaparam: {metaParam}")
            for I, IdxList in enumerate(Test_Indices):
                _, MSE = Regression.train_model_for(IdxList, metaParam)
                MSE_List[I] = MSE
            return sysstat.mean(MSE_List)
        Argmin, min = golden_section_search(mse_error, self._MetaParamLower, self._MetaParamUpper, tol)
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

        MyRegression = MyLittleRegression(X, Y)
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
        X = np.random.uniform(-3, 3, testPoints)
        Y = generate_random_poly(X, epsilon=2, roots=np.array([-1, 1]))
        X, Y = scale(X), scale(Y)

        Trainer1 = MyLittleRegressionTrainer(1, 10, MyLittleRegression(X, Y))
        Trainer2 = MyLittleRegressionTrainer(0, 10, MultiVarRegularizedRegression(X, Y, deg=10))
        Trainer3 = MyLittleRegressionTrainer(0, 1, MultiVarRegularizedRegression(
            X,
            Y,
            deg=10,
            modelTrainingFxn=MultiVarRegularizedRegression.train_model_lasso_style))
        Trainer4 = MyLittleRegressionTrainer(0, 0.1, MultiVarRegularizedRegression(
            X,
            Y,
            deg=10,
            modelTrainingFxn=MultiVarRegularizedRegression.train_model_lassoLARS_style)
        )

        Deg, MinMSE, LittleRegression = Trainer1.train_it_on(N=20, tol=1)
        print(f"test4, simple regression get deg = {Deg}, min MSE = {MinMSE}")
        Alpha, MinMSE, RidgeRegression = Trainer2.train_it_on(N=20, tol=0.01)
        print(f"test4 ridge regression get alpha = {Alpha}, min MSE = {MinMSE}")
        print(f"Ridge Coefficients: {RidgeRegression.LinModel.coef_}")
        Alpha, MinMSE, LassoRegression = Trainer3.train_it_on(N=20, tol=0.01)
        print(f"test4 lasso regression get alpha = {Alpha}, min MSE = {MinMSE}")
        print(f"Lasso Coefficients: {LassoRegression.LinModel.coef_}")
        Alpha, MinMSE, LassoLARSRegression = Trainer4.train_it_on(N=20, tol=1e-4)
        print(f"test 4, lassoLARS regression get alpha={Alpha}")
        print(f"Lasso LARS Coefficients: {LassoLARSRegression.LinModel.coef_}")


        X_GridPoints = np.linspace(min(X), max(X), 100)
        Y_points = LittleRegression.query(X_GridPoints)
        pyplt.scatter(X, Y, color="m")
        pyplt.plot(X_GridPoints, Y_points, color="r")

        Y_points = LassoRegression.query(X_GridPoints)
        pyplt.plot(X_GridPoints, Y_points, color="b")

        Y_points = LassoLARSRegression.query(X_GridPoints)
        pyplt.plot(X_GridPoints, Y_points, color="c")

        Y_points = RidgeRegression.query(X_GridPoints)
        pyplt.plot(X_GridPoints, Y_points, color="g")
        pyplt.show()

    def test5():
        X, Y = MultiVarRegularizedRegression.random_3d_regression_data(3)
        print(X, Y)

    def test6():
        X, Y = MultiVarRegularizedRegression.random_3d_regression_data(30)
        TheRegression = MultiVarRegularizedRegression(X, Y, deg=3)
        LinModel, MSE = TheRegression.train_model_for(rand_split_idx(len(X)), alpha=0)
        print(f"Ridge Linear Model coef:{LinModel.coef_}")
        print(f"Ridge MSe: {MSE}")

    def test7():
        X, Y = MultiVarRegularizedRegression.random_3d_regression_data(20)
        Trainee = MultiVarRegularizedRegression(X, Y, deg=3)
        Trainer = MyLittleRegressionTrainer(metaParamLower=0, metaParamUpper=20, regressionTrainee=Trainee)
        Argmin, Min, RegressionModel = Trainer.train_it_on(N=5, tol=0.1)
        print(f"Argmin: {Argmin}, Min: {Min}")
        print(f"regression Model Coefficients: P {RegressionModel.LinModel.coef_}")

    test4(200)

if __name__ == "__main__":
    main()