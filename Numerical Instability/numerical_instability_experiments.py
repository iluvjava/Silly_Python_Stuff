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
import matplotlib.pyplot as pyplt
import quick_json.quick_json as qj


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


def python_fsum(theList: List[float]) -> float:
    import math
    return math.fsum(theList)


def main():

    def GetListofErrorsFor(sum1:callable, trials:int=30, listSize:int=30):
        Errors = []
        for TheList in [rand_list(listSize) for _ in range(trials)]:
            S2 = sum1(TheList)
            S3 = rational_sum(TheList)
            Errors.append(abs(S2 - S3))
        return stats.mean(Errors), stats.stdev(Errors)

    def GetExecutionTimeFor(fxn: callable, trials: int=30, listSize: int=30):
        import time
        Times = []
        for TheList in [rand_list(listSize) for _ in range(trials)]:
            TStart = time.time()
            fxn(TheList)
            Times.append(time.time() - TStart)
        return stats.mean(Times), stats.stdev(Times)

    def BenchMarkOnErrors():
        ErrorsMeanPythonSum, Sizes = [], list(range(10, 2000, 10))
        ErrorsMeanNumpySum = []
        ErrorMeanStdPythonSum, ErrorMeanStdNumpySum = [], []
        for I in Sizes:
            LoopTempVar = GetListofErrorsFor(python_sum, trials=30, listSize=I)
            ErrorsMeanPythonSum.append(LoopTempVar[0])
            ErrorMeanStdPythonSum.append(LoopTempVar[1])
            LoopTempVar = GetListofErrorsFor(numpy_sum, trials=30, listSize=I)
            ErrorsMeanNumpySum.append(LoopTempVar[0])
            ErrorMeanStdNumpySum.append(LoopTempVar[1])
        # Plot the Mean of Errors --------------------------------------------------------------------------------------
        fig, ax = pyplt.subplots()
        pyplt.scatter(Sizes, ErrorsMeanPythonSum, label="Default Sum Mean Errors")
        pyplt.scatter(Sizes, ErrorsMeanNumpySum, color="r", label="Numpy Sum Mean Errors")
        legend = ax.legend(loc='upper left', shadow=True, fontsize='small')
        ax.set_xlabel("Array Size")
        ax.set_ylabel("Errors")

        # Plot the Std curve -------------------------------------------------------------------------------------------
        PythonSumStds = ([M + D for M, D in zip(ErrorsMeanPythonSum, ErrorMeanStdPythonSum)],
                         [M - D for M, D in zip(ErrorsMeanPythonSum, ErrorMeanStdPythonSum)])
        NumpySumStds = ([M + D for M, D in zip(ErrorsMeanNumpySum, ErrorMeanStdNumpySum)],
                         [M - D for M, D in zip(ErrorsMeanNumpySum, ErrorMeanStdNumpySum)])
        pyplt.plot(Sizes, PythonSumStds[0], color="b")
        pyplt.plot(Sizes, PythonSumStds[1], color="b")
        pyplt.plot(Sizes, NumpySumStds[0], color="r")
        pyplt.plot(Sizes, NumpySumStds[1], color = "r")

        # Plot and save these things -----------------------------------------------------------------------------------
        pyplt.savefig("Error plots for Numpy, and default python.png", dpi=400)
        pyplt.show()
        JsonData = {}
        JsonData["Sizes"] = Sizes
        JsonData["ErrorsMeanPythonSum"] = ErrorsMeanPythonSum
        JsonData["ErrorsMeanNumpySum"] = ErrorsMeanNumpySum
        JsonData["PythonSumStdUpper"], JsonData["PythonSumStdLower"] = PythonSumStds[0], PythonSumStds[1]
        JsonData["NumpySumStdUpper"], JsonData["NumpySumStdLower"] = NumpySumStds[0], NumpySumStds[1]
        qj.json_encode(JsonData, "errors.json")

    def BenchMarkOnTimesFor(fxn: callable, sizes: List[int]):
        Means, Upper, Lower = [], [], []

        for I in sizes:
            TheSumMeans, TheSumStd, = GetExecutionTimeFor(fxn, trials=10, listSize=I)
            Means.append(TheSumMeans)
            Upper.append(TheSumMeans + TheSumStd)
            Lower.append(TheSumMeans - TheSumStd)

        return Means, Upper, Lower


    def PlotTheExecutionTime():
        # Prepare the Json to store things -----------------------------------------------------------------------------
        JsonData = {}

        fig, ax = pyplt.subplots()
        Xs = list(range(10, 2000, 10))
        JsonData["Sizes"] = Xs
        # Calling the modules and scatter plot the data ----------------------------------------------------------------
        Means, Upper, Lower = BenchMarkOnTimesFor(fxn=khan_sum, sizes=Xs)
        ax.scatter(Xs, Means, color="r", s=0.5, label="Kahan sum")
        JsonData["Kahan Sum"] = {"Means": Means, "Upper": Upper, "Lower": Lower}

        Means, Upper, Lower = BenchMarkOnTimesFor(fxn=python_fsum, sizes=Xs)
        ax.scatter(Xs, Means, color="b", s=0.5, label="python fsum")
        JsonData["Python Fsum"] = {"Means": Means, "Upper": Upper, "Lower": Lower}

        Means, Upper, Lower = BenchMarkOnTimesFor(fxn=numpy_sum, sizes=Xs)
        ax.scatter(Xs, Means, color="g", s=0.5, label="numpy sum")
        JsonData["Numpy Sum"] = {"Means": Means, "Upper": Upper, "Lower": Lower}

        Means, Upper, Lower = BenchMarkOnTimesFor(fxn=rational_sum, sizes=Xs)
        ax.scatter(Xs, Means, color="black", s=0.5, label="rational sum")
        JsonData["Rational Sum"] = {"Means": Means, "Upper": Upper, "Lower": Lower}

        legend = ax.legend(loc='upper left', shadow=True, fontsize='small')
        ax.set_xlabel("Array Size")
        ax.set_ylabel("Execution time/sec")
        pyplt.savefig("Execution time.png", dpi=400)
        # showing and saving stuff: ------------------------------------------------------------------------------------
        fig.show()
        qj.json_encode(JsonData, "Execution time.json")


    # BenchMarkOnErrors()
    PlotTheExecutionTime()




if __name__ == "__main__":
    main()