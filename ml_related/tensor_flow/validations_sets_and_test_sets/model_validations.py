# Import relevant modules ----------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# Prepare the data we are gonna use ------------------------------------------------------------------------------------
TRAINED_DF = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
TEST_DF = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")
ScaleFactor = 10000.0
TRAINED_DF["median_house_value"] /= ScaleFactor
TEST_DF["median_house_value"] /= ScaleFactor

def main():
    pass


def build_model(learningRate):
    """
        Build a model with learnng rate.
    :param learningRate:
        positive float, it's for gradient descend.
    :return:
    """
    tfk = tf.keras
    model = tfk.models.Sequential()
    model.add(tfk.layers.Dense(units=1, input_shape=(1,)))
    model.compile(
        optimizer=tfk.optimizers.RMSprop(lr=learningRate),
        loss="mean_squared_error",
        metrics=[tfk.metrics.RootMeanSquaredError]
    )
    return model


if __name__ == "__main__":
    import sys, traceback
    try:
        main()
        sys.exit(0)
    except:
        traceback.print_exc()
        sys.exit(-1)