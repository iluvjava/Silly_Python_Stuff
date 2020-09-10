"""
Learning Objective:
* Making the model.
* Getting the training and test data.
* Split the training set into validations and
    * Get the model efficiency from the validation set,
    * Adding a new parameter called the validation split when doing the machine neuro-training.
* Handling data from the Pandas Dataframe
"""


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
print("Test Data Obtained")
print(TRAINED_DF);print(TEST_DF)
# ----------------------------------------------------------------------------------------------------------------------


# Defining stuff needed for the training of the model ------------------------------------------------------------------
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
        loss="mean_squared_error",  # This is penalty function, 2-norm.
        metrics=[tfk.metrics.RootMeanSquaredError()]
    )
    """
        The metric defined here will be related to measuring the efficiencies of the 
        model on the validations set. 
    """
    print("Model has been constructed: ---------------")
    print(model.summary())
    return model


def train_model(model, df, feature, label, epochs, batchSize=None, validationSplit=0.1):
    History = model.fit(x=df[feature],
                        y=df[label],
                        batch_size=batchSize,
                        epochs=epochs,
                        validation_split=validationSplit  # !!! Validations !!!
        )
    TrainedWeight = model.get_weights()[0]
    TrainedBias = model.get_weights()[1]
    Epochs = History.epoch
    hist = pd.DataFrame(History.history)
    # Extract some information on the metrics for efficiencies measure.
    # RMESE is the root mean squared error on each epoch of the model training.
    RMSE = hist["root_mean_squared_error"]
    return Epochs, RMSE, History.history


# Plotting the training process of the model ---------------------------------------------------------------------------
def plot_the_loss_curve(epochs, maeTraining, maeValidation):
    fig = plt.figure()
    plt.xlabel("Epoch"); plt.ylabel("Root Mean Squared Error")

    # loss on first epoch is ignored.
    plt.plot(epochs[1:], maeTraining[1:], label="Training Loss")
    plt.plot(epochs[1:], maeValidation[1:], label="Validations Loss")
    plt.legend()

    MergedMaeLists = maeTraining[1:] + maeValidation[1:]
    HighestLoss = max(MergedMaeLists); LowestLost = min(MergedMaeLists)
    delta = HighestLoss - LowestLost
    print(delta)  # This is the margin on the axis.
    YaxisTop = HighestLoss + (delta*0.05)
    YaxisBottom = LowestLost - (delta*0.05)
    plt.ylim([YaxisBottom, YaxisTop])
    plt.show()
    return


def plot_the_curve_fit(model, feature, label, df):
    Features, Labels = df[feature], df[label]
    XStart, XEnd = min(Features), max(Features)
    XRange = XEnd - XStart
    XGrid = np.arange(XStart, XEnd, XRange/1000)
    fig = plt.figure()
    plt.scatter(Features, Labels)
    plt.plot(XGrid, model(XGrid[:, np.newaxis]), color="red")
    plt.show()
    return


if __name__ == "__main__":
    def main():
        def Task1():
            """

            :return:
            """
            LearningRate, Epochs, BatchSize = 0.08, 80, 2000
            ValidationSplit = 0.5
            MyFeature, MyLabel = "median_income", "median_house_value"  # Easy regression on the city's median
            # Income and the value median of the house value on that region of the city.

            TheModel = build_model(LearningRate)
            EpochsList, RMSE, History = train_model(
                TheModel, TRAINED_DF, MyFeature, MyLabel, Epochs, BatchSize, ValidationSplit
            )
            plot_the_loss_curve(EpochsList, History["root_mean_squared_error"], History["val_root_mean_squared_error"])
            plot_the_curve_fit(model=TheModel, feature=MyFeature, label=MyLabel, df=TEST_DF)
        Task1()
        return
    import sys, traceback
    try:
        main()
        sys.exit(0)
    except:
        traceback.print_exc()
        sys.exit(-1)