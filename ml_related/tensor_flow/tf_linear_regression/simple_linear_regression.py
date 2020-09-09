"""
    I will simply follows the Google tutorial and then type them in my
    own ways.
"""

# Importing the modules ------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


# Defines functions that trains the regression model -------------------------------------------------------------------
def build_model(learningRate):
    """
        Define the learning rate, and we will build a simple model that can do regression on
        this type of data: x ~ y

    :param learningRate:
        This is related to the parameters of gradient descent.
    :return:
        The model built
    """

    Model = tf.keras.models.Sequential()
    Model.add(tf.keras.layers.Dense(units=8, input_shape=(1,), activation=tf.keras.activations.relu))
    Model.add(tf.keras.layers.Dense(units=8, activation=tf.keras.activations.relu))
    Model.add(tf.keras.layers.Dense(units=1))
    """
        Model has one node as input, one larger with 5 nodes, and one node as output. 
    """
    Model.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr=learningRate),
        loss="mean_squared_error",  # this defines the penalty for the gradient descend. -----------------------------
        metrics = [tf.keras.metrics.RootMeanSquaredError()]
    )
    print(Model.summary())
    return Model


def train_model(model, features, label, epochs, batchSize):
    """
        This function trains the model, given the hyper parameters

    :param model:
        The model
    :param features:
        The features
    :param label:
        The label of the data
    :param epochs:
        How many times we want to train the stuff.
    :param batch_size:
        The size of samples for each step of training.
    :return:
        Trained_weight, trained_bias, epochs, and the mean square errors of training
    """
    History = model.fit(
        x=features,
        y=label,
        batch_size=batchSize,
        epochs=epochs
    )
    TrainedWeightsBiases = model.get_weights()
    Epochs = History.epoch
    Hist = pd.DataFrame(History.history)  # Make the training history into a pandas df
    Rmse = Hist["root_mean_squared_error"]
    return TrainedWeightsBiases, Epochs, Rmse


def get_training_data():
    def generate_random_poly(queryPoints, epsilon, **kwargs):
        """

        :param queryPoints:
        :param epsilon:
        :param kwargs:
            roots:
                given the roots of the polynomial as a NParray.
            fxn:
                manually provides a function that takes in one
                argument "x" and return the number as the output for the function.
        :param roots:
            the roots of the polynomial
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
                assert type(V).__module__ == np.__name__, f"the type we got is" \
                                                          f": {type(V).__module__ }"  # The type comes from numpy module
                roots = V
            if K is "fxn":
                EvalAt = V

        Res = np.zeros(len(queryPoints))
        Noise = epsilon * np.random.randn(len(queryPoints))
        for I, V in enumerate(queryPoints):
            Res[I] = EvalAt(V) + Noise[I]
        return Res
    # MyFeature = ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    # MyLabel = ([5.0, 8.8, 9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])

    MyFeature = np.arange(0, 10, 0.01)
    MyLabel = generate_random_poly(queryPoints=MyFeature, epsilon=50, roots=np.array([0, 1, 7, 10]))
    return MyFeature, MyLabel


def predict_using_model(model, data):
    return model.predict(data).flatten()


# Visualizing the regression model we made -----------------------------------------------------------------------------

def main():
    my_feature, my_label = get_training_data()
    learning_rate = 0.01
    epochs = 40
    my_batch_size = 20
    my_model = build_model(learning_rate)

    TrainedWeightsBiases, Epochs, Rmse = train_model(
        my_model, my_feature,
        my_label, epochs,
        my_batch_size
    )

    # print(TrainedWeightsBiases)
    # print(Epochs)
    # print(Rmse)
    # print(my_model.layers[0].weights)
    # print(my_model.layers[0].bias.numpy())

    Xs = np.arange(min(my_feature), max(my_feature), 0.01)
    Predictions = predict_using_model(model=my_model, data=Xs)

    plt.scatter(my_feature, my_label)
    plt.scatter(Xs, Predictions)

    plt.show()




if __name__ == "__main__":
    import sys, traceback
    try:
        main()
        sys.exit(0)
    except:
        print(traceback.print_exc())
        sys.exit(-1)
