# Machine learning with Tensorflow
* [Major Refernces and Stuff](#major-references-and-stuff)

* [TF on a high level](#tf-on-a-high-level)

## Major References and Stuff

* [Tensor Flow](https://tensorflow.org/)

* The bible: [Tensor Flow Python](https://www.tensorflow.org/api_docs/python/tf)

* The Keras API for tensor flow [link](https://www.tensorflow.org/guide/keras/sequential_model0)

* Geting started with TF from Google [link](https://developers.google.com/machine-learning/crash-course/first-steps-with-tensorflow/toolkit)

* TF used for linear regression [link](https://www.tensorflow.org/tutorials/keras/regression)

* Google's Machine learning crash course: [link](https://developers.google.com/machine-learning/crash-course)

## Some Vocabularies

* For the list of glossaries, see [here](https://developers.google.com/machine-learning/glossary/#L2_regularization)

* Learning Rate: The eta, or the mu, or the delta, the stepsizes for somekind of machine learining algorithms.

* Features: The predictors, x1, x2, x1*x2, x1^2 ... etc

* Label: Predictants: Y_hat, \vec{\hat{y}}, the quantity to predict using the predictors.

* Hyper Parameters:
  * the set of parameters that defines the shape/sizes/topology of the ML model.

* Metrics: Some kind of methods the determine how well the neuro-net is doing; it's not the
loss function it's never related to the training, it's just a measure of performance of the
model once it has been trained.

  * More on [Tensor Flow Metric](https://www.tensorflow.org/api_docs/python/tf/keras/metrics)

* Biases: The constant term in a linear regression model.

* Layers: The layers of the network:
  * The layers of network is a matrix, where the width is the width of the layers, and the height is the
  width of the previous layer.
  * Yes, each layeris modeled like a row vector.
    * (L1.T)*W = (L2.T)

* Softmax: This function, is like, taking the exponential of the each fothe elements in the array and then
take the value of the variable over the sum of all the other.

  * It appears by the end output side of the neuro-net where it predicts the probability of a output for
  a certain classification results.

* Epoch: A full training pass over the entire dataset such that each example has been seen once. Thus, an epoch
represtnes N/(batch size) training iterations, where N is the total number of examples.
  * (BatchSize)*Epochs = N
  * Note: Batchsize is an random sampling of the training dataset.

* RMSprop: A gradient descend methods that is designed for neuro-net:
  * [Link](https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a)
  * "Rprop combines the idea of only using the sign of the gradient with the idea of adapting the step
  size individually for each weight." -- Top hightlight from the article.

* Loss: The pentalty function for the models, it's NOT the metrics for measuring the efficiency of the model. Don't
confuse these 2 please.

 ## TF on a high level

 * With a set of data, we need to produce a solution using neuro-network
  * Get the data.
  * Define a model.

    * Using the tensorflow.keras.Sequantial.
    * Knowing how to make a model and compile them for training.
    * Knowing all the details about the weights, and biases in the model.
    * Knowing how to save and train the model when needed.
    * Knowing how to tune the meta parameters to get the best model.

  * Train/tune the model.
    * Knowing how to split the data into training and validations set for the model.

------------------------------------------------------------------------------------------------------------------------

* The sequential model [link](https://www.tensorflow.org/guide/keras/sequential_model):
  * One of the APIs provided by the tensor flow is the sequential model, where the user creates the model layers by
  layers, and multiple inputs and outputs for the model is not allowed.
