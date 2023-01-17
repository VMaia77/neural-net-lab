## **NeuralNetLab**


The goal of this project was to design and implement a customizable neural network library from scratch in Python. The resulting library offers a user-friendly interface for creating and training neural networks. The project is open-source, feel free to contribute.

## Table of contents

- How to use and contribute
- Model architeture
- Regression
- Classification
- Activations
- Regularization
- Optimizers
- Model initialization
- Hyperparameters
- Examples
- Limitations
- References


## How to use and contribute


You can start forking, cloning, and installing the requirements. After this you can start using and contributing to this project!

See the examples section for more usage examples.


## Model architeture


The implemented neural network is a Multi-layer Perceptron (MLP) consisting of an input layer, hidden layers, and an output layer. The input layer is flexible, allowing for input data with any number of features (*n* samples x *n* features). The number of hidden layers and their number of neurons can be chosen by the user. The output layer is also flexible, allowing for output data with any number of values/classes to predict (*n* samples x *n* values/classes to predict).


## Regression


For regression tasks, the neural network predicts continuous values. The model's predictions can be compared to the true values using a threshold of precision, which measures the absolute difference between the predictions and true values and compare the error with a threshold value. The model can also be evaluated using the mean absolute error (MAE), mean squared error (MSE), and root mean squared error (RMSE). Normalized versions of these metrics, such as MAE divided by the standard deviation of the true values (y) (MAEr) or MSE divided by the standard deviation of y, can also be used. However, only MAEr and the threshold of precision are implemented in the library.

It's common to confuse the term 'multivariate regression' with 'multiple regression', however, multivariate regression is a method where multiple dependent variables are predicted using one or more independent variables. In contrast, multiple regression analysis is a technique that uses multiple independent variables to predict one or more dependent variables. The model works for both.


### **Accuracies**


`AccuracyRegressionThreshold` (Precision threshold): This measures the absolute difference between the model's predictions and the true values and compares it to a specified threshold (precision). It returns a boolean array indicating whether the difference between the predictions and true values is less than the threshold for each sample. For example, if the predictions are [1.2, 3.4, 5.6] and the true values are [1.0, 3.5, 5.7], and the threshold is 0.5, the result would be [True, True, False]. This means that the model's predictions are within the specified threshold of the true values for the first two samples, but not for the third sample.

This accuracy metric is useful for regression tasks where you want to know how close the model's predictions are to the true values. By setting the threshold to a higher or lower value, you can control how much error you are willing to tolerate.
The neural network model is capable of predicting multiple continuous values for each sample in the input data, making it suitable for multivariable regression tasks.


`AccuracyRegressionMAEr` (Mean absolute error relative to std of y): this metric measures the model's accuracy for regression tasks by calculating the mean absolute error (MAE) relative to the standard deviation of the true values (y) it calculates the average absolute difference between the model's predictions and the true values, normalized by the spread of the true values. This helps to account for cases where the true values have a large or variable range, as it helps to normalize and interpret the error.


### **Loss functions**


`LossMeanAbsoluteError` (Mean absolute error): measures the average absolute difference between the model's predictions and the true values.

`LossMeanSquaredError` (Mean squared error): measures the average squared difference between the model's predictions and the true values.


## Classification


For classification tasks, the neural network predicts discrete classes. The model can be evaluated using accuracy, which measures the proportion of correct predictions. The neural network supports binary and categorical classification.

For multivariable binary classification tasks, the neural network model is capable of predicting multiple discrete classes for each sample in the input data. 


### **Accuracies**


`AccuracyCategorical`: measures the model's accuracy for classification tasks with multiple classes. It is calculated as the average of the model's correct predictions across all classes. For example, if the model correctly predicts 3 out of 5 samples for one class and 4 out of 5 samples for another class, its AccuracyCategorical would be 3.5/5 = 70%.


### **Loss functions**


`LossBinaryCrossentropy` (Binary cross-entropy): measures the difference between the model's predicted probabilities and the true labels for binary classification tasks.

`LossCategoricalCrossentropy` (Categorical cross-entropy): measures the difference between the model's predicted probabilities and the true labels for classification tasks with multiple classes.


## Activations


`ActivationLinear` (Linear): This activation function simply returns the input value.

`ActivationReLU` (ReLU; Rectified Linear Unit): This activation function returns the input value if it is positive, and 0 if it is negative.

`ActivationSigmoid` (Sigmoid): This activation function squashes input values between 0 and 1.

`ActivationSoftmax` (Softmax): This activation function squashes input values between 0 and 1, and scales the values so that they add up to 1.


## Regularization


The library includes dropout, L1 and L2 regularization to prevent overfitting. Dropout regularization randomly sets a fraction of input units to zero during training, while L1 and L2 regularization add penalties to the cost function based on the absolute or squared values of the weights and biases, respectively. These techniques can help to reduce the complexity of the model and improve its generalization to new data.


## Optimizers


`OptimizerSGD` (SGD; Stochastic Gradient Descent): This optimizer performs gradient descent by updating the model's weights and biases based on the gradient of the loss function with respect to the weights and biases. It does this by taking small steps in the direction that minimizes the loss, based on the learning rate specified by the user.

`OptimizerAdagrad` (Adagrad; Adaptive Gradient)*: This optimizer is similar to stochastic gradient descent, but it adjusts the learning rate adaptively based on the historical gradient information. This helps to improve the convergence of the model, particularly when the data has a high variance or is sparse.

`OptimizerRMSprop` (RMSProp; Root Mean Squared Propagation)*: This optimizer is also similar to stochastic gradient descent, but it uses a running average of the squared gradient to scale the learning rate adaptively. This helps to prevent oscillations and improve the convergence of the model.

`OptimizerAdam` (Adam; Adaptive Moment Estimation): This optimizer is a combination of stochastic gradient descent and RMSprop, with additional momentum terms to improve the convergence of the model. It adjusts the learning rate adaptively based on the historical gradient and momentum information.

*Implemented in code but not in the library.


## Model initialization


The start_params dictionary specifies the following parameters for the NeuralNetwork class:

`input_shape`: the number of features in the input data (X_train.shape[1]).

`output_n_neurons`: the number of neurons in the output layer (y_train.shape[1]).

`output_activation`: the activation function for the output layer.  Activations: `'linear'`, `'relu'`, `'sigmoid'`, `'softmax'`.

`accuracy`: the metric used to evaluate the model's performance. Accuracy metrics: `'regression_threshold'`, `'regression_mae'`, `'categorical'`.

The NeuralNetwork class is then instantiated using these parameters with the **start_params syntax, which "unpacks" the dictionary into keyword arguments.


## Hyperparameters


The neural network library offers a range of hyperparameters that can be customized by the user to optimize model performance. 

The following hyperparameters can be specified when initializing a NeuralNetwork object:

`loss`: the loss function used to measure the error of the model. Possible values: 'binary_cross_entropy', 'categorical_cross_entropy', 'mean_absolute_error', 'mean_squared_error'.

`optimizer` (optional): the optimizer used to adjust the model's weights and biases based on the loss function. Possible values: 'sgd', 'adam'. Default value: 'adam'.

`epochs` (optional): the number of training epochs. Default value: 100.

`batch_size` (optional): the size of each training batch. If set to None, each epoch will consist of a single batch containing the entire dataset. Default value: None.

`n_layers`: the number of training layers in the neural network.

`n_neurons`: the number of neurons in each training layer.

`learning_rate` (optional): the learning rate used by the optimizer. Default value: 0.0001.

`decay` (optional): the decay rate used by the optimizer. Default value: 1e-3.

`epsilon` (optional): the epsilon value used by the Adam optimizer. Default value: 1e-7.

`beta_1` (optional): the beta_1 value used by the Adam optimizer. Default value: 0.9.

`beta_2` (optional): the beta_2 value used by the Adam optimizer. Default value: 0.999.

`momentum` (optional): the momentum value used by the SGD optimizer. Default value: 0.4.

`add_dropout` (optional): a flag indicating whether to add a dropout layer after each training layer. Possible values: 0, 1. Default value: 0.

`dropout_rate` (optional): the dropout rate to use. Default value: 0.3.

`weights_l1_regularizer` (optional): the L1 regularization strength applied to the weights. Default value: 0.

`weights_l2_regularizer` (optional): the L2 regularization strength applied to the weights. Default value: 0.

`biases_l1_regularizer` (optional): the L1 regularization strength applied to the biases. Default value: 0.

`biases_l2_regularizer` (optional): the L2 regularization strength applied to the biases. Default value: 0.

These hyperparameters are then used to train the model.

Note that this code only includes a subset of the possible hyperparameters listed in the previous example. To include all of the hyperparameters, you would need to add the additional key-value pairs to the params dictionary.

Hyperparameter tunning can be carried out using the built-in `GridSearch` class.


## Examples


In the notebooks folder ('./notebooks') there are examples for regression (uni and multivariate), classification (binary and multiclass) and hyperparameter tunning. Note that these notebooks may not run if they are not in the root of the project, in this case, you can move from the notebooks folder to the root of the project.


## Limitations


The neural network's architecture is limited to a Multi-layer Perceptron (MLP) with fully connected layers, which may not be suitable for all types of data and tasks. In particular, the MLP architecture may struggle to learn complex relationships and patterns in the data, such as those found in images and natural language.  Other types of neural network architectures, such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs), may be better suited to these types of data and tasks. The current version also don't support hidden layers with different number of neurons, unless the user add the layers manually using the `add_layer()` and `build()` methods of the model class.


## References

- Chollet, F.: Deep Learning with Python, 1st edn. Manning Publications Co., Greenwich (2017).

- https://github.com/tensorflow/tensorflow

- https://github.com/keras-team/keras


The main reference of this project was this book:

- Harrison Kinsley & Daniel Kukieła. Neural Networks from Scratch (NNFS). https://nnfs.io

Despite the fact that the library contains additional code and that many modifications have been made to the original book code, the MIT license of the book code can be found in the `src/.licenses/nnfs` folder.


