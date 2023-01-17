import numpy as np
import copy
import pickle
from typing import Tuple
from src.layers.layers import LayerInput, LayerDense, LayerDropout, LayerActivationSoftmaxLossCategoricalCrossentropy
from src.activations.activation_functions import ActivationLinear, ActivationReLU, ActivationSigmoid, ActivationSoftmax
from src.losses.losses import LossBinaryCrossentropy, LossCategoricalCrossentropy, LossMeanAbsoluteError, LossMeanSquaredError
from src.optimizers.optimizers import OptimizerSGD, OptimizerAdam
from src.accuracies.accuracies import AccuracyRegressionThreshold, AccuracyRegressionMAEr, AccuracyCategorical
from src.utils.batches import set_steps, batch_split


class NeuralNetwork:
    """Neural network model.
    """

    def __init__(self, *, input_shape: int, output_n_neurons: int, 
        output_activation: str, accuracy: str, 
        accuracy_regression_threshold_std_divisor: float = 200, verbose: bool = True) -> None:
        """
        Initalizes the model with the starting parameters.

        Args:
            input_shape (int): Number of neurons in the input layer (X).

            output_n_neurons (int): Number of neurons in the output layer.

            output_activation (str): Activation function of the output layer. Activations: `'linear'`, `'relu'`, `'sigmoid'`, `'softmax'`.

            accuracy (str): Accuracy metrics: `'regression_threshold'`, `'regression_mae'`, `'categorical'`.

            accuracy_regression_threshold_std_divisor (float, optional): If accuracy is regression threshold, set the divisor. Defaults to 200.

            verbose (bool, optional): Prints training information if `True`. Defaults to True.
        """
        self.input_shape = input_shape

        self.output_n_neurons = output_n_neurons

        self.output_activation = output_activation        

        self.accuracy_regression_threshold_std_divisor = accuracy_regression_threshold_std_divisor
        self.set_accuracy(accuracy)

        self.input_layer = None
        self.layers = []
        
        self.softmax_classifier_output = None

        self.epoch_losses = []
        self.epoch_accuracies = []

        self.accumulated_epoch_losses = []
        self.accumulated_epoch_accuracies = []

        self.validation_epoch_losses = []
        self.validation_epoch_accuracies = []
        
        self.loss_str = None
        self.optimizer_str = None

        self.verbose = verbose
        
    def set_params(self, *, loss: str, optimizer: str = 'adam', 
        epochs: int = 100, batch_size: int = None,        
        n_layers: int = None, n_neurons: int = None, 
        learning_rate: float = 0.0001, decay: float = 1e-3, epsilon: float = 1e-7, 
        beta_1: float = 0.9, beta_2: float = 0.999, momentum: float = 0.4, add_dropout: int = 0, dropout_rate: float = 0.3, 
        weights_l1_regularizer: float = 0, weights_l2_regularizer: float = 0, 
        biases_l1_regularizer: float = 0, biases_l2_regularizer: float = 0) -> None:
        """Set model parameters, useful in hyperparameter tunning.

        Args:

            loss (str): Loss functions: `'binary_cross_entropy'`, `'categorical_cross_entropy'`, `'mean_absolute_error'`, `'mean_squared_error'`.

            optimizer (str, optional): Set the optimizer. Optimizers: `'sgd'`, `'adam'`. Defaults to `'adam'`.

            epochs (int, optional): Number of training epochs. Defaults to 100.

            batch_size (int, optional): Size of each batch. If None, each epoch will consist in one batch of the whole data. Defaults to None.

            n_layers (int): Number of training layers in the neural network.

            n_neurons (int): Number of output neurons in each training layer.

            learning_rate (float, optional): Optimizer learning rate. Defaults to 0.0001.

            decay (float, optional): Optimizer decay. Defaults to 1e-3.

            epsilon (float, optional): Optimizer epsilon (only for Adam). Defaults to 1e-7.

            beta_1 (float, optional): Optimizer beta 1 (only for Adam). Defaults to 0.9.

            beta_2 (float, optional): Optimizer beta 2 (only for Adam). Defaults to 0.999.

            momentum (float, optional): Optimizer momentum (only for SGD). Defaults to 0.4.

            add_dropout (int, optional): Add, if 1, dropout layer after each training layer. Defaults to 0.

            dropout_rate (float, optional): Dropout rate. Defaults to 0.3.

            weights_l1_regularizer (float, optional): Weights L1 regularizer. Defaults to 0.

            weights_l2_regularizer (float, optional):  Weights L2 regularizer. Defaults to 0.

            biases_l1_regularizer (float, optional):  Biases L1 regularizer. Defaults to 0.

            biases_l2_regularizer (float, optional): Biases L2 regularizer. Defaults to 0.
        """ 
        
        self.set_loss(loss)
        self.set_optimizer(optimizer)
        
        self.n_layers = n_layers
        self.n_neurons = n_neurons

        self.epochs = epochs
        self.batch_size = batch_size

        self.learning_rate = learning_rate
        self.decay = decay 
        self.epsilon = epsilon
        self.beta_1 = beta_1 
        self.beta_2 = beta_2 

        self.momentum = momentum

        self.add_dropout = add_dropout 
        self.dropout_rate = dropout_rate

        self.weights_l1_regularizer = weights_l1_regularizer 
        self.weights_l2_regularizer = weights_l2_regularizer 
        self.biases_l1_regularizer = biases_l1_regularizer
        self.biases_l2_regularizer = biases_l2_regularizer

        self.add_layer('dense', n_neurons = self.n_neurons, input_shape = self.input_shape,  activation='relu', \
            weights_l1_regularizer=self.weights_l1_regularizer, weights_l2_regularizer=self.weights_l2_regularizer,\
                biases_l1_regularizer=self.biases_l1_regularizer, biases_l2_regularizer=self.biases_l2_regularizer)

        for _ in range(self.n_layers):

            self.add_layer('dense', n_neurons = self.n_neurons, activation='relu', \
                weights_l1_regularizer=self.weights_l1_regularizer, weights_l2_regularizer=self.weights_l2_regularizer,\
                    biases_l1_regularizer=self.biases_l1_regularizer, biases_l2_regularizer=self.biases_l2_regularizer)

            if self.add_dropout > 0.5:
                self.add_layer('dropout', rate=self.dropout_rate)

        self.add_layer('dense', n_neurons = self.output_n_neurons, activation=self.output_activation, \
            weights_l1_regularizer=self.weights_l1_regularizer, weights_l2_regularizer=self.weights_l2_regularizer, \
                biases_l1_regularizer=self.biases_l1_regularizer, biases_l2_regularizer=self.biases_l2_regularizer)        

        if isinstance(self.optimizer , OptimizerAdam):
            optimizer_parameters = dict(learning_rate=self.learning_rate, decay=self.decay, epsilon=self.epsilon, \
                beta_1=self.beta_1, beta_2=self.beta_2)
        else:
            optimizer_parameters = dict(learning_rate=self.learning_rate, decay=self.decay, momentum=self.momentum)

        self.set_optimizer_parameters(optimizer_parameters)

        self.build()

    def add_layer(self, layer_class: str, n_neurons: int = 1, input_shape: int = None, activation: str = None, *, 
        weights_l1_regularizer: float = 0, weights_l2_regularizer: float = 0, 
        biases_l1_regularizer: float = 0, biases_l2_regularizer: float = 0, rate: float = 0) -> None:
        """Add layer to the model.

        Args:
            layer_class (str): Layer type. Layers: `'dense'`, `'dropout'`.

            n_neurons (int): Number of output neurons in the layer.

            input_shape (int, optional): Number of neurons in the input layer (X).

            activation (str, optional): Activation function of the output layer. 

            Activations: `'linear'`, `'relu'`, `'sigmoid'`, `'softmax'`. Defaults to None.

            weights_l1_regularizer (float, optional): Weights L1 regularizer. Defaults to 0.

            weights_l2_regularizer (float, optional):  Weights L2 regularizer. Defaults to 0.

            biases_l1_regularizer (float, optional):  Biases L1 regularizer. Defaults to 0.

            biases_l2_regularizer (float, optional): Biases L2 regularizer. Defaults to 0.

            rate (float, optional): Dropout rate. Defaults to 0.
        """
        if input_shape is None:
            input_shape = self.layers[-1].get_n_neurons()

        if layer_class == 'dense':
            layer = LayerDense(input_shape, n_neurons, \
                weights_l1_regularizer, weights_l2_regularizer, biases_l1_regularizer, biases_l2_regularizer)

        elif layer_class == 'dropout':
            layer = LayerDropout(n_neurons=self.layers[-1].get_n_neurons(), rate=rate)

        self.layers += layer,

        if activation == 'linear':
            activation = ActivationLinear(n_neurons = self.layers[-1].get_n_neurons())
            self.layers += activation,
        if activation == 'relu':
            activation = ActivationReLU(n_neurons = self.layers[-1].get_n_neurons())
            self.layers += activation,
        elif activation == 'sigmoid':
            activation = ActivationSigmoid(n_neurons = self.layers[-1].get_n_neurons())
            self.layers += activation,
        elif activation == 'softmax':
            activation = ActivationSoftmax(n_neurons = self.layers[-1].get_n_neurons())
            self.layers += activation,
        
    def set_verbose(self, verbose: bool) -> None:
        """Set verbose parameter.

        Args:
            verbose (bool): True to print training and prediction information.
        """
        self.verbose = verbose

    def set_loss(self, loss: str) -> None:
        """Set the Loss function.

        Args:
            loss (str): Loss functions: `'binary_cross_entropy'`, `'categorical_cross_entropy'`, `'mean_absolute_error'`, `'mean_squared_error'`
        """
        self.loss_str = loss

        if loss == 'mean_squared_error':
            self.loss = LossMeanSquaredError()
        if loss == 'mean_absolute_error':
            self.loss = LossMeanAbsoluteError()    
        if loss == 'binary_cross_entropy':
            self.loss = LossBinaryCrossentropy()
        if loss == 'categorical_cross_entropy':
            self.loss = LossCategoricalCrossentropy()

    def set_optimizer(self, optimizer: str) -> None:
        """Set the optimizer.

        Args:
            optimizer (str): Set the optimizer. Optimizers: `'sgd'`, `'adam'`.
        """
        self.optimizer_str = optimizer

        if optimizer == 'sgd':
            self.optimizer = OptimizerSGD()
        # if optimizer == 'adagrad':
        #     self.optimizer = OptimizerAdagrad()
        # if optimizer == 'rmsprop':
        #     self.optimizer = OptimizerRMSprop()
        if optimizer == 'adam':
            self.optimizer = OptimizerAdam()

    def set_accuracy(self, accuracy: str) -> None:
        """Set accuracy.

        Args:
            accuracy (str): Accuracy metrics: `'regression_threshold'`, `'regression_mae'`, `'categorical'`.
        """
        if accuracy == 'regression_threshold':
            self.accuracy = AccuracyRegressionThreshold()
            self.set_accuracy_parameters(dict(std_divisor=self.accuracy_regression_threshold_std_divisor))
        if accuracy == 'regression_mae':
            self.accuracy = AccuracyRegressionMAEr()
        if accuracy == 'categorical':
            if self.output_activation == 'sigmoid':
                self.accuracy = AccuracyCategorical(sigmoid=True)
            else:
                self.accuracy = AccuracyCategorical()

    def set_optimizer_parameters(self, parameters: dict) -> None:
        """Set optimizer parameters.

        Args:
            parameters (dict): Dictionary with parameters names as keys and paremeters values as values.
        """
        if self.optimizer is not None:
            self.optimizer = self.optimizer.__class__(**parameters)

    def set_accuracy_parameters(self, parameters: dict) -> None:
        """Set accuracy parameters.

        Args:
            parameters (dict): Dictionary with parameters names as keys and paremeters values as values.
        """
        if self.accuracy is not None:
            self.accuracy = self.accuracy.__class__(**parameters)

    def set_layer_parameter_by_index(self, layer_index: int, parameters: dict) -> None:
        """Set layer parameters.

        Args:
            layer_index (int): Index of the layer in the model layers' list to change the parameters.
            parameters (dict): Dictionary with parameters names as keys and paremeters values as values.
        """
        self.layers[layer_index] = self.layers[layer_index].__class__(**parameters)
        
    def get_params(self) -> dict:
        """Return the current model parameters. Useful in hyperparamter tunning.

        Returns:
            dict: Dictionary with parameter name as keys and parameters values as values.
        """
        params = dict(input_shape = self.input_shape,
            output_n_neurons = self.output_n_neurons,
            output_activation = self.output_activation,
            loss = self.loss_str,
            optimizer = self.optimizer_str,
            epochs = self.epochs, 
            batch_size = self.batch_size, 
            n_layers = self.n_layers, 
            n_neurons = self.n_neurons, 
            learning_rate = self.learning_rate, 
            decay = self.decay, 
            epsilon = self.epsilon, 
            beta_1 = self.beta_1, 
            beta_2 = self.beta_2, 
            momentum = self.momentum, 
            add_dropout = self.add_dropout, 
            dropout_rate = self.dropout_rate, 
            weights_l1_regularizer = self.weights_l1_regularizer, 
            weights_l2_regularizer = self.weights_l2_regularizer, 
            biases_l1_regularizer = self.biases_l1_regularizer, 
            biases_l2_regularizer = self.biases_l2_regularizer,
        )
        return params

    def compute_loss(self, output: np.ndarray, batch_y: np.ndarray, include_regularization: bool = True) -> Tuple[float, float, float]:
        """Compute data loss and regularization loss.

        Args:
            output (np.ndarray): Predicted target values; 2D array containing data with `float` or `integer` type. 
            batch_y (np.ndarray): Observed target in the batch; 2D array containing data with `float` or `integer` type.  
            include_regularization (bool, optional): If `True`, include regularization loss. Defaults to True.

        Returns:
            Tuple[float, float, float]: Loss (data loss + regularization loss), data loss (`~ y - y_hat`) and regularization loss.
        """
        data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=include_regularization)
        loss = data_loss + regularization_loss
        return (loss, data_loss, regularization_loss)

    def compute_accuracy(self, output: np.ndarray, batch_y: np.ndarray) -> float:
        """Compute accuracy.

        Args:
            output (np.ndarray): Predicted target values; 2D array containing data with `float` or `integer` type. 
            batch_y (np.ndarray): Observed target in the batch; 2D array containing data with `float` or `integer` type.  

        Returns:
            float: Accuracy as a `float`.
        """
        predictions = self.output_layer_activation.predictions(output)
        accuracy = self.accuracy.calculate(predictions, batch_y)
        return accuracy

    def optimizer_update(self) -> None:
        """Optimizer parameters update.
        """
        self.optimizer.pre_update_params()
        for layer in self.training_layers:
            self.optimizer.update_params(layer)
        self.optimizer.post_update_params()

    def compute_accumulated_loss(self, include_regularization: bool = True) -> float:
        """Calculate tha accumulated loss accross epochs.

        Args:
            include_regularization (bool, optional): If `True`, regularization loss is included. Defaults to True.

        Returns:
            float: Accuracy as a `float`.
        """
        accumulated_data_loss, regularization_loss = self.loss.calculate_accumulated(include_regularization=include_regularization)
        accumulated_loss = accumulated_data_loss + regularization_loss
        return accumulated_loss

    def reset_loss_accuracy(self) -> None:
        """Reset loss and accuracy accumulated values.
        """
        self.loss.reset_accumulated()
        self.accuracy.reset_accumulated()

    def build(self) -> None:
        """Build the model.
        """
        self.input_layer = LayerInput()
        layer_count = len(self.layers)
        self.training_layers = []
        for i in range(layer_count):
            if i == 0:
                self.layers[i].set_previous_layer(self.input_layer)
                self.layers[i].set_next_layer(self.layers[i + 1])          
            elif i < layer_count - 1:
                self.layers[i].set_previous_layer(self.layers[i - 1])
                self.layers[i].set_next_layer(self.layers[i + 1])
            else:                
                self.layers[i].set_previous_layer(self.layers[i - 1])
                self.layers[i].set_next_layer(self.loss)
                # Save the reference of the last layer, which its output is the model's output
                self.output_layer_activation = self.layers[i] 
            if hasattr(self.layers[i], 'weights'): 
                self.training_layers += self.layers[i],

        self.loss.set_training_layers(self.training_layers) 

        # Create an object of combined activation and loss function for faster gradient computation
        if isinstance(self.layers[-1], ActivationSoftmax) and isinstance(self.loss, LossCategoricalCrossentropy):
            self.softmax_classifier_output = LayerActivationSoftmaxLossCategoricalCrossentropy()

    def forward(self, X: np.ndarray, training: bool) -> np.ndarray:
        """Forward pass.

        Args:
            X (np.ndarray): Input data; 2D array containing data with `float` or `integer` type.
            training (bool): Indicating if it's in training. Useful for dropout layer because when not in training (i.e. during prediction). 

        Returns:
            np.ndarray: Output; 2D array containing data with `float` or `integer` type.
        """
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.previous_layer.output, training)
        return layer.output

    def backward(self, output: np.ndarray, y: np.ndarray) -> None:
        """Backward pass.

        Args:
            output (np.ndarray): Predicted target values; 2D array containing data with `float` or `integer` type. 
            y (np.ndarray): Observed target; 2D array containing data with `float` or `integer` type. 
        """
        if self.softmax_classifier_output is not None:
            # Call backward method on the combined activation/loss to set d_inputs
            self.softmax_classifier_output.backward(output, y)
            # So, we'll not call backward method of the last layer (Softmax)
            # We need to set d_inputs in this object using the d_inputs computed above
            self.layers[-1].d_inputs = self.softmax_classifier_output.d_inputs
            # Perform the backward pass going through all the objects but last, passing d_inputs from the next layer as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next_layer.d_inputs)
            return
        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next_layer.d_inputs)

    def verbose_training(self, epoch: int, step: int, print_every_n_steps: int, number_of_steps: int, accuracy: float, loss: float) -> None:
        """Print training information.

        Prints: current epoch, current step, accuracy and loss.

        Args:
            epoch (int): Current epoch.
            step (int): Current step.
            print_every_n_steps (int): Frequency to print.
            number_of_steps (int): Total number of steps.
            accuracy (float): Current accuracy.
            loss (float): Current loss.
        """
        if not step % print_every_n_steps or step == number_of_steps - 1:
            print(f'Epoch: {epoch} ' +  f'Step: {step} ' + f'Acc: {accuracy:.3f} ' + f'Loss: {loss:.3f}')

    def verbose_accuracy_loss(self, step: int, accuracy: float, loss: float) -> None:
        """Print accuracy and loss information.

        Args:
            step (int): Current step.
            accuracy (float): Current accuracy.
            loss (float): Current loss.
        """
        print(f'{step}-> ' + f'Acc: {accuracy:.3f} ' + f'Loss: {loss:.3f}')

    def fit(self, X: np.ndarray, y: np.ndarray, *, print_every_n_steps: int = 1, validation_data: Tuple[np.ndarray, np.ndarray] = None) -> None:
        """Fit the model.

        Args:
            X (np.ndarray): Input features; 2D array containing data with `float` or `integer` type.
            y (np.ndarray): True target values; 2D array containing data with `float` or `integer` type.
            print_every_n_steps (int, optional): At how many steps should accuracy and loss should be printed. Defaults to 1.
            validation_data (Tuple[np.ndarray, np.ndarray], optional): Tuple containing validation data X and y arrays. Defaults to None. 
            If None the training_data is used as validation data.
        """
        self.accuracy.initiate(y)
        training_steps = set_steps(X, self.batch_size)

        for epoch in range(1, self.epochs+1):
            self.reset_loss_accuracy()

            for step in range(training_steps):
                batch_X, batch_y = batch_split(X, step, self.batch_size), batch_split(y, step, self.batch_size)

                output = self.forward(batch_X, training=True)

                loss, data_loss, regularization_loss = self.compute_loss(output, batch_y, include_regularization=True)
                accuracy = self.compute_accuracy(output, batch_y)
               
                self.backward(output, batch_y)                
                self.optimizer_update()

                if self.verbose:
                    self.verbose_training(epoch, step, print_every_n_steps, training_steps, accuracy, loss)
                
            self.epoch_losses += loss,
            self.epoch_accuracies += accuracy,

            epoch_loss = self.compute_accumulated_loss(include_regularization=True)
            epoch_accuracy = self.accuracy.calculate_accumulated()
            self.accumulated_epoch_losses += epoch_loss,
            self.accumulated_epoch_accuracies += epoch_accuracy,

            if self.verbose:
                print('Epoch summary:')
                self.verbose_accuracy_loss(epoch, epoch_accuracy, epoch_loss)

            if validation_data is not None:
                if self.verbose:
                    print('Validation:')
                self.evaluate(*validation_data, batch_size=self.batch_size)

    def predict(self, X: np.ndarray, *, batch_size: int = None, type: str = 'classes') -> np.ndarray:
        """Predict method.

        Args:
            X (np.ndarray): Input features; 2D array containing data with `float` or `integer` type.
            batch_size (int, optional): Size of each batch. If None, each epoch will consist in one batch of the whole data. Defaults to None.
            type (str, optional): _description_. Defaults to 'classes'.

        Returns:
            np.ndarray: _description_
        """
        prediction_steps = set_steps(X, batch_size)
        output = []

        for step in range(prediction_steps):
            batch_X = batch_split(X, step, batch_size)
            batch_output = self.forward(batch_X, training=False)            
            output += batch_output,

        if type == 'classes':
            return self.layers[-1].predict_classes(np.vstack(output))

        return np.vstack(output)

    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray, *, batch_size: int = None) -> tuple[float, float]:
        """Evaluate model loss and accuracy on a given X, y data (usually validation and test sets).

        Args:
            X_val (np.ndarray): Input features; 2D array containing data with `float` or `integer` type.
            y_val (np.ndarray): True target values; 2D array containing data with `float` or `integer` type.
            batch_size (int, optional): Size of each batch. If None, each epoch will consist in one batch of the whole data.

        Returns:
            tuple[float, float]: Average loss in validation data, average accuracy in validation data.
        """
        validation_steps = set_steps(X_val, batch_size)

        self.reset_loss_accuracy()

        for step in range(validation_steps):
            batch_X, batch_y = batch_split(X_val, step, batch_size), batch_split(y_val, step, batch_size)

            output = self.forward(batch_X, training=False)

            self.loss.calculate(output, batch_y)

            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

            validation_loss, _ = self.loss.calculate_accumulated()
            validation_accuracy = self.accuracy.calculate_accumulated()
            
            if self.verbose:
                self.verbose_accuracy_loss(step, validation_accuracy, validation_loss)

        self.validation_epoch_losses += validation_loss,
        self.validation_epoch_accuracies += validation_accuracy,

        return validation_loss, validation_accuracy

    def get_layer_parameters(self) -> list[Tuple[np.ndarray, np.ndarray]]:
        """Return a list which each item is a tuple of weights array and biases array of each layer.

        Returns:
            list[Tuple[np.ndarray, np.ndarray]]: List which each item is a tuple of weights array and biases array of each layer.
        """
        return [layer.get_parameters() for layer in self.training_layers]

    def set_layer_parameters(self, parameters: list[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Set layer parameters.

        Args:
            parameters (list[Tuple[np.ndarray, np.ndarray]]): List which each item is a tuple of weights array and biases array of each layer.
        """
        for parameter_set, layer in zip(parameters, self.training_layers):
            layer.set_parameters(*parameter_set)

    def save_parameters(self, path: str) -> None:
        """Save model parameters (list[Tuple[np.ndarray, np.ndarray]]). 

        Args:
            path (str): Path to save.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.get_layer_parameters(), f)

    def load_parameters(self, path: str) -> None:
        """Load the list[Tuple[np.ndarray, np.ndarray]] of model parameters.

        Args:
            path (str): Path to load.
        """
        with open(path, 'rb') as f:
            self.set_layer_parameters(pickle.load(f))

    def save_model(self, path: str) -> None:
        """Save the model.

        Args:
            path (str): Path to save.
        """
        model = copy.deepcopy(self)

        model.reset_loss_accuracy()

        model.epoch_losses = []       
        model.epoch_accuracies = []
        model.accumulated_epoch_losses = []
        model.accumulated_epoch_accuracies = []
        model.validation_epoch_losses = []
        model.validation_epoch_accuracies = []

        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('d_inputs', None)

        for layer in model.layers:
            for attribute in ['inputs', 'output', 'd_inputs', 'd_weights', 'd_biases', \
                'weight_momentums', 'bias_momentums', 'weight_cache', 'bias_cache']:
                layer.__dict__.pop(attribute, None)

        with open(path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path: str):
        """Load a model object

        Args:
            path (str): Path to load.

        Returns:
            NeuralNetwork: NeuralNetwork model object
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model