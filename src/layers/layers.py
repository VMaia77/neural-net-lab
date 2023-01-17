import numpy as np
from typing import Tuple


class LayerInput:
    """Input layer used as the first layer in the neural network. It's output is used as the next layer (first training layer) input.
    """
    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """
        Args:
            inputs (`np.ndarray`): Inputs; 2D array containing data with `float` or `integer` type.
            training (`bool`): For compatibility with other layers.
        """
        self.output = inputs

    def set_previous_layer(self, layer) -> None: # layer is any layer object.
        """Set previous layer.
        """
        self.previous_layer = layer

    def set_next_layer(self, layer) -> None: # layer is any layer object.
        """Set next layer.
        """
        self.next_layer = layer

    def get_n_neurons(self) -> int:
        """Returns the number of neurons in the layer.

        Returns:
            `int`: Number of neurons.
        """
        return self.n_neurons


class LayerDense(LayerInput):
    """Dense layer.

    `f(x) = XW + B`. Where X are the inputs, W the weights and B the biases.

    Args:
        LayerInput (LayerInput): Inherits LayerInput methods.
    """
    def __init__(self, n_inputs: int, n_neurons: int, \
        weights_l1_regularizer: float = 0, weights_l2_regularizer: float = 0, \
            biases_l1_regularizer: float = 0, biases_l2_regularizer: float = 0) -> None:
        """
        Args:
            n_inputs (int): Number of inputs neurons.
            n_neurons (int): Set the number of neurons in the layer. Defaults to None.
            weights_l1_regularizer (float, optional): Weights L1 regularizer. Defaults to 0.
            weights_l2_regularizer (float, optional): Weights L2 regularizer. Defaults to 0.
            biases_l1_regularizer (float, optional): Biases L1 regularizer. Defaults to 0.
            biases_l2_regularizer (float, optional): Biases L2 regularizer. Defaults to 0.
        """
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights = 0.01 * np.random.randn(self.n_inputs, self.n_neurons)
        self.biases = np.zeros((1, self.n_neurons))

        self.weights_l1_regularizer = weights_l1_regularizer
        self.weights_l2_regularizer = weights_l2_regularizer
        self.biases_l1_regularizer = biases_l1_regularizer
        self.biases_l2_regularizer = biases_l2_regularizer

        self.previous_layer = None
        self.next_layer = None  

    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """Forward pass.

        `f(X) = XW + B`. Where X are the inputs, W the weights and B the biases.

        Args:
            inputs (`np.ndarray`): Layer inputs; 2D array containing data with `float` or `integer` type.
            training (`bool`): For compatibility with other layers.
        """
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, d_outputs: np.ndarray) -> None:
        """Backward pass.

        X are the inputs, W the weights and B the biases.

        The derivative of outputs wrt W is X (inputs), because the output is W x X (+B), so the derivative of outputs wrt X is W.
        The derivative of outputs wrt B is 1 because it's a sum (XW + B).

        The derivative of regularizer L1 wrt W is `lambda * {1 if w > 0 else -1}` because it's an absolute function. The same applies to B.

        The derivative of L2 wrt W is `2 * lambda * W`. The same applies to B.

        `lambda` is the size of the penalty. 
        
        Args:
            d_outputs (np.ndarray): Derivatives coming from the next layer (previous layer in backpropagation); 2D array containing data with `float` type.
        """
        # Gradients on parameters
        self.d_weights = np.dot(self.inputs.T, d_outputs)
        self.d_biases = np.sum(d_outputs, axis=0, keepdims=True)
        # Gradients on regularization
        # L1 on weights
        if self.weights_l1_regularizer > 0:
            d_l1 = np.ones_like(self.weights)
            d_l1[self.weights < 0] = -1
            self.d_weights += self.weights_l1_regularizer * d_l1
        # L2 on weights
        if self.weights_l2_regularizer > 0:
            self.d_weights += 2 * self.weights_l2_regularizer * self.weights
        # L1 on biases
        if self.biases_l1_regularizer > 0:
            d_l1 = np.ones_like(self.biases)
            d_l1[self.biases < 0] = -1
            self.d_biases += self.biases_l1_regularizer * d_l1
        # L2 on biases
        if self.biases_l2_regularizer > 0:
            self.d_biases += 2 * self.biases_l2_regularizer * self.biases
        # Gradient on values
        self.d_inputs = np.dot(d_outputs, self.weights.T)

    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns layer parameters (W, B).

        Returns:
            Tuple[np.ndarray, np.ndarray]: W, B.
        """
        return self.weights, self.biases

    def set_parameters(self, weights: np.ndarray, biases: np.ndarray) -> None:
        """Set layer parameters (W, B).

        Args:
            weights (np.ndarray): W; 2D array containing data with `float` type.
            biases (np.ndarray): B; 2D array containing data with `float` type.
        """
        self.weights = weights
        self.biases = biases


class LayerDropout(LayerInput):
    """Dropout layer.
    
    Dropout is a regularization technique where some neurons are disabled during training. It prevents the model to become too 
    dependent on any neuron, reducing overfitting (neuron specializes in a specific instance). It also helps in co-adoption, which occurs 
    when neurons become dependent on others neurons values. Also deals with noise by 
    sharing and homogeneizing neurons effects, allowing the model to learning more complex functions.

    Args:
        LayerInput (LayerInput): Inherits LayerInput methods.
    """
    def __init__(self, n_neurons: int = None, rate: float = 0.3) -> None:
        """
        Args:
            n_neurons (int, optional): Set the number of neurons in the layer. Defaults to None.
            rate (float, optional): Proportion of neurons to disable. Defaults to 0.3.
        """
        self.n_neurons = n_neurons
        # Store rate, we invert it as for example for dropout of 0.1 we need success rate of 0.9
        self.rate = 1 - rate
        self.previous_layer = None
        self.next_layer = None  

    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """Forward pass.

        `f(z, q) = z / (1 - q)`, where z is the input (and output) and q is the proportion of neurons to not disable.

        During training a rate of the neurons is disabled. 

        The disabling mask is standardized by the rate (divide the inputs by the rate, which is lower than 1, so the values increases)
        to compensate the values losses from the disabling, keeping the magnitude of the output values.

        The neurons are selected using a binomial distribution with `success_rate = 1 - proportion_to_disable`.

        Args:
            inputs (`np.ndarray`): Layer inputs; 2D array containing data with `float` or `integer` type.
            training (`bool`): Indicating if it's in training. Useful because when not in training (i.e. during prediction). 
            dropout shouldn't be used.
        """
        self.input = inputs
        # If not in the training mode -> return inputs
        if not training:
            self.output = inputs.copy()
            return
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    def backward(self, d_outputs: np.ndarray) -> None:
        """Backward pass.

        The derivative of the output wrt inputs is `{1 / (1 - q) if ri = 1 else 0}`, where ri is the random value sampled from the 
        binomial distribution (0, 1 ins this case) and q is the proportion of neurons to not disable.

        Args:
            d_outputs (np.ndarray): Derivatives coming from the next layer (previous layer in backpropagation); 2D array containing data with `float` type.
        """
        self.d_inputs = d_outputs * self.binary_mask


class LayerActivationSoftmaxLossCategoricalCrossentropy:
    """This layer is the faster simplification of the `dLoss` wrt the inputs of the softmax. Note that the outputs of the softmax are the
    inputs of the categorical cross-entropy loss function.

    The formula is `dLi / dzik = dLi / dy_hatij * DSij / dzik`, where Li / dzik is the derivative of the loss function of the i-the sample
    wrt the i-th z input of the softmax; dy_hatij is the softmax output of the i-th sample of the j-th neuron. 

    The simplification occurs because dy_hatij = DSij, after simplification we achieve: `dLi / dzik = y_hatik - yik`. 
    """

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """Backward pass.

        `dLi / dzik = y_hatik - yik`. 

        Args:
            y_pred (np.ndarray): Predicted outputs = softmax outputs; 2D array containing data with `float` or `integer` type.
            y_true (np.ndarray): Observed y; 2D array containing data with `float` or `integer` type.
        """
        samples = len(y_pred)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.d_inputs = y_pred.copy()
        # Gradients
        self.d_inputs[range(samples), y_true] -= 1 # because of one hot encoding y_hatik - y_trueik ... -1
        # Normalize gradient
        self.d_inputs = self.d_inputs / samples