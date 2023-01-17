import numpy as np 


class Activation:
    """Base activation function.

    The activation output equals to the activation input, so we use a Linear activation as template because of it's intrinsic meaning.

    """
    def __init__(self, n_neurons: int = None) -> None:
        """
        Args:
            n_neurons (`int`, optional): Set the number of neurons in the layer. Defaults to None.
        """
        self.n_neurons = n_neurons
        self.previous_layer = None
        self.next_layer = None  

    def forward(self, inputs: np.ndarray, training: bool) -> None: # training is for compatibility
        """Forward pass.
        
        Set outputs as equal to the inputs.

        Args:
            inputs (`np.ndarray`): Inputs; 2D array containing data with `float` or `integer` type.
            training (`bool`): For compatibility with other layers.
        """
        self.inputs = inputs
        self.output = inputs

    def backward(self, d_outputs: np.ndarray) -> None:
        """Backward pass.

        Derivative of the Linear activation is 1, so, because of the chain rule we have: d_outputs * 1 = d_outputs.

        Args:
            d_outputs (`np.ndarray`): Derivatives coming from the next layer (previous layer in backpropagation); 2D array containing data with `float` type.            
        """        
        self.d_inputs = d_outputs.copy()

    def predictions(self, outputs: np.ndarray) -> np.ndarray:
        """Post-proccessing for the outputs. Just for compatibility with other layers.

        Args:
            outputs (`np.ndarray`): Activation outputs; 2D array containing data with `float` or `integer` type.

        Returns:
            `np.ndarray`: Output; 2D array containing data with `float` or `integer` type.
        """
        return outputs

    # Predict classes (just for compatibility with others activations)
    def predict_classes(self, outputs: np.ndarray = None) -> np.ndarray:
        """Compatibility with other layers.

        Args:
            outputs (`np.ndarray`, optional): Activation outputs; 2D array containing data with `float` or `integer` type. Defaults to None.

        Returns:
            `np.ndarray`: Output; 2D array containing data with `float` or `integer` type.
        """
        if outputs is None:
            return self.output
        return outputs

    def get_n_neurons(self) -> int:
        """Returns the number of neurons in the layer.

        Returns:
            `int`: Number of neurons.
        """
        return self.n_neurons

    def set_previous_layer(self, layer) -> None: # layer is any layer class.
        """Set previous layer.
        """
        self.previous_layer = layer

    def set_next_layer(self, layer) -> None: # layer is any layer class.
        """Set next layer.
        """
        self.next_layer = layer


class ActivationLinear(Activation):
    """Linear activation function.

    The activation output equals to the activation input.

    `f(x) = x`.
    """


class ActivationReLU(Activation):
    """Rectified linear unit function.

    ReLU is a non-linear function where `f(x) = x if x > 0 else 0`.

    `f(x) = max(0, x)`.

    """
    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """Forward pass.

        Args:
            inputs (`np.ndarray`): Layer inputs; 2D array containing data with `float` or `integer` type.
            training (`bool`): For compatibility with other layers.
        """
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, d_outputs: np.ndarray) -> None:
        """Backward pass.

        For values greater than 0 the derivative of ReLU is 1 and for values lower than zero the derivative is 0.

        Because of the chain rule we have `d_outputs * 1 if d_output > 0 else 0`.

        Args:
            d_outputs (`np.ndarray`): Derivatives coming from the next layer (previous layer in backpropagation); 2D array containing data with `float` type. 
        """
        self.d_inputs = d_outputs.copy()        
        self.d_inputs[self.inputs <= 0] = 0 # Zero gradient where input values were negative


class ActivationSigmoid(Activation):
    """Sigmoid activation function.

    `f(x) = 1 / (1 + exp(-x))`.

    """
    def __init__(self, n_neurons: int = None, threshold: float = 0.5) -> None:
        """
        Args:
            n_neurons (int, optional): Set the number of neurons in the layer. Defaults to None.
            threshold (float, optional): Set the probability threshold. Defaults to 0.5.
        """
        self.n_neurons = n_neurons
        self.threshold = threshold
        self.previous_layer = None
        self.next_layer = None  

    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """Forward pass.

        Args:
            inputs (`np.ndarray`): Layer inputs; 2D array containing data with `float` or `integer` type.
            training (`bool`): For compatibility with other layers.
        """
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, d_outputs: np.ndarray) -> None:
        """Backward pass.

        The derivative of the sigmoid function is: `(1 - output) * output`. Where output is the output of the sigmoid function.

        Because of the chain rule we have `d_outputs * (1 - output) * output`.

        Args:
            d_outputs (`np.ndarray`): Derivatives coming from the next layer (previous layer in backpropagation); 2D array containing data with `float` type. 
        """
        # Derivative - calculates from output of the sigmoid function
        self.d_inputs = d_outputs * (1 - self.output) * self.output

    def predictions(self, outputs: np.ndarray) -> np.ndarray:
        """Post-proccessing for the outputs. Predict the class label index.

        Args:
            outputs (`np.ndarray`): Activation outputs; 2D array containing data with `float` or `integer` type.

        Returns:
            `np.ndarray`: Output; 2D array containing data with `float` or `integer` type.
        """
        return self.predict_classes(outputs)

    def predict_classes(self, outputs: np.ndarray = None, threshold: float = None) -> np.ndarray:
        """Predict the class label index.

        Args:
            outputs (np.ndarray, optional): Activation outputs; 2D array containing data with `float` or `integer` type. Defaults to None.
            threshold (float, optional): Probability threshold. Defaults to None.

        Returns:
            np.ndarray: Predicted class labels indexes; 2D array containing data with `float` or `integer` type.
        """
        if threshold is None:
            threshold = self.threshold
        if outputs is None:
            return (self.output > threshold) * 1
        return (outputs > threshold) * 1


class ActivationSoftmax(Activation):
    """Softmax activation function.

    `f(z)ij = exp(zij) / sum(exp(Zi))`, where i is the i-th sample and j is the j-th input in the input matrix (Z). 

    """
    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """Forward pass.

        Args:
            inputs (`np.ndarray`): Layer inputs; 2D array containing data with `float` or `integer` type.
            training (`bool`): For compatibility with other layers.
        """
        # Unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # subtract maximum to avoid exploding values because of the exp.
        # Normalize probabilities
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, d_outputs: np.ndarray) -> None:
        """Backward pass.

        In practice, this pass is not used because a much faster simplification of the derivative is implemented in the 
        `LayerActivationSoftmaxLossCategoricalCrossentropy`.

        S is the softmax output and Z is the softmax input. zik is k-th Softmax’s input of i-th sample.

        Kronecker delta function: `kdf(ij) = {1 if i = j else 0}`

        Softmax derivative:

        `dSij wrt dzik = Sij * kdf(jk) - Sij * Sik`.

        This forms a jacobian matrix for each i-th sample, so for each sample we will have:

        J = [[dSj wrt dzk ... dSj wrt dzkn] . . .
               
               [dSjn wrt dzk ... dSjn wrt dzkn]]

        Where n is n-th element relative to k or j.

        We need a Jacobian matrix because for each element j (Sj) in softmax output we have a partial derivative of Sj wrt each element
        in the input (zk ... zkn). 

        Then, we have a 3D array containing jacobians for each sample, being sample the first dimension.

        Args:
            d_outputs (np.ndarray): Derivatives coming from the next layer (previous layer in backpropagation); 2D array containing data with `float` type.
        """
        self.d_inputs = np.empty_like(d_outputs)
        
        for index, (single_output, single_d_outputs) in enumerate(zip(self.output, d_outputs)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and add it to the array of sample gradients; chain rule.
            self.d_inputs[index] = np.dot(jacobian_matrix, single_d_outputs)

    def predictions(self, outputs: np.ndarray) -> np.ndarray:
        """Post-proccessing for the outputs. Predict the class label index.

        Args:
            outputs (`np.ndarray`): Activation outputs; 2D array containing data with `float` or `integer` type.

        Returns:
            `np.ndarray`: Output; 2D array containing data with `float` or `integer` type.
        """
        return self.predict_classes(outputs)

    def predict_classes(self, outputs: np.ndarray = None) -> np.ndarray:
        """Predict the class label index.

        Args:
            outputs (np.ndarray, optional): Activation outputs; 2D array containing data with `float` or `integer` type. Defaults to None.

        Returns:
            np.ndarray: Predicted class labels indexes; 2D array containing data with `float` or `integer` type.
        """
        if outputs is None:
            return np.argmax(self.output, axis=1)
        return np.argmax(outputs, axis=1)