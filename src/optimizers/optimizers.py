import numpy as np
from src.layers.layers import LayerDense


class OptimizerSGD:
    """Stochastic gradient descent optimizer.
    """
    def __init__(self, learning_rate: float = 0.01, decay: float = 1e-3, momentum: float = 0.5) -> None:
        """Initiate the class.

        Args:
            learning_rate (float, optional): Optimizer learning rate. Defaults to 0.01.
            decay (float, optional): Optimizer decay. Defaults to 1e-3.
            momentum (float, optional): Optimizer momentum. Defaults to 0.5.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self) -> None:
        """Pre-update procedure.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer: LayerDense) -> None:
        """Update layer parameters.

        Args:
            layer (LayerDense): Dense layer (training layer), must have weights and biases attributes.
        """
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.d_weights
            layer.weight_momentums = weight_updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.d_biases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.d_weights
            bias_updates = -self.current_learning_rate * layer.d_biases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self) -> None:
        """Post-update procedure.
        """
        self.iterations += 1


class OptimizerAdagrad:
    """AdaGrad (adaptive gradient) optimizer.
    """

    def __init__(self, learning_rate: float = 1.0, decay: float = 0., epsilon: float = 1e-7) -> None:
        """Initiate the class.

        Args:
            learning_rate (float, optional): Optimizer learning rate. Defaults to 1.0
            decay (float, optional): Optimizer decay. Defaults to 0.
            epsilon (float, optional): Optimizer epsilon. Defaults to 1e-7.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self) -> None:
        """Pre-update procedure.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer: LayerDense) -> None:
        """Update layer parameters.

        Args:
            layer (LayerDense): Dense layer (training layer), must have weights and biases attributes.
        """
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.d_weights**2
        layer.bias_cache += layer.d_biases**2

        layer.weights += -self.current_learning_rate * layer.d_weights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.d_biases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self) -> None:
        """Post-update procedure.
        """
        self.iterations += 1


class OptimizerRMSprop:
    """RMSProp (Root Mean Square Propagation) optimizer.
    """

    def __init__(self, learning_rate: float = 0.001, decay: float = 0., epsilon: float = 1e-7, rho: float = 0.9) -> None:
        """Initiate the class.

        Args:
            learning_rate (float, optional): Optimizer learning rate. Defaults to 0.001.
            decay (float, optional): Optimizer decay. Defaults to 0..
            epsilon (float, optional): Optimizer epsilon. Defaults to 1e-7.
            rho (float, optional): Optimizer rho. Defaults to 0.9.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self) -> None:
        """Pre-update procedure.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer: LayerDense) -> None:
        """Update layer parameters.

        Args:
            layer (LayerDense): Dense layer (training layer), must have weights and biases attributes.
        """
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.d_weights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.d_biases**2

        layer.weights += -self.current_learning_rate * layer.d_weights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.d_biases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self) -> None:
        """Post-update procedure.
        """
        self.iterations += 1


class OptimizerAdam:
    """Adam (Adaptive Momentum) optimizer.
    """

    def __init__(self, learning_rate: float = 0.001, decay: float = 1e-3, epsilon: float = 1e-7, 
        beta_1: float = 0.9, beta_2: float = 0.999) -> None:
        """Initiate the class.

        Args:
            learning_rate (float, optional): Optimizer learning rate. Defaults to 0.001.
            decay (float, optional): Optimizer decay. Defaults to 1e-3.
            epsilon (float, optional): Optimizer epsilon. Defaults to 1e-7.
            beta_1 (float, optional): Optimizer beta1. Defaults to 0.9.
            beta_2 (float, optional): Optimizer beta2. Defaults to 0.999.
        """

        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self) -> None:
        """Pre-update procedure.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer: LayerDense) -> None:
        """Update layer parameters.

        Args:
            layer (LayerDense): Dense layer (training layer), must have weights and biases attributes.
        """
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.d_weights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.d_biases
 
        # Divide momentums and cache by 1-beta**step. As steps increases the divisor approximates 1, speeding up training in the initial 
        # stages before the matrixes warm up during multiple initial steps.
        # Iteration is 0 at first pass so we need to start with 1 
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.d_weights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.d_biases**2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self) -> None:
        """Post-update procedure.
        """
        self.iterations += 1