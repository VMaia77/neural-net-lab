import numpy as np
from typing import Tuple


class Loss:
    """Loss base class.
    """

    def regularization_loss(self) -> float:
        """Compute regularization loss by adding the L1 and L2 penalties weighted by the L1, L2 regularizers (`lambda`). 

        `L1: lambda * sum(|W|)`.

        `L2: lambda * sum(W * W)`. 

        Where `lambda` is the size of the penalty and W the weights. Same applies to biases. 

        Returns:
            float: Regularization loss.
        """
        regularization_loss = 0
        for layer in self.training_layers:
            # Weights L1 regularization
            if layer.weights_l1_regularizer > 0:
                regularization_loss += layer.weights_l1_regularizer * np.sum(np.abs(layer.weights))
            # Weights L2 regularization
            if layer.weights_l2_regularizer > 0:
                regularization_loss += layer.weights_l2_regularizer * np.sum(layer.weights * layer.weights)
            # Biases L1 regularization
            if layer.biases_l1_regularizer > 0:
                regularization_loss += layer.biases_l1_regularizer * np.sum(np.abs(layer.biases))
            # Biases L2 regularization
            if layer.biases_l2_regularizer > 0:
                regularization_loss += layer.biases_l2_regularizer * np.sum(layer.biases * layer.biases)
                
        return regularization_loss

    def set_training_layers(self, training_layers: list) -> None:
        """Set the training layers (layers with training paremeters, such as weights and biases).

        Args:
            training_layers (list): List containing the layers with training parameters.
        """
        self.training_layers = training_layers

    def calculate(self, output: np.ndarray, y: np.ndarray, *, include_regularization: bool = True) -> Tuple[float, float]:
        """Calculate loss and accumulated loss.

        Args:
            output (np.ndarray): Predicted value of the model; 2D array containing data with `float` or `integer` type.
            y (np.ndarray): Observed target; 2D array containing data with `float` or `integer` type.
            include_regularization (bool, optional): If True, includes regularization loss. Defaults to True.

        Returns:
            Tuple[float, float]: Data loss (`~ y - y_hat`) and regularization loss.
        """
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        if not include_regularization:
            return (data_loss, 0.0)
        return (data_loss, self.regularization_loss())

    def calculate_accumulated(self, *, include_regularization: bool = False) -> Tuple[float, float]:
        """Calculate accumulated loss.

        Args:
            include_regularization (bool): Bool indicating if regularization loss should be included.

        Returns:
            Tuple[float, float]: Returns the accumulted data loss (`~ y - y_hat`) and regularization loss.
        """
        data_loss = self.accumulated_sum / self.accumulated_count
        if not include_regularization:
            return data_loss, 0
        return data_loss, self.regularization_loss()

    def reset_accumulated(self) -> None:
        """Reset accumulated sum (of the losses) and counts of losses instances.
        """
        self.accumulated_sum = 0
        self.accumulated_count = 0


class LossMeanAbsoluteError(Loss):
    """Mean absolute error (MAE) - L1 loss. MAE is the absolute difference between the predicted and true values
    in a single output and average those absolute values.

    `MAEi = 1 / J * sumj(|yij - y_hatij|)`, where y means the target value, y_hat means predicted value, index i means the current sample, 
    index j means the current output in this sample, and the J means the number of outputs.
 
    Args:
        Loss (Loss): Loss main class
    """
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Forward pass.

        Args:
            y_pred (np.ndarray): Predicted target values; 2D array containing data with `float` or `integer` type.
            y_true (np.ndarray): True target values; 2D array containing data with `float` or `integer` type.

        Returns:
            np.ndarray: Losses for each sample.
        """
        # Average over the outputs (can be multiple outputs); axis=-1: compute along the last dimension, so returns avg losses for each sample.
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1) 
        return sample_losses

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """Backward pass.

        The derivative of MAE Loss of the sample i wrt y_pred (predicted value; y_hat) is 
        `{1 / J * a, a = 1 if y - y_hat > 0 else a = -1}`, 
        where J is the number of outputs in the i-th sample.

        Args:
            y_pred (np.ndarray): Predicted target values; 2D array containing data with `float` or `integer` type.
            y_true (np.ndarray): True target values; 2D array containing data with `float` or `integer` type.
        """
        samples = len(y_pred)        
        outputs = len(y_pred[0]) # Number of outputs in a sample, use the first sample to count
        # Gradients
        self.d_inputs = np.sign(y_true - y_pred) / outputs
        # Normalize gradient
        self.d_inputs = self.d_inputs / samples


# Mean Squared Error loss
class LossMeanSquaredError(Loss): # L2 loss
    """Mean squared error (MSE) - L2 loss. MSE is the squared difference between the predicted and true values
    in a single output and average those squared values.

    `MAEi = 1 / J * sumj((yij - y_hatij)^2)`, where y means the target value, y_hat means predicted value, index i means the current sample, 
    index j means the current output in this sample, and the J means the number of outputs.
 
    Args:
        Loss (Loss): Loss main class
    """
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """Forward pass.

        Args:
            y_pred (np.ndarray): Predicted target values; 2D array containing data with `float` or `integer` type.
            y_true (np.ndarray): True target values; 2D array containing data with `float` or `integer` type.

        Returns:
            np.ndarray: Losses for each sample.
        """
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1) # Average over the outputs (can be multiple outputs)
        return sample_losses

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """Backward pass.

        The derivative of MSE Loss of the sample i wrt y_pred (predicted value; y_hat) is `-2 / J * (y - y_hat)`, 
        where J is the number of outputs in the i-th sample.

        Args:
            y_pred (np.ndarray): Predicted target values; 2D array containing data with `float` or `integer` type.
            y_true (np.ndarray): True target values; 2D array containing data with `float` or `integer` type.
        """
        samples = len(y_pred)
        outputs = len(y_pred[0]) # Number of outputs in a sample, use the first sample to count
        # Gradients
        self.d_inputs = -2 * (y_true - y_pred) / outputs
        # Normalize gradient
        self.d_inputs = self.d_inputs / samples


# Binary cross-entropy loss
class LossBinaryCrossentropy(Loss):
    """Binary cross-entropy loss. Calculate the negative log-likelihood of the correct and incorrect classes, adding them together.

    `Loss = -yij * log(y_hatij) - (1 - yij) * log(1 - y_hatij)`, where y means the target value, y_hat means predicted value, 
    index i means the current sample, index j means the current output in this sample.

    Args:
        Loss (Loss): Loss main class
    """
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """Forward pass.

        Args:
            y_pred (np.ndarray): Predicted target values; 2D array containing data with `float` or `integer` type.
            y_true (np.ndarray): True target values; 2D array containing data with `float` or `integer` type.

        Returns:
            np.ndarray: Losses for each sample.
        """
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) # Clip data to prevent 0's and exploding when greater than 1
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1) # Average over the outputs (can be multiple outputs)
        return sample_losses
        
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """Backward pass.

        The derivative of Binary cross-entropy Loss of the sample i wrt y_pred (predicted value; y_hat) is 
        `-(y / y_hat - (1 - y) / (1 - y_hat)) / J`, where J is the number of outputs in the i-th sample.

        Args:
            y_pred (np.ndarray): Predicted target values; 2D array containing data with `float` or `integer` type.
            y_true (np.ndarray): True target values; 2D array containing data with `float` or `integer` type.
        """
        samples = len(y_pred)
        outputs = len(y_pred[0]) # Number of outputs in a sample, use the first sample to count
        clipped_y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7) # Clip data to prevent division by 0 and exploding when greater than 1
        # Gradients
        self.d_inputs = -(y_true / clipped_y_pred - (1 - y_true) / (1 - clipped_y_pred)) / outputs
        # Normalize gradient
        self.d_inputs = self.d_inputs / samples


# Cross-entropy loss
class LossCategoricalCrossentropy(Loss):
    """Categorical cross-entropy loss.

    `Loss = -sum(yij * log(y_hatij))`, where y means the target value, y_hat means predicted value, 
    index i means the current sample, index j means the current output in this sample.

    It simplifies to: `-log(y_hatik)`, where k is the index of the target label (ground-true label),

    Args:
        Loss (Loss): Loss main class
    """
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """Forward pass.

        Args:
            y_pred (np.ndarray): Predicted target values; 2D array containing data with `float` or `integer` type.
            y_true (np.ndarray): True target values; 2D array containing data with `float` or `integer` type.

        Returns:
            np.ndarray: Losses for each sample.
        """
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) # Clip data to prevent division by 0 and exploding when greater than 1
        # Probabilities for target values:
        # If categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]        
        # If one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """Backward pass.

        The derivative of Categorical cross-entropy Loss of the sample i wrt y_pred (predicted value; y_hat) is 
        `-(y / y_hat)`.

        Args:
            y_pred (np.ndarray): Predicted target values; 2D array containing data with `float` or `integer` type.
            y_true (np.ndarray): True target values; 2D array containing data with `float` or `integer` type.
        """
        samples = len(y_pred)
        labels = len(y_pred[0]) # Number of outputs in a sample, use the first sample to count
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Gradients
        self.d_inputs = -y_true / y_pred
        # Normalize gradient
        self.d_inputs = self.d_inputs / samples