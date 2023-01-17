import numpy as np


class Accuracy:
    """Accuracy base class.
    """
    def __init__(self) -> None:
        """Start accumulated sum and accumulated count.
        """
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def calculate(self, predictions: np.ndarray, y: np.ndarray) -> float:
        """Average the error computed by the accuracy class which will inherit self.

        Add values to the accumulated sum and accumlated count.

        Args:
            predictions (`np.ndarray`): Predicted target values; 2D array containing data with `float` or `integer` type.
            y (`np.ndarray`): True target values; 2D array containing data with `float` or `integer` type.

        Returns:
            `float`: Accuracy as a `float`.
        """
        errors = self.compute_error(predictions, y)
        accuracy = np.mean(errors)
        self.accumulated_sum += np.sum(errors)
        self.accumulated_count += len(errors)
        return accuracy

    def calculate_accumulated(self) -> float:
        """Calculate accumulated accuracy.

        Returns:
            float: Accumulated accuracy.
        """
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy

    def reset_accumulated(self) -> None:
        """Reset accumulated sum and count.
        """
        self.accumulated_sum = 0
        self.accumulated_count = 0


class AccuracyRegressionThreshold(Accuracy):
    """Accuracy for regression using a threshold. 
    
    The threshold is a precision parameter computed as standard deviation of y divided by a divisor.

    The accuracy is measured as the proportion of the predictions which the error `|y - yhat|` is smaller then `sd(y) / divisor`

    Args:
        Accuracy (Accuracy): Accuracy base class.
    """
    def __init__(self, std_divisor: int = 100) -> None:
        """
        Args:
            std_divisor (`int`, optional): Divisor of the `sd(y)`. Defaults to 100.
        """        
        super().__init__()
        self.precision = None
        self.std_divisor = std_divisor

    def initiate(self, y: np.ndarray) -> None:
        """Initate or reset the precision.

        Args:
            y (`np.ndarray`): True target values; 2D array containing data with `float` or `integer` type.
        """
        self.precision = np.std(y) / self.std_divisor

    def compute_error(self, predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute the error. 

        It's a `f(|y - yhat|)`.

        Args:
            predictions (`np.ndarray`): Predicted target values; 2D array containing data with `float` or `integer` type.
            y (`np.ndarray`): True target values; 2D array containing data with `float` or `integer` type..

        Returns:
            `np.ndarray`: 2D array containing data with `bool` type, indicating if the absolute error is smaller than precision.
        """
        return np.absolute(predictions - y) < self.precision


class AccuracyRegressionMAEr(Accuracy):
    """Accuracy for regression using relative mean absolute error (MAE).

    Relative MAE (MAEr) because it's computed as `MAE / sd(y)`.

    The divisor is called as 'precision' for compatibility with other accuracies.

    Args:
        Accuracy (Accuracy): Accuracy base class.
    """
    def __init__(self) -> None:
        """Start precision and inherited attributes."""
        super().__init__()
        self.precision = None        

    def initiate(self, y: np.ndarray, reset: bool = False) -> None:
        """Initate or reset the precision (divisor).

        Args:
            y (`np.ndarray`): True target values; 2D array containing data with `float` or `integer` type.
            reset (bool, optional): Indicating if it's reseting. Defaults to False.
        """
        if self.precision is None or reset:
            self.precision = np.std(y)

    def compute_error(self, predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The error is computed as `(MAE / sd(y)) * -1`, so higher the value higher the accuracy.

        Args:
            predictions (`np.ndarray`): Predicted target values; 2D array containing data with `float` or `integer` type.
            y (`np.ndarray`): True target values; 2D array containing data with `float` or `integer` type.

        Returns:
            `np.ndarray`: Error; 2D array containing data with `float` type.
        """
        return -(np.absolute(predictions - y) / self.precision) 


class AccuracyCategorical(Accuracy):
    """Accuracy for classification. 

    Args:
        Accuracy (Accuracy): Accuracy base class.
    """
    def __init__(self, *, sigmoid: bool = False) -> None:
        """
        Args:
            sigmoid (`bool`, optional): Indicates if the activation function is sigmoid. Defaults to False.
        """
        super().__init__()
        self.sigmoid = sigmoid

    def initiate(self, y: np.ndarray) -> None:
        """Used for compatibility with other accuracies.

        Args:
            y (`np.ndarray`): True target values; 2D array containing data with `float` or `integer` type.
        """
        pass

    def compute_error(self, predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The error is computed as checking if y and yhat matches.

        Args:
            predictions (`np.ndarray`): Predicted target values; 2D array containing data with `float` or `integer` type.
            y (`np.ndarray`): True target values; 2D array containing data with `float` or `integer` type.

        Returns:
            `np.ndarray`: Indicating if predictions = true values; 2D array containing data with `bool` type.
        """
        if not self.sigmoid and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y