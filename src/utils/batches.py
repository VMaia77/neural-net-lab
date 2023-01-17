import numpy as np


def calculate_number_of_steps(X: np.ndarray, batch_size: int) -> int:
    """Return the number of steps (floor division) based on the length (number of samples) of X and the size of each batch.

    Args:
        X (np.ndarray): Input data as a 2D array.
        batch_size (int): Number of samples in each batch.

    Returns:
        int: Number of steps
    """
    number_of_steps = len(X) // batch_size
    return number_of_steps


def set_steps(X: np.ndarray, batch_size: int) -> int:
    """Return the number of steps based on the length (number of samples) of X and the size of each batch. 
    It's the calculate_number_of_steps output + 1 if there are remaining samples else calculate_number_of_steps output.

    Args:
        X (np.ndarray): Input data as a 2D array.
        batch_size (int): Number of samples in each batch.

    Returns:
        int: Number of steps
    """
    number_of_steps = 1
    if batch_size is not None:
        number_of_steps = calculate_number_of_steps(X, batch_size)
        # If there are remaining data add a batch so the whole data is used.
        if batch_size * number_of_steps < len(X): 
            number_of_steps += 1
    return number_of_steps


def batch_split(data: np.ndarray, step: int, batch_size: int) -> np.ndarray:
    """Split the data based on the step and batch size.

    Args:
        data (np.ndarray): X or y data as 2D arrays.
        step (int): Current step.
        batch_size (int): batch size.

    Returns:
        np.ndarray: Data corresponding to the batch of the current step.
    """
    if batch_size is None:
        batch_ = data
    else:
        batch_ = data[batch_size*step:batch_size*(step+1)]
    return batch_
