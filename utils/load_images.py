import numpy as np
import cv2
import os


def load_image_dataset(dataset: str, path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load image dataset where the images of the same label are at a folder with the label index.

    Args:
        dataset (str): Name of the folder of the dataset, usually `'train'` and `'test'`.
        path (str): Path of the dataset, at the level above the `dataset` argument. 

    Returns:
        tuple[np.ndarray, np.ndarray]: Images as arrays.
    """
    # Check all directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))
    # Create lists for samples and labels
    X = []
    y = []
    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            # And append it and a label to the lists
            X += image,
            y += label,
            # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')


def create_image_data(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create train and test sets.

    Args:
        path (str): Path of the dataset, containing train and test folders.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X, y, X_test, y_test, each one as a 2D array.
    """
    X, y = load_image_dataset('train', path)
    X_test, y_test = load_image_dataset('test', path)
    return X, y, X_test, y_test