import numpy as np
from typing import Tuple

def ts_train_test_split(
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float
) -> Tuple[
    np.ndarray, 
    np.ndarray, 
    np.ndarray, 
    np.ndarray
]:
    
    """
    Splits time-series data arrays into training and testing sets.

    Parameters:
    - X (np.ndarray): A 3D matrix containing input features. Shape: (samples, time_steps, features)
    - y (np.ndarray): A 2D matrix containing output targets. Shape: (samples, targets)
    - test_size (float): The proportion of the dataset to be allocated to the testing set (e.g., 0.2 for 20%).

    Returns:
    - X_train (np.ndarray): Training set of the input features.
    - y_train (np.ndarray): Training set of the output targets.
    - X_test (np.ndarray): Testing set of the input features.
    - y_test (np.ndarray): Testing set of the output targets.

    Note:
    - The function assumes that the time-series data is ordered chronologically.
    """
    
    slice_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:slice_idx], X[slice_idx:]
    y_train, y_test = y[:slice_idx], y[slice_idx:]
    return X_train, y_train, X_test, y_test
