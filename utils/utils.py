from tkinter import filedialog
from numpy import ndarray
import numpy as np


def csv_to_matrix() -> ndarray:
    matrix = np.loadtxt(filedialog.askopenfilename(), delimiter=',', dtype=float)
    return matrix


def calculate_euclidean_distance(vector1 : ndarray , vector2 : ndarray) -> float:
    """
    Calculate the Euclidean distance between two vectors.

    Args:
        vector1 (ndarray): The first vector.
        vector2 (ndarray): The second vector.

    Returns:
        float: The Euclidean distance between the two vectors.
    """
    return np.sqrt(np.sum((vector1 - vector2) ** 2))