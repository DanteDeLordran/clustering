from tkinter import filedialog
from numpy import ndarray
import numpy as np


def csv_to_matrix() -> ndarray:
    """
    Takes a CSV file and turns it into a numpy matrix.

    Returns:
        ndarray: A numpy matrix filled with the CSV data.
    """
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


def get_max_distance_centers(num : int) -> tuple[float,...]:
    return tuple( i / (num - 1) for i in range(num))
