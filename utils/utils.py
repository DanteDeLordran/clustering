import random
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


def calculate_binary_centers(matrix: ndarray) -> tuple[list[float], list[float]]:
    c0 = []
    c1 = []

    for i in range(matrix.shape[1]):
        c0.append(min(matrix[:,i]))
        c1.append(max(matrix[:,i]))

    return c0, c1


def get_matrix_slope_centers(matrix : ndarray) -> tuple[list[float], list[float]]:

    """
    Returns the slope between the two centers of the given matrix
    Args:
        matrix: The original matrix

    Returns:
        m , b, c0, c1
    """

    m = []
    b = []

    c0 , c1 = calculate_binary_centers(matrix)

    for i in range(len(c0)):
        m.append(1/(c1[i]-c0[i]))
        b.append(-m[i]*c0[i])

    return m, b


def matrix_to_new_base_matrix(matrix : ndarray, m : list[float], b : list[float]) -> tuple[ndarray, list[int], list[int]] :

    """
    Defines the new base matrix given the original matrix and the slope centers.
    Args:
        matrix: The original matrix
        m: The slope of the original matrix between the centers
        b:

    Returns:
        matrix: The new base matrix
        c0: The new centers of the new base matrix
        c1: The new centers of the new base matrix
    """

    new_matrix = np.zeros((matrix.shape[0],matrix.shape[1]))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            new_matrix[i,j] = m[j] * matrix[i,j] + b[j]

    c0, c1 = calculate_binary_centers(new_matrix)

    return new_matrix, [int(i) for i in c0], [int(i) for i in c1]


def generate_n_sized_random_centers(matrix : ndarray, n : int):
    random_centers = np.zeros((n,matrix.shape[1]))
    for i in range(n):
        for j in range(matrix.shape[1]):
            random_centers[i,j] = random.random()
    return random_centers


def get_max_distance_centers(num : int) -> tuple[float,...]:
    return tuple( i / (num - 1) for i in range(num))
