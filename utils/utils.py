import random
from tkinter import filedialog
from numpy import ndarray
import numpy as np
import csv


def csv_to_matrix() -> ndarray:
    """
    Takes a CSV file and turns it into a numpy matrix.

    Returns:
        ndarray: A numpy matrix filled with the CSV data.
    """
    matrix = np.loadtxt(filedialog.askopenfilename(), delimiter=',', dtype=float, skiprows=1)
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


def slope_normalize_matrix(matrix : ndarray, m : list[float], b : list[float]) -> tuple[ndarray, list[int], list[int]] :

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


def z_score_normalize_matrix( matrix : ndarray ) -> tuple[ndarray, list[int], list[int]]:
    """

    Args:
        matrix:

    Returns:

    """
    u = np.mean(matrix)
    o = np.std(matrix)
    normalized_matrix = (matrix - u) / o

    with open("../z-score_matrix.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(normalized_matrix)
    print("z-scored matrix saved on csv")

    c0 , c1 = calculate_binary_centers(normalized_matrix)
    return normalized_matrix, [int(i) for i in c0], [int(i) for i in c1]


def generate_dissimilarity_matrix(matrix: np.ndarray, filename: str):
    """
    Generates a dissimilarity matrix using Euclidean distances and saves it to a CSV file.

    Args:
        matrix: The original matrix (each row is a sample, each column is a feature)
        filename: The name of the CSV file to save the dissimilarity matrix

    Returns:
        dissimilarity_matrix: The dissimilarity matrix
    """
    num_samples = matrix.shape[0]

    dissimilarity_matrix = np.zeros((num_samples, num_samples))

    for i in range(num_samples):
        for j in range(i, num_samples):
            distance = np.linalg.norm(matrix[i] - matrix[j])
            dissimilarity_matrix[i, j] = distance
            dissimilarity_matrix[j, i] = distance

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(dissimilarity_matrix)

    print("Saved dissimilarity matrix in csv")

    return dissimilarity_matrix