import numpy as np
from tkinter import filedialog

from numpy import ndarray


def csv_to_matrix() -> ndarray:
    matrix = np.loadtxt(filedialog.askopenfilename(), delimiter=',', dtype=float)
    return matrix


def calculate_binary_centers(matrix: ndarray) -> tuple[list[float], list[float]]:
    c0 = []
    c1 = []

    for i in range(matrix.shape[1]):
        c0.append(min(matrix[:,i]))
        c1.append(max(matrix[:,i]))

    return c0, c1

def matrix_to_binary_centers(matrix : ndarray) -> tuple[list[float], list[float], list[float], list[float]] :
    m = []
    b = []

    c0 , c1 = calculate_binary_centers(matrix)

    for i in range(len(c0)):
        m.append(1/(c1[i]-c0[i]))
        b.append(-m[i]*c0[i])

    return m, b,c0, c1


def matrix_to_new_base_matrix(matrix : ndarray, m : list[float], b : list[float]) -> tuple[ndarray, list[int], list[int]] :
    new_matrix = np.zeros((matrix.shape[0],matrix.shape[1]))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            new_matrix[i,j] = m[j] * matrix[i,j] + b[j]

    c0, c1 = calculate_binary_centers(new_matrix)

    return new_matrix, [int(i) for i in c0], [int(i) for i in c1]


def main():
    matrix = csv_to_matrix()
    m, b, c0, c1  = matrix_to_binary_centers(matrix)
    new_matrix, c0, c1 = matrix_to_new_base_matrix(matrix, m, b)
    print('new matrix', new_matrix)
    print('c0', c0)
    print('c1', c1)


if __name__ == '__main__':
    main()