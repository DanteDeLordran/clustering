import numpy as np
from numpy import ndarray
from utils.utils import csv_to_matrix, calculate_euclidean_distance
import json


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


def get_best_clustering(matrix : ndarray, c0 : list[float], c1 : list[float]) -> list[dict] :
    results = []
    dc_min = None
    is_better = True
    c0 = c0.copy()
    c1 = c1.copy()
    iterations = 1

    while is_better:
        new_matrix = np.zeros((matrix.shape[0], 5))

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                distance_c0 = calculate_euclidean_distance(matrix[i,:], np.array(c0))
                new_matrix[i,0] = distance_c0
                distance_c1 = calculate_euclidean_distance(matrix[i,:], np.array(c1))
                new_matrix[i,1] = distance_c1
                new_matrix[i,2] = min(distance_c0, distance_c1)
                new_matrix[i,3] = 1 if new_matrix[i,2] != new_matrix[i,1] else 0
                new_matrix[i,4] = 1 if new_matrix[i,2] != new_matrix[i,0] else 0

        for i in range(len(c0)):
            c0[i] = ((np.sum(new_matrix[:,3] * matrix[:,i])) + c0[i]) / (np.sum(new_matrix[:,3]) + 1)
            c1[i] = ((np.sum(new_matrix[:,4] * matrix[:,i])) + c1[i]) / (np.sum(new_matrix[:,4]) + 1)

        results.append( { f'Iteration {iterations}' : {'DC MIN' : f'{np.sum(new_matrix[:, 2])}', 'C0' : f'{np.sum(new_matrix[:,3])}', 'C1' : f'{np.sum(new_matrix[:,4])}' } })
        iterations = iterations + 1

        if dc_min is None:
            dc_min = (sum(new_matrix[:, 2]))

        elif dc_min > (sum(new_matrix[:, 2])):
            dc_min = (sum(new_matrix[:, 2]))
        elif dc_min <= (sum(new_matrix[:, 2])):
            is_better = False

    return results


def run():
    matrix = csv_to_matrix()
    m, b, c0, c1  = matrix_to_binary_centers(matrix)
    new_matrix, c0, c1 = matrix_to_new_base_matrix(matrix, m, b)
    results = get_best_clustering(new_matrix, c0, c1)
    print('Results')
    print(json.dumps(results, indent=4))
    with open('../results.txt', 'w') as file:
        json.dump(results, file, indent=4)
        print("Results have been saved to 'results.txt'")


if __name__ == '__main__':
    run()