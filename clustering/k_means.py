import json

import numpy as np
from numpy import ndarray

from utils.utils import csv_to_matrix, calculate_euclidean_distance, get_matrix_slope_centers, matrix_to_new_base_matrix


def get_best_kmeans(matrix : ndarray, c0 : list[float], c1 : list[float]) -> list[dict] :
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
        iterations += 1

        if dc_min is None:
            dc_min = (sum(new_matrix[:, 2]))
        elif dc_min > (sum(new_matrix[:, 2])):
            dc_min = (sum(new_matrix[:, 2]))
        elif dc_min <= (sum(new_matrix[:, 2])):
            is_better = False

    return results


def run():
    matrix = csv_to_matrix()
    m, b  = get_matrix_slope_centers(matrix)
    new_matrix, c0, c1 = matrix_to_new_base_matrix(matrix, m, b)
    results = get_best_kmeans(new_matrix, c0, c1)
    print('Results')
    print(json.dumps(results, indent=4))
    with open('../results_kmeans.txt', 'w') as file:
        json.dump(results, file, indent=4)
        print("Results have been saved to 'results.txt'")


if __name__ == '__main__':
    run()