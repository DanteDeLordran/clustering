import json

import numpy as np
from numpy import ndarray

from utils.utils import csv_to_matrix, calculate_euclidean_distance, get_matrix_slope_centers, slope_normalize_matrix


def calculate_medoids(matrix : ndarray, dc0_row : int, dc1_row : int) -> tuple[list,list]:
    m1 = []
    m2 = []
    for i in range(matrix.shape[1]):
        m1.append(matrix[dc0_row, i])
        m2.append(matrix[dc1_row, i])
    return m1, m2


def get_best_kmedoids(matrix : ndarray, c0 : list[float], c1 : list[float]) -> list[dict] :
    results = []
    dc_min = None
    is_better = True
    c0 = c0.copy()
    c1 = c1.copy()
    iterations = 1
    m0_index = None
    m1_index = None

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


        if iterations == 1:
            for i in range(len(c0)):
                c0[i] = ((np.sum(new_matrix[:,3] * matrix[:,i])) + c0[i]) / (np.sum(new_matrix[:,3]) + 1)
                c1[i] = ((np.sum(new_matrix[:,4] * matrix[:,i])) + c1[i]) / (np.sum(new_matrix[:,4]) + 1)

            results.append( { f'Iteration {iterations}' : {'DC MIN' : f'{np.sum(new_matrix[:, 2])}', 'C0' : f'{np.sum(new_matrix[:,3])}', 'C1' : f'{np.sum(new_matrix[:,4])}' } })
        elif iterations % 2 == 0:
            filtered_c0_col = new_matrix[new_matrix[:,3] == 1]
            min_c0_index = np.where(new_matrix[:,3] == 1)[0][np.argmin(filtered_c0_col[:,0])]

            filtered_c1_col = new_matrix[new_matrix[:,4] == 1]
            min_c1_index = np.where(new_matrix[:,4] == 1)[0][np.argmin(filtered_c1_col[:,1])]

            m0, m1 = calculate_medoids(matrix, int(min_c0_index), int(min_c1_index))
            c0 = m0.copy()
            c1 = m1.copy()

            if m0_index is None and m1_index is None:
                m0_index = int(min_c0_index)
                m1_index = int(min_c1_index)
            elif m0_index == int(min_c0_index) and m1_index == int(min_c1_index):
                is_better = False

            results.append( { f'Iteration {iterations}' : {'DC MIN' : f'{np.sum(new_matrix[:, 2])}', 'M0' : f'{np.sum(new_matrix[:,3])}', 'M1' : f'{np.sum(new_matrix[:,4])}' } })
        else:
            for i in range(len(c0)):
                c0[i] = (np.sum(new_matrix[:, 3] * matrix[:, i])) / (np.sum(new_matrix[:, 3]))
                c1[i] = (np.sum(new_matrix[:, 4] * matrix[:, i])) / (np.sum(new_matrix[:, 4]))

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
    new_matrix, c0, c1 = slope_normalize_matrix(matrix, m, b)
    results = get_best_kmedoids(new_matrix, c0, c1)
    print('Results')
    print(json.dumps(results, indent=4))
    with open('../results_kmedoids.txt', 'w') as file:
        json.dump(results, file, indent=4)
        print("Results have been saved to 'results.txt'")


if __name__ == '__main__':
    run()