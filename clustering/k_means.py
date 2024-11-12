import json

import numpy as np
from numpy import ndarray

from utils.utils import csv_to_matrix, calculate_euclidean_distance, get_matrix_slope_centers, matrix_to_new_base_matrix, generate_n_sized_random_centers


def get_best_two_center_kmeans(matrix : ndarray, c0 : list[float], c1 : list[float]) -> list[dict] :
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
                new_matrix[i,3] = 1 if new_matrix[i,2] == new_matrix[i,0] else 0
                new_matrix[i,4] = 1 if new_matrix[i,2] == new_matrix[i,1] else 0

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


def get_best_three_center_kmeans(matrix : ndarray, c0 : list[float], c1 : list[float], c2 : list[float]) -> list[dict] :
    results = []
    dc_min = None
    is_better = True
    c0 = c0.copy()
    c1 = c1.copy()
    c2 = c2.copy()
    iterations = 1

    while is_better:
        new_matrix = np.zeros((matrix.shape[0], 7))

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                distance_c0 = calculate_euclidean_distance(matrix[i,:], np.array(c0))
                new_matrix[i,0] = distance_c0
                distance_c1 = calculate_euclidean_distance(matrix[i,:], np.array(c1))
                new_matrix[i,1] = distance_c1
                distance_c2 = calculate_euclidean_distance(matrix[i,:], np.array(c2))
                new_matrix[i,2] = distance_c2
                new_matrix[i,3] = min(distance_c0, distance_c1, distance_c2)
                new_matrix[i,4] = 1 if new_matrix[i,3] == new_matrix[i,0] else 0
                new_matrix[i,5] = 1 if new_matrix[i,3] == new_matrix[i,1] else 0
                new_matrix[i,6] = 1 if new_matrix[i,3] == new_matrix[i,2] else 0

        for i in range(len(c0)):
            c0[i] = ((np.sum(new_matrix[:,4] * matrix[:,i])) + c0[i]) / (np.sum(new_matrix[:,4]) + 1)
            c1[i] = ((np.sum(new_matrix[:,5] * matrix[:,i])) + c1[i]) / (np.sum(new_matrix[:,5]) + 1)
            c2[i] = ((np.sum(new_matrix[:,6] * matrix[:,i])) + c2[i]) / (np.sum(new_matrix[:,6]) + 1)

        results.append( { f'Iteration {iterations}' : {'DC MIN' : f'{np.sum(new_matrix[:, 3])}', 'C0' : f'{np.sum(new_matrix[:,4])}', 'C1' : f'{np.sum(new_matrix[:,5])}', 'C2' : f'{np.sum(new_matrix[:,6])}' } })
        iterations += 1

        if dc_min is None:
            dc_min = (sum(new_matrix[:, 3]))
        elif dc_min > (sum(new_matrix[:, 3])):
            dc_min = (sum(new_matrix[:, 3]))
        elif dc_min <= (sum(new_matrix[:, 3])):
            is_better = False

    return results


def get_best_four_center_kmeans(matrix : ndarray, c0 : list[float], c1 : list[float], c2 : list[float], c3 : list[float]) -> list[dict] :
    results = []
    dc_min = None
    is_better = True
    c0 = c0.copy()
    c1 = c1.copy()
    c2 = c2.copy()
    c3 = c3.copy()
    iterations = 1

    while is_better:
        new_matrix = np.zeros((matrix.shape[0], 9))

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                distance_c0 = calculate_euclidean_distance(matrix[i,:], np.array(c0))
                new_matrix[i,0] = distance_c0
                distance_c1 = calculate_euclidean_distance(matrix[i,:], np.array(c1))
                new_matrix[i,1] = distance_c1
                distance_c2 = calculate_euclidean_distance(matrix[i,:], np.array(c2))
                new_matrix[i,2] = distance_c2
                distance_c3 = calculate_euclidean_distance(matrix[i,:], np.array(c3))
                new_matrix[i,3] = distance_c3
                new_matrix[i,4] = min(distance_c0, distance_c1, distance_c2)
                new_matrix[i,5] = 1 if new_matrix[i,4] == new_matrix[i,0] else 0
                new_matrix[i,6] = 1 if new_matrix[i,4] == new_matrix[i,1] else 0
                new_matrix[i,7] = 1 if new_matrix[i,4] == new_matrix[i,2] else 0
                new_matrix[i,8] = 1 if new_matrix[i,4] == new_matrix[i,3] else 0

        for i in range(len(c0)):
            c0[i] = ((np.sum(new_matrix[:,5] * matrix[:,i])) + c0[i]) / (np.sum(new_matrix[:,5]) + 1)
            c1[i] = ((np.sum(new_matrix[:,6] * matrix[:,i])) + c1[i]) / (np.sum(new_matrix[:,6]) + 1)
            c2[i] = ((np.sum(new_matrix[:,7] * matrix[:,i])) + c2[i]) / (np.sum(new_matrix[:,7]) + 1)
            c3[i] = ((np.sum(new_matrix[:,8] * matrix[:,i])) + c3[i]) / (np.sum(new_matrix[:,8]) + 1)

        results.append( { f'Iteration {iterations}' : {'DC MIN' : f'{np.sum(new_matrix[:, 4])}', 'C0' : f'{np.sum(new_matrix[:,5])}', 'C1' : f'{np.sum(new_matrix[:,6])}', 'C2' : f'{np.sum(new_matrix[:,7])}', 'C3' : f'{np.sum(new_matrix[:,8])}' } })
        iterations += 1

        if dc_min is None:
            dc_min = (sum(new_matrix[:, 4]))
        elif dc_min > (sum(new_matrix[:, 4])):
            dc_min = (sum(new_matrix[:, 4]))
        elif dc_min <= (sum(new_matrix[:, 4])):
            is_better = False

    return results


def run():
    matrix = csv_to_matrix()
    m, b  = get_matrix_slope_centers(matrix)
    new_matrix, c0, c1 = matrix_to_new_base_matrix(matrix, m, b)
    results = get_best_two_center_kmeans(new_matrix, c0, c1)

    with open('../results_kmeans_2.txt', 'w') as file:
        json.dump(results, file, indent=4)
        print("Results have been saved to 'results_kmeans_2.txt'")

    three_centers = generate_n_sized_random_centers(matrix,3)
    c0 = [center for center in three_centers[0][:]]
    c1 = [center for center in three_centers[1][:]]
    c2 = [center for center in three_centers[2][:]]

    results = get_best_three_center_kmeans(new_matrix, c0, c1, c2)
    with open('../results_kmeans_3.txt', 'w') as file:
        json.dump(results, file, indent=4)
        print("Results have been saved to 'results_kmeans_3.txt'")

    four_centers = generate_n_sized_random_centers(matrix,4)
    c0 = [center for center in four_centers[0][:]]
    c1 = [center for center in four_centers[1][:]]
    c2 = [center for center in four_centers[2][:]]
    c3 = [center for center in four_centers[3][:]]

    results = get_best_four_center_kmeans(new_matrix, c0, c1, c2, c3)
    with open('../results_kmeans_4.txt', 'w') as file:
        json.dump(results, file, indent=4)
        print("Results have been saved to 'results_kmeans_4.txt'")


if __name__ == '__main__':
    run()