import json
import numpy as np
from numpy import ndarray
from utils.utils import (
    csv_to_matrix,
    calculate_euclidean_distance,
    generate_n_sized_random_centers,
    z_score_normalize_matrix, generate_dissimilarity_matrix
)


def get_best_two_center_kmeans(matrix : ndarray, first_col : ndarray, c0 : list[float], c1 : list[float]) :
    results = []
    ownerships = []
    dc_min = None
    is_better = True
    c0 = c0.copy()
    c1 = c1.copy()
    iterations = 1
    results.append(
        {
            f'Initial cluster centers' : {
                'Center C0' : f'{c0}',
                'Center C1' : f'{c1}'
            }
        }
    )
    while is_better:
        new_matrix = np.zeros((matrix.shape[0], 5))
        ownership_c0 = []
        ownership_c1 = []

        for i in range(matrix.shape[0]):
            distance_c0 = calculate_euclidean_distance(matrix[i,:], np.array(c0))
            new_matrix[i,0] = distance_c0
            distance_c1 = calculate_euclidean_distance(matrix[i,:], np.array(c1))
            new_matrix[i,1] = distance_c1
            new_matrix[i,2] = min(distance_c0, distance_c1)

            if new_matrix[i,2] == new_matrix[i,0]:
                new_matrix[i,3] = 1
                ownership_c0.append(f'{first_col[i]}')
            else:
                new_matrix[i,3] = 0

            if new_matrix[i,2] == new_matrix[i,1]:
                new_matrix[i,4] = 1
                ownership_c1.append(f'{first_col[i]}')
            else:
                new_matrix[i,4] = 0

        for i in range(len(c0)):
            c0[i] = ((np.sum(new_matrix[:,3] * matrix[:,i])) + c0[i]) / (np.sum(new_matrix[:,3]) + 1)
            c1[i] = ((np.sum(new_matrix[:,4] * matrix[:,i])) + c1[i]) / (np.sum(new_matrix[:,4]) + 1)

        results.append(
            {
                f'Iteration {iterations}' : {
                    'DC MIN' : f'{np.sum(new_matrix[:, 2])}',
                    'DC0' : f'{np.sum(new_matrix[:, 0])}',
                    'DC1' : f'{np.sum(new_matrix[:, 1])}',
                    'C0' : f'{np.sum(new_matrix[:,3])}',
                    'C1' : f'{np.sum(new_matrix[:,4])}',
                    'Center C0' : f'{c0}',
                    'Center C1' : f'{c1}'
                }
            }
        )
        ownerships.append(
            {
                f'Ownerships iteration: {iterations}' : {
                    'C0' : f'{ownership_c0}',
                    'C1' : f'{ownership_c1}'
                }
            }
        )

        iterations += 1

        if dc_min is None:
            dc_min = (sum(new_matrix[:, 2]))
        elif dc_min > (sum(new_matrix[:, 2])):
            dc_min = (sum(new_matrix[:, 2]))
        elif dc_min <= (sum(new_matrix[:, 2])):
            is_better = False

    final_clusters_distance = calculate_euclidean_distance(np.array(c0),np.array(c1))

    results.append(
        {
            f'Distance between final clusters' : f'{final_clusters_distance}',
        }
    )
    return results, ownerships


def get_best_three_center_kmeans(matrix : ndarray, first_col : ndarray, c0 : list[float], c1 : list[float], c2 : list[float]) :
    results = []
    ownerships = []
    dc_min = None
    is_better = True
    c0 = c0.copy()
    c1 = c1.copy()
    c2 = c2.copy()
    iterations = 1
    results.append(
        {
            f'Initial cluster centers' : {
                'Center C0' : f'{c0}',
                'Center C1' : f'{c1}',
                'Center C2' : f'{c2}'
            }
        }
    )
    while is_better:
        new_matrix = np.zeros((matrix.shape[0], 7))
        ownership_c0 = []
        ownership_c1 = []
        ownership_c2 = []
        for i in range(matrix.shape[0]):
            distance_c0 = calculate_euclidean_distance(matrix[i,:], np.array(c0))
            new_matrix[i,0] = distance_c0
            distance_c1 = calculate_euclidean_distance(matrix[i,:], np.array(c1))
            new_matrix[i,1] = distance_c1
            distance_c2 = calculate_euclidean_distance(matrix[i,:], np.array(c2))
            new_matrix[i,2] = distance_c2
            new_matrix[i,3] = min(distance_c0, distance_c1, distance_c2)

            if new_matrix[i,3] == new_matrix[i,0]:
                new_matrix[i,4] = 1
                ownership_c0.append(f'{first_col[i]}')
            else:
                new_matrix[i,4] = 0

            if new_matrix[i,3] == new_matrix[i,1]:
                new_matrix[i,5] = 1
                ownership_c1.append(f'{first_col[i]}')
            else:
                new_matrix[i,5] = 0

            if new_matrix[i,3] == new_matrix[i,2]:
                new_matrix[i,6] = 1
                ownership_c2.append(f'{first_col[i]}')
            else:
                new_matrix[i,6] = 0

        for i in range(len(c0)):
            c0[i] = ((np.sum(new_matrix[:,4] * matrix[:,i])) + c0[i]) / (np.sum(new_matrix[:,4]) + 1)
            c1[i] = ((np.sum(new_matrix[:,5] * matrix[:,i])) + c1[i]) / (np.sum(new_matrix[:,5]) + 1)
            c2[i] = ((np.sum(new_matrix[:,6] * matrix[:,i])) + c2[i]) / (np.sum(new_matrix[:,6]) + 1)

        results.append(
            {
                f'Iteration {iterations}' : {
                    'DC MIN' : f'{np.sum(new_matrix[:, 3])}',
                    'DC0' : f'{np.sum(new_matrix[:, 0])}',
                    'DC1' : f'{np.sum(new_matrix[:, 1])}',
                    'DC2' : f'{np.sum(new_matrix[:, 2])}',
                    'C0' : f'{np.sum(new_matrix[:,4])}',
                    'C1' : f'{np.sum(new_matrix[:,5])}',
                    'C2' : f'{np.sum(new_matrix[:,6])}',
                    'Center C0' : f'{c0}',
                    'Center C1' : f'{c1}',
                    'Center C2' : f'{c2}'
                }
            }
        )
        ownerships.append(
            {
                f'Ownerships iteration: {iterations}' : {
                    'C0' : f'{ownership_c0}',
                    'C1' : f'{ownership_c1}',
                    'C2' : f'{ownership_c2}'
                }
            }
        )

        iterations += 1

        if dc_min is None:
            dc_min = (sum(new_matrix[:, 3]))
        elif dc_min > (sum(new_matrix[:, 3])):
            dc_min = (sum(new_matrix[:, 3]))
        elif dc_min <= (sum(new_matrix[:, 3])):
            is_better = False

    final_clusters_distance_c0_c1 = calculate_euclidean_distance(np.array(c0),np.array(c1))
    final_clusters_distance_c0_c2 = calculate_euclidean_distance(np.array(c0),np.array(c2))
    final_clusters_distance_c1_c2 = calculate_euclidean_distance(np.array(c1),np.array(c2))

    results.append(
        {
            'Distance between final clusters C0-C1' : f'{final_clusters_distance_c0_c1}',
            'Distance between final clusters C0-C2' : f'{final_clusters_distance_c0_c2}',
            'Distance between final clusters C1-C2' : f'{final_clusters_distance_c1_c2}',
        }
    )

    return results, ownerships


def get_best_four_center_kmeans(matrix : ndarray, first_col : ndarray, c0 : list[float], c1 : list[float], c2 : list[float], c3 : list[float]) :
    results = []
    ownerships = []
    dc_min = None
    is_better = True
    c0 = c0.copy()
    c1 = c1.copy()
    c2 = c2.copy()
    c3 = c3.copy()
    iterations = 1
    results.append(
        {
            f'Initial cluster centers' : {
                'Center C0' : f'{c0}',
                'Center C1' : f'{c1}',
                'Center C2' : f'{c2}',
                'Center C3' : f'{c3}'
            }
        }
    )
    while is_better:
        new_matrix = np.zeros((matrix.shape[0], 9))
        ownership_c0 = []
        ownership_c1 = []
        ownership_c2 = []
        ownership_c3 = []
        for i in range(matrix.shape[0]):
            distance_c0 = calculate_euclidean_distance(matrix[i,:], np.array(c0))
            new_matrix[i,0] = distance_c0
            distance_c1 = calculate_euclidean_distance(matrix[i,:], np.array(c1))
            new_matrix[i,1] = distance_c1
            distance_c2 = calculate_euclidean_distance(matrix[i,:], np.array(c2))
            new_matrix[i,2] = distance_c2
            distance_c3 = calculate_euclidean_distance(matrix[i,:], np.array(c3))
            new_matrix[i,3] = distance_c3
            new_matrix[i,4] = min(distance_c0, distance_c1, distance_c2)

            if new_matrix[i,4] == new_matrix[i,0]:
                new_matrix[i,5] = 1
                ownership_c0.append(f'{first_col[i]}')
            else:
                new_matrix[i,5] = 0

            if new_matrix[i,4] == new_matrix[i,1]:
                new_matrix[i,6] = 1
                ownership_c1.append(f'{first_col[i]}')
            else:
                new_matrix[i,6] = 0

            if new_matrix[i,4] == new_matrix[i,2]:
                new_matrix[i,7] = 1
                ownership_c2.append(f'{first_col[i]}')
            else:
                new_matrix[i,7] = 0

            if new_matrix[i,4] == new_matrix[i,3]:
                new_matrix[i,8] = 1
                ownership_c3.append(f'{first_col[i]}')
            else:
                new_matrix[i,8] = 0


        for i in range(len(c0)):
            c0[i] = ((np.sum(new_matrix[:,5] * matrix[:,i])) + c0[i]) / (np.sum(new_matrix[:,5]) + 1)
            c1[i] = ((np.sum(new_matrix[:,6] * matrix[:,i])) + c1[i]) / (np.sum(new_matrix[:,6]) + 1)
            c2[i] = ((np.sum(new_matrix[:,7] * matrix[:,i])) + c2[i]) / (np.sum(new_matrix[:,7]) + 1)
            c3[i] = ((np.sum(new_matrix[:,8] * matrix[:,i])) + c3[i]) / (np.sum(new_matrix[:,8]) + 1)

        results.append(
            {
                f'Iteration {iterations}' : {
                    'DC MIN' : f'{np.sum(new_matrix[:, 4])}',
                    'DC0' : f'{np.sum(new_matrix[:, 0])}',
                    'DC1' : f'{np.sum(new_matrix[:, 1])}',
                    'DC2' : f'{np.sum(new_matrix[:, 2])}',
                    'DC3': f'{np.sum(new_matrix[:, 3])}',
                    'C0' : f'{np.sum(new_matrix[:,5])}',
                    'C1' : f'{np.sum(new_matrix[:,6])}',
                    'C2' : f'{np.sum(new_matrix[:,7])}',
                    'C3' : f'{np.sum(new_matrix[:,8])}',
                    'Center C0' : f'{c0}',
                    'Center C1' : f'{c1}',
                    'Center C2' : f'{c2}',
                    'Center C3' : f'{c3}'
                }
            }
        )
        ownerships.append(
            {
                f'Ownerships iteration: {iterations}' : {
                    'C0' : f'{ownership_c0}',
                    'C1' : f'{ownership_c1}',
                    'C2' : f'{ownership_c2}',
                    'C3' : f'{ownership_c3}'
                }
            }
        )

        iterations += 1

        if dc_min is None:
            dc_min = (sum(new_matrix[:, 4]))
        elif dc_min > (sum(new_matrix[:, 4])):
            dc_min = (sum(new_matrix[:, 4]))
        elif dc_min <= (sum(new_matrix[:, 4])):
            is_better = False

    final_clusters_distance_c0_c1 = calculate_euclidean_distance(np.array(c0),np.array(c1))
    final_clusters_distance_c0_c2 = calculate_euclidean_distance(np.array(c0),np.array(c2))
    final_clusters_distance_c0_c3 = calculate_euclidean_distance(np.array(c0),np.array(c3))
    final_clusters_distance_c1_c2 = calculate_euclidean_distance(np.array(c1),np.array(c2))

    results.append(
        {
            'Distance between final clusters C0-C1' : f'{final_clusters_distance_c0_c1}',
            'Distance between final clusters C0-C2' : f'{final_clusters_distance_c0_c2}',
            'Distance between final clusters C0-C3' : f'{final_clusters_distance_c0_c3}',
            'Distance between final clusters C1-C2' : f'{final_clusters_distance_c1_c2}',
        }
    )

    return results, ownerships


def save_results(filename: str, results: list[dict]):
    with open(filename, "w") as file:
        json.dump(results, file, indent=4)
    print(f"Results have been saved to '{filename}'")


def run():
    matrix, first_col = csv_to_matrix()
    normalized_matrix, c0, c1 = z_score_normalize_matrix(matrix)
    generate_dissimilarity_matrix(normalized_matrix, "z-dissimilarity_matrix.csv")
    results, ownerships = get_best_two_center_kmeans(normalized_matrix, first_col, c0, c1)
    save_results("z-results_kmeans_2.txt", results)
    save_results("z-ownerships_kmeans_2.txt", ownerships)

    three_centers = generate_n_sized_random_centers(matrix,3)
    c0 = [center for center in three_centers[0][:]]
    c1 = [center for center in three_centers[1][:]]
    c2 = [center for center in three_centers[2][:]]
    results, ownerships = get_best_three_center_kmeans(normalized_matrix, first_col, c0, c1, c2)
    save_results("z-results_kmeans_3.txt", results)
    save_results("z-ownerships_kmeans_3.txt", ownerships)

    four_centers = generate_n_sized_random_centers(matrix,4)
    c0 = [center for center in four_centers[0][:]]
    c1 = [center for center in four_centers[1][:]]
    c2 = [center for center in four_centers[2][:]]
    c3 = [center for center in four_centers[3][:]]
    results, ownerships = get_best_four_center_kmeans(normalized_matrix, first_col, c0, c1, c2, c3)
    save_results("z-results_kmeans_4.txt", results)
    save_results("z-ownerships_kmeans_4.txt", ownerships)


if __name__ == "__main__":
    run()
