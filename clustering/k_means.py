import json
import numpy as np
from numpy import ndarray
from utils.utils import (
    csv_to_matrix,
    calculate_euclidean_distance,
    get_matrix_slope_centers,
    slope_normalize_matrix,
    generate_n_sized_random_centers,
    z_score_normalize_matrix
)


def calculate_new_matrix(matrix: ndarray, centers: list[ndarray]) -> ndarray:
    """Calculate the new matrix with distances and cluster assignments."""
    num_centers = len(centers)
    new_matrix = np.zeros((matrix.shape[0], num_centers + 2))  # Last two cols for min distance and cluster assignments

    for i in range(matrix.shape[0]):
        distances = [calculate_euclidean_distance(matrix[i, :], center) for center in centers]
        new_matrix[i, :num_centers] = distances
        new_matrix[i, num_centers] = min(distances)
        min_index = distances.index(new_matrix[i, num_centers])
        new_matrix[i, num_centers + 1] = min_index

    return new_matrix


def update_centers(matrix: ndarray, new_matrix: ndarray, centers: list[ndarray]) -> list[ndarray]:
    """Update the centers based on cluster assignments."""
    num_centers = len(centers)
    updated_centers = []

    for k in range(num_centers):
        cluster_mask = (new_matrix[:, -1] == k).astype(float)
        cluster_sum = np.sum(cluster_mask * matrix.T, axis=1)
        center = (cluster_sum + centers[k]) / (np.sum(cluster_mask) + 1)
        updated_centers.append(center)

    return updated_centers


def get_best_kmeans(matrix: ndarray, initial_centers: list[ndarray]) -> list[dict]:
    num_centers = len(initial_centers)
    centers = initial_centers.copy()
    results = []
    dc_min = None
    iterations = 1
    is_better = True

    while is_better:
        new_matrix = calculate_new_matrix(matrix, centers)
        centers = update_centers(matrix, new_matrix, centers)

        total_distance = np.sum(new_matrix[:, -2])
        cluster_counts = [np.sum(new_matrix[:, -1] == k) for k in range(num_centers)]

        results.append(
            {
                f"Iteration {iterations}": {
                    "DC MIN": f"{total_distance}",
                    **{f"C{k}": f"{cluster_counts[k]}" for k in range(num_centers)},
                }
            }
        )

        if dc_min is None or total_distance < dc_min:
            dc_min = total_distance
        else:
            is_better = False

        iterations += 1

    return results


def save_results(filename: str, results: list[dict]):
    with open(filename, "w") as file:
        json.dump(results, file, indent=4)
    print(f"Results have been saved to '{filename}'")


def run():
    matrix = csv_to_matrix()
    #m, b = get_matrix_slope_centers(matrix)
    #normalized_matrix, c0, c1 = slope_normalize_matrix(matrix, m, b)
    normalized_matrix, c0, c1 = z_score_normalize_matrix(matrix)

    results = get_best_kmeans(normalized_matrix, [c0, c1])
    save_results("../zresults_kmeans_2.txt", results)

    centers = generate_n_sized_random_centers(matrix, 3)
    results = get_best_kmeans(normalized_matrix, centers)
    save_results("../zresults_kmeans_3.txt", results)

    centers = generate_n_sized_random_centers(matrix, 4)
    results = get_best_kmeans(normalized_matrix, centers)
    save_results("../zresults_kmeans_4.txt", results)


if __name__ == "__main__":
    run()
