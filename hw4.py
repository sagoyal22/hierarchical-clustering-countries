import csv
import geopandas
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram


def load_data(filepath):
    """
    Reads a CSV file and returns a list of dictionaries,
    where each dictionary represents a row in the dataset.
    """
    data = []
    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(dict(row))
    return data

def calc_features(row):
     """
    Converts a single country's dictionary into a 9-dimensional NumPy feature vector.

    Parameters:
        row (dict): A dictionary representing one country's data.

    Returns:
        np.ndarray: A NumPy array of shape (9,) and dtype float64.
    """
     features = np.array([
        float(row['child_mort']),
        float(row['exports']),
        float(row['health']),
        float(row['imports']),
        float(row['income']),
        float(row['inflation']),
        float(row['life_expec']),
        float(row['total_fer']),
        float(row['gdpp'])
    ], dtype=np.float64)

     return features


def hac(features, linkage_type):
    """
    Manual Hierarchical Agglomerative Clustering (single/complete linkage).

    Args:
        features (list[np.ndarray]): list of n vectors (shape (9,), dtype float64)
        linkage_type (str): "single" or "complete"

    Returns:
        np.ndarray: Z of shape (n-1, 4) where each row i is:
            [i_id, j_id, dist, new_size]
            - i_id, j_id: indices of merged clusters (i_id < j_id)
            - dist: linkage distance between them (float)
            - new_size: number of original points in the new cluster
        New cluster formed at step i has index n + i.
    """
    if linkage_type not in {"single", "complete"}:
        raise ValueError("linkage_type must be 'single' or 'complete'")

    X = np.vstack(features).astype(np.float64, copy=False)
    n = X.shape[0]
    if n <= 1:
        return np.zeros((0, 4), dtype=float)

    diffs = X[:, None, :] - X[None, :, :]
    D_points = np.sqrt(np.sum(diffs * diffs, axis=2))
    np.fill_diagonal(D_points, np.inf)

    clusters = {i: [i] for i in range(n)}
    active = list(range(n)) 
    Z = np.zeros((n - 1, 4), dtype=float)
    eps = 1e-12

    def cluster_distance(id_a, id_b):
        A, B = clusters[id_a], clusters[id_b]
        sub = D_points[np.ix_(A, B)]
        if linkage_type == "single":
            return np.min(sub)
        else: 
            return np.max(sub)

    for step in range(n - 1):
        best_dist = np.inf
        best_i = None
        best_j = None

        for ai in range(len(active)):
            i_id = active[ai]
            for aj in range(ai + 1, len(active)):
                j_id = active[aj]
                d = cluster_distance(i_id, j_id)
                if (d < best_dist - eps or
                    (abs(d - best_dist) <= eps and
                     (best_i is None or i_id < best_i or
                      (i_id == best_i and j_id < best_j)))):
                    best_dist = d
                    best_i, best_j = i_id, j_id

        i_id, j_id = (best_i, best_j) if best_i < best_j else (best_j, best_i)

        new_size = len(clusters[i_id]) + len(clusters[j_id])
        Z[step, 0] = i_id
        Z[step, 1] = j_id
        Z[step, 2] = float(best_dist)
        Z[step, 3] = float(new_size)

        new_id = n + step
        clusters[new_id] = clusters.pop(i_id) + clusters.pop(j_id)

        active = [cid for cid in active if cid not in (i_id, j_id)]
        active.append(new_id)
        active.sort()

    return Z


def fig_hac(Z, names):
    """
    Visualize hierarchical clustering with a dendrogram.

    Args:
        Z (np.ndarray): (n-1) x 4 linkage matrix produced by your hac().
        names (list[str]): country names (length n).

    Returns:
        matplotlib.figure.Figure: the created figure.
    """
    n = Z.shape[0] + 1
    labels = [str(name).strip() for name in names[:n]]

    fig = plt.figure(figsize=(12, 6))
    dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,   
        above_threshold_color="C0",
        color_threshold=None  
    )
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Country")
    plt.ylabel("Linkage distance")
    plt.tight_layout()
    return fig

import numpy as np

def normalize_features(features):
    """
    Normalizes the feature vectors using z-score normalization.

    Args:
        features (list of np.ndarray): 
            A list of NumPy arrays, each of shape (9,) and dtype float64.

    Returns:
        list of np.ndarray:
            A list of normalized NumPy arrays (same shape, dtype=float64).
    """
    X = np.vstack(features).astype(np.float64)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    std[std == 0] = 1.0

    X_norm = (X - mean) / std

    normalized_features = [X_norm[i, :] for i in range(X_norm.shape[0])]

    return normalized_features

       

def world_map(Z, names, K_clusters):
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    world['name'] = world['name'].str.strip()
    names = [name.strip() for name in names]

    world['cluster'] = np.nan

    n = len(names)
    clusters = {j: [j] for j in range(n)}

    for step in range(n-K_clusters):
        cluster1 = Z[step][0]
        cluster2 = Z[step][1]

        new_cluster_id = n + step

        clusters[new_cluster_id] = clusters.pop(cluster1) + clusters.pop(cluster2)

    for i, value in enumerate(clusters.values()):
        for val in value:
            world.loc[world['name'] == names[val], 'cluster'] = i

    world.plot(column='cluster', legend=True, figsize=(15, 10), missing_kwds={
        "color": "lightgrey",  
        "label": "Other countries"
    })

    plt.show()

if __name__ == "__main__":
    data = load_data("Country-data.csv")
    features = [calc_features(row) for row in data]
    names = [row["country"] for row in data]
    features_normalized = normalize_features(features)
    np.savetxt("output.txt", features_normalized)
    n = 20
    Z = hac(features[:n], linkage_type="complete")
    fig = fig_hac(Z, names[:n])
    plt.show()

    



