import numpy as np
import math
from class_offfense import Offense

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt



############################# DTW #########################
def dp(dist_mat):
    N, M = dist_mat.shape
    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1))
    cost_mat[1:, 0] = np.inf
    cost_mat[0, 1:] = np.inf

    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],               # match (0)
                cost_mat[i, j + 1],           # insertion (1)
                cost_mat[i + 1, j]]           # deletion (2)
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]

    # Strip infinity edges from cost_mat before returning
    cost = cost_mat[N, M] / (N + M)
    return  cost

def DTW(offense1, offense2):
    N = len(offense1.list_coords)
    M = len(offense2.list_coords)
    dist_mat = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            dist_mat[i, j] = custom_distance_between_points(offense1, i, offense2, j)

    cost = dp(dist_mat)
    return cost

def custom_distance_between_points(offense1, i, offense2, j):
    coords_d = np.sqrt((offense1.list_coords[i][0] - offense2.list_coords[j][0])**2 + 
                       (offense1.list_coords[i][1] - offense2.list_coords[j][1])**2)
    time_d = abs(offense1.list_time[i] - offense2.list_time[j])
    # Normalize
    time_d /= 248
    coords_d /= 170
    d = 0.6 * coords_d + 0.4 * time_d 
    return d





######################## K-means #########################

def custom_kmeans(data, n_clusters, max_iterations , custom_distance ):
    n_samples = len(data)
    labels = [-1 for i in data]
    # Initialize centroids randomly
    centroids = [int(np.random.uniform(40, n_samples)) for k in range(n_clusters)]
    centroids[0] = 26
    centroids[1] = 6
    centroids[2] = 36
    centroids[3] = 14
    for _ in range(max_iterations):
        # Assign each data point to the nearest centroid
        for i in data:
            most_closest = np.inf
            closest_centroid = labels[i]
            for c in centroids:
                tmp = custom_distance(i, c)
                if tmp < most_closest:
                    most_closest = tmp
                    closest_centroid = c
            labels[i] = closest_centroid
        
        new_centroids = []
        for c in centroids:
                indices_with_c = [index for index, value in enumerate(labels) if value == c]
                new_centroids.append(calculate_centroid(indices_with_c , custom_distance))

        # Check convergence
        if np.allclose(np.array(new_centroids), np.array(centroids)):
            print("stop_before_max_iteration")
            break
        
        centroids = new_centroids
    
    return labels, centroids

######################## Hierarchical Clustering ############################################
def hierarchical_clustering(dist_matrix, method='average'):
    """
    this function preforms hierarchical clustering on the list of offense
    :param dist_matrix: the distance matrix with DTW between offense
    :param method: the linkage method : average/single/complete
    :return: plots the dendrogram and returns the linkage matrix
    """
    linkage_matrix = linkage(dist_matrix, method=method)
    fig, ax = plt.subplots(figsize=(15,6))
    dendrogram(linkage_matrix)
    plt.show()
    return linkage_matrix

def choose_nclusters(linkage_matrix, n_clusters):
    """
    this function performs cluster assignment after the dendrogram analysis
    :param linkage_matrix: the linkage matrix which the hierarchical clystering returns
    :param n_clusters: the number of clusters decided
    :return: return the cluster assignment - array with the indices and their cluster
    """
    cluster_assignments = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
    return cluster_assignments



######################## SCORE ####################

def convert_to_clusters(labels):
    clusters = [[] for _ in range(max(labels) + 2)]
    
    for i, label in enumerate(labels):
        if label != -1:
            clusters[label].append(i)
    
    # Append -1 cluster at the end
    clusters[-1] = [i for i, label in enumerate(labels) if label == -1]
    
    return clusters


def calculate_centroid(cluster , distance):
    centroid = None
    min_distance_sum = float('inf')

    for i1 in cluster:
        distance_sum = sum(distance(i1, i2) for i2 in cluster)
        
        if distance_sum < min_distance_sum:
            min_distance_sum = distance_sum
            centroid = i1

    return centroid


def calculate_bcss_wcss_ratio(clusters, custom_distance , global_centroid):
    num_clusters = len(clusters)
    
    # Calculate WCSS and BCSS
    wcss = 0.0
    bcss = 0.0
    
    
    for i in range(num_clusters):
        centroid = calculate_centroid(clusters[i] , custom_distance)
        
        for point in clusters[i]:
            wcss += custom_distance(point, centroid)
            bcss += custom_distance(point, global_centroid)
    
    # Calculate the ratio
    ratio = bcss / wcss if wcss != 0 else float('inf')
    return ratio




from sklearn.metrics import silhouette_score

def calculate_silhouette_score(clusters, custom_distance=None):
    """
    Calculate the silhouette score for a clustering model.

    Parameters:
    clusters (list of lists): A list of clusters, where each cluster is a list of data points.
    custom_distance (function, optional): A custom distance metric. Default is None.

    Returns:
    float: The silhouette score for the clustering.
    """
    if custom_distance is None:
        raise ValueError("A custom distance function is required for silhouette score.")

    all_points = []
    labels = []
    
    for cluster_idx, cluster in enumerate(clusters):
        all_points.extend(cluster)
        labels.extend([cluster_idx] * len(cluster))
    
    distances = [[custom_distance(p1, p2) for p2 in all_points] for p1 in all_points]
    silhouette_avg = silhouette_score(distances, labels)
    
    return silhouette_avg