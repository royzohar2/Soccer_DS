import numpy as np
import math
from class_offfense import Offense
import open3d as o3d


############################# DTW #########################
def dp(dist_mat):
    N, M = dist_mat.shape 
    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf

    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],      # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j]]  # deletion (2)
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]

    # Strip infinity edges from cost_mat before returning
    cost = cost_mat[N, M] / (N + M)
    return  cost

def custom_distance_between_points(offense1 , i  , offense2 , j):
    coords_d = math.sqrt((offense1.list_coords[i][0] - offense2.list_coords[j][0])**2 + (offense1.list_coords[i][1] - offense2.list_coords[j][1])**2)
    time_d = abs(offense1.list_time[i] - offense2.list_time[j])
    if offense1.list_action_type[i] == offense2.list_action_type[j]:
        action_d = 0
    else:
        action_d = 1

    # normalize
    time_d = time_d / 248
    coords_d =coords_d / 170
    d = 0.5 * coords_d + 0.25*time_d + 0.25*action_d
    return d

def DTW(offense1 ,offense2):
    N = len(offense1.list_coords)
    M = len(offense2.list_coords)
    dist_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            dist_mat[i, j] = custom_distance_between_points(offense1 , i  , offense2 , j)
    cost = dp(dist_mat)
    return  cost



############################# O3D #########################

def plot_points_cloud(pc1, pc2):
    # Set visualization options
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Set different colors for the point clouds
    pc1.paint_uniform_color([1, 0, 0])  # Red color
    pc2.paint_uniform_color([0, 0, 1])  # Blue color

    # Add point clouds to the visualization
    vis.add_geometry(pc1)
    vis.add_geometry(pc2)

    # Start visualization
    vis.run()

    # Close the visualization window
    vis.destroy_window()



def O3D(offense1, offense2):
    # Extract relevant data from offense instances
    coords1 = np.array(offense1.list_coords)
    coords2 = np.array(offense2.list_coords)
    list_time1 = np.array(offense1.list_time)
    list_time2 = np.array(offense2.list_time)
    # Normalize
    coords1 = coords1 / 170
    coords2 = coords2 / 170
    list_time1 = list_time1 / 248
    list_time2 = list_time2 / 248
    # Create the point clouds by combining the stacked points with list_time
    point_cloud1 = np.hstack((coords1, list_time1.reshape(-1, 1)))
    point_cloud2 = np.hstack((coords2, list_time2.reshape(-1, 1)))

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(point_cloud1)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(point_cloud2)

    #plot_points_cloud(pcd1 , pcd2)

    distances1 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    mean_squared_distance1 = np.mean(distances1 ** 2)
    rmse_score1 = np.sqrt(mean_squared_distance1)

    distances2 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))
    mean_squared_distance2 = np.mean(distances2 ** 2)
    rmse_score2 = np.sqrt(mean_squared_distance2)


    rmse_score = 0.5* rmse_score1 + 0.5* rmse_score2
    
    return rmse_score

######################## K-means #########################

def custom_kmeans(data, n_clusters, max_iterations , custom_distance ):
    n_samples = len(data)
    labels = [-1 for i in data]
    # Initialize centroids randomly
    centroids = [int(np.random.uniform(0, n_samples)) for k in range(n_clusters)]
    
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
            break
        
        centroids = new_centroids
    
    return labels, centroids


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
