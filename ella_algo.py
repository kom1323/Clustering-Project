from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from utils import *
import numpy as np
import time
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


# Helper function to update centroids
def update_centroids(centroids, cluster_assignment, embeddings):
    new_centroids = []
    for i in range(len(centroids)):
        if not len(embeddings[cluster_assignment == i]) == 0:
            new_centroids.append(np.mean(embeddings[cluster_assignment == i], axis=0))

    return new_centroids

# Helper function to group sentences by clusters
def make_cluster_vectors(cluster_assignment, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for i, cluster in enumerate(cluster_assignment):
        if cluster != -1:
            clusters[cluster].append(i)  # Store the index of the vector in the cluster
    return clusters

def dynamic_means_clustering_cosine(embeddings, similarity_threshold, min_size, max_iterations=100):
    start_time = time.time()  # Start time count
    centroids = []
    cluster_assignment_prev = np.array([-1] * len(embeddings))
    
    for _ in tqdm(range(max_iterations), desc="Clustering Iterations"):  # Progress bar for iterations
        cluster_assignment = np.array([-1] * len(embeddings))
        shuffled_indices = np.random.permutation(len(embeddings))
        
        for i in shuffled_indices:
            emb = embeddings[i]
            
            if len(centroids) == 0:
                centroids.append(emb)
                cluster_assignment[i] = 0
            else:
                # Calculate cosine similarity between the current vector and all centroids
                similarities = cosine_similarity([emb], centroids)[0]
                max_sim = np.max(similarities)
                
                if max_sim < similarity_threshold:
                    centroids.append(emb)
                    cluster_assignment[i] = len(centroids) - 1
                else:
                    # Assign to the most similar centroid
                    max_ind = np.argmax(similarities)
                    cluster_assignment[i] = max_ind
        
        if np.array_equal(cluster_assignment_prev, cluster_assignment):
            break
        
        cluster_assignment_prev = cluster_assignment
        centroids = update_centroids(centroids, cluster_assignment, embeddings)

    clusters = make_cluster_vectors(cluster_assignment, centroids)

    # Handle clusters with fewer than min_size points
    unclustered_points = []
    valid_clusters = []
    valid_cluster_indices = []
    
    for i, cluster in enumerate(clusters):
        if len(cluster) >= min_size:
            valid_clusters.append(cluster)
            valid_cluster_indices.extend(cluster)
        else:
            unclustered_points.extend(cluster)

    valid_cluster_assignment = np.array([cluster_assignment[i] if i in valid_cluster_indices else -1 for i in range(len(cluster_assignment))])

    # Print out the results
    end_time = time.time()  # End time count
    elapsed_time_minutes = (end_time - start_time) / 60  # Convert time to minutes

    # Calculate silhouette score for valid clusters
    silhouette_start_time = time.time()  # Start time count for silhouette score
    if len(valid_clusters) > 1:  # Silhouette score requires at least 2 clusters
        silhouette_avg = silhouette_score(embeddings[valid_cluster_indices], valid_cluster_assignment[valid_cluster_indices])
    else:
        silhouette_avg = None  # Not enough clusters to calculate silhouette score
    silhouette_end_time = time.time()  # End time count for silhouette score

    # Calculate the number of clusters and percentage of unclustered points
    num_clusters = len(valid_clusters)
    percentage_unclustered = len(unclustered_points) / len(embeddings) * 100

    
    silhouette_elapsed_time_seconds = silhouette_end_time - silhouette_start_time  # Time for silhouette in seconds

    print(f"Clustering completed in {elapsed_time_minutes:.2f} minutes.")
    print(f"Number of clusters: {num_clusters}")
    if silhouette_avg is not None:
        print(f"Silhouette Score: {silhouette_avg:.4f}")
        print(f"Silhouette Score Calculation Time: {silhouette_elapsed_time_seconds:.2f} seconds")
    else:
        print("Silhouette Score: Not enough clusters to calculate.")
    print(f"Percentage of unclustered points: {percentage_unclustered:.2f}%")

    print("-----------------------------------------")

    return valid_clusters, unclustered_points, centroids




if __name__ == '__main__':
    dataset = fvecs_read(r"datasets/sift/sift_base.fvecs")
    
    # Apply Min-Max Scaling
    scaler = MinMaxScaler()
    dataset_scaled = scaler.fit_transform(dataset)

    iteration_values = [3]
    similarity_threshold_values = [0.75]

    # Define the percentage for min_size
    min_size_percentage = 0.001  # For example, 1% of the dataset size
    dataset_size = len(dataset_scaled)
    min_size = int(min_size_percentage * dataset_size)

    # Loop through different combinations of parameters
    for max_iter in iteration_values:
        for sim_threshold in similarity_threshold_values:
            print(f"\nTesting max_iterations={max_iter}, similarity_threshold={sim_threshold}, min_size={min_size}")
            clusters, unclustered_points, centroids = dynamic_means_clustering_cosine(
                dataset_scaled, similarity_threshold=sim_threshold, min_size=min_size, max_iterations=max_iter
            )