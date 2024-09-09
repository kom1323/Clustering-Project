import numpy as np
import matplotlib.pyplot as plt
import math
import time
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import math

MODEL_NAME = 'all-MiniLM-L6-v2'
model = SentenceTransformer(MODEL_NAME)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs')


# blob parameters
# parameters = {
#     'eps': 1 / 10,
#     'k': 5,
#     'b': 0.05,
#     'num_iterations': 15
# }


# SIFT PARAMERTERS
parameters = {
    'eps': 1 / 100,
    'k': 50,
    'b': 1.3,
    'num_iterations': 5
}




# for reading sift dataset
def fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv



# For reading SIFT groundtruth
def ivecs_read(filename, bounds=None):
    with open(filename, 'rb') as f:
        # Read the vector size (first 4 bytes)
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        vec_size = 1 * 4 + d * 4  # Size of each vector in bytes
        
        # Get the total number of vectors
        f.seek(0, 2)  # Move the pointer to the end of the file
        bmax = f.tell() // vec_size
        a = 1
        b = bmax

        if bounds is not None:
            if isinstance(bounds, int):
                b = bounds
            elif isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                a, b = bounds

        assert a >= 1
        if b > bmax:
            b = bmax

        if b == 0 or b < a:
            return np.array([])

        # Number of vectors to read
        n = b - a + 1

        # Move the pointer to the start of the vectors to read
        f.seek((a - 1) * vec_size, 0)

        # Read the vectors
        data = np.fromfile(f, dtype=np.int32, count=(d + 1) * n)
        data = data.reshape(n, d + 1)

        # Check if the first column (dimension of the vectors) is consistent
        assert np.all(data[:, 0] == d)

        # Return the vectors (excluding the first column)
        return data[:, 1:]


def draw_vectors(vectors: np.ndarray) -> None:

    # Extract x and y coordinates
    x_values = [point[0] for point in vectors]
    y_values = [point[1] for point in vectors]


    # Create a scatter plot
    plt.scatter(x_values, y_values)

    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot')

    # Show the plot
    plt.show()




def display_clustering(pipe, data, true_labels, n_clusters,algorithm_type, iteration):

    unclustered_counter = 0
    start_time = time.time()  # Record the start time

    pipe.fit(data)


    end_time = time.time()  # Record the end time

    elapsed_time = end_time - start_time  # Calculate elapsed time in seconds

    minutes, seconds = divmod(elapsed_time, 60)  # Convert to minutes and seconds
    print(f"Time taken for K-Means (k={n_clusters}) in Iteration #{iteration}: {int(minutes)} minutes and {seconds:.2f} seconds")



    if algorithm_type == "New Algorithm":
        if pipe["clusterer"][algorithm_type].result == False:
            return None
        


    # Transform the data using the preprocessor
    transformed_data = pipe["preprocessor"].transform(data)
    
    # Determine the number of features in the transformed data
    num_features = transformed_data.shape[1]

    # Create dynamic column names based on the number of features
    columns = [f"component_{i+1}" for i in range(num_features)]
    pcadf = pd.DataFrame(transformed_data, columns=columns)
    
    
    #pcadf = pd.DataFrame(transformed_data, columns=["component_1", "component_2"])


    algorithm_labels = pipe["clusterer"][algorithm_type].labels_
    algorithm_cluster_centers = pipe["clusterer"][algorithm_type].cluster_centers_

    unclustered_counter = np.sum(algorithm_labels == -1)

    pcadf["predicted_cluster"] = algorithm_labels
    pcadf["predicted_cluster"] = pcadf["predicted_cluster"].replace({-1: "Unclustered"})


    if true_labels is not None:
        pcadf["true_label"] = true_labels

    if num_features == 2:
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(12, 8))

        if true_labels is not None:
            scat = sns.scatterplot(
                x="component_1",
                y="component_2",
                s=50,
                data=pcadf,
                hue="predicted_cluster",
                style="true_label",
                palette="Set2",
            )
        else:
            scat = sns.scatterplot(
                x="component_1",
                y="component_2",
                s=50,
                data=pcadf,
                hue="predicted_cluster",
                palette="Set2",
            )

        scat.set_title(
            f"Clustering test {algorithm_type}"
        )

        # Extract unique colors used in the scatter plot
        colors = list(set(tuple(color) for color in scat.get_children()[0].get_facecolors()))        



        if algorithm_type == "kmeans":
                # Draw convex hulls around clusters
                for cluster_label in range(len(algorithm_cluster_centers)):
                    points = pcadf[pcadf['predicted_cluster'] == cluster_label][['component_1', 'component_2']].values
                    if len(points) > 2:  # ConvexHull requires at least 3 points
                        hull = ConvexHull(points)
                        for simplex in hull.simplices:
                            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
        else:    
            # Find a point in the same cluster as the cluster center and use its color for the circle
            for cluster_label, cluster_center in enumerate(algorithm_cluster_centers):
                # Find points in the same cluster
                cluster_points = pcadf[pcadf['predicted_cluster'] == cluster_label][['component_1', 'component_2']].values
                
                # Calculate distances from cluster center to all points in the cluster
                distances = cdist([cluster_center], cluster_points, 'euclidean')[0]
                
                # Find maximum distance
                max_distance = np.max(distances)

                # Find a point in the same cluster and take its color
                cluster_point_index = np.where(algorithm_labels == cluster_label)[0][0]
                cluster_point_color = scat.get_children()[0].get_facecolors()[cluster_point_index]

                # Plot circle around the cluster center with the same color as the cluster point
                circle = plt.Circle(cluster_center, radius=max_distance, edgecolor=cluster_point_color, facecolor='None')
                plt.gca().add_patch(circle)


    total_points = len(algorithm_labels)
    percentage_unclustered = (unclustered_counter / total_points) * 100

    # Calculate silhouette score if not all labels are -1
    if len(set(algorithm_labels)) > 1 and -1 not in set(algorithm_labels):
        silhouette_avg = silhouette_score(transformed_data, algorithm_labels)
    else:
        silhouette_avg = None

    # Calculate the median distance from points to their cluster centers
    
    # distances = []
    # for i, label in enumerate(algorithm_labels):
    #     if label != -1:
    #         point = transformed_data[i]
    #         center = algorithm_cluster_centers[label]
    #         distance = np.linalg.norm(point - center)
    #         distances.append(distance)

    # if distances:
    #     median_distance = np.median(distances)
    #     average_distance = np.mean(distances)
    # else:
    #     median_distance = np.nan
    #     average_distance = np.nan

   
    # Create a TensorBoard writer
    writer = SummaryWriter(f'logs/kmeans_clusters_{n_clusters}_iter_{iteration}')

    if num_features == 2:
        if algorithm_type == "New Algorithm":
            #Add algorithm legend that display parameters
            legend_labels = [f"$k$ = {pipe['clusterer'][algorithm_type].k}",
                                f"$b$ = {pipe['clusterer'][algorithm_type].b:.2f}",
                                f"$\\epsilon$ = {pipe['clusterer'][algorithm_type].eps:.2f}",
                                f"$unclustered percentage$ = {percentage_unclustered:.2f}%"]
            handles, labels = scat.get_legend_handles_labels()
            handles.extend([plt.Line2D([0], [0], label=label) for label in legend_labels])
            labels.extend(legend_labels)
            plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        else:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.tight_layout()

        # Write the plot to TensorBoard
        writer.add_figure(algorithm_type, plt.gcf(), global_step=iteration)

        # Close the plot to release memory
        plt.close()


    writer.add_scalar('Percentage of Unclustered Points (%)', percentage_unclustered, iteration)
        
    if silhouette_avg is not None:
        print("Silhouette Score: ", silhouette_avg, iteration)
        writer.add_scalar('Silhouette Score', silhouette_avg, iteration)
    else:
        writer.add_text('Silhouette Score', 'Not Applicable (Insufficient number of clusters)', iteration)

    # Log the median distance as text
    #writer.add_text('Median Distance to Cluster Center', f'Median distance: {median_distance:.4f}', iteration)
    #writer.add_text('Average Distance to Cluster Center', f'Average distance: {average_distance:.4f}', iteration)
    writer.add_text('Time Taken', f"{int(minutes)} minutes and {seconds:.2f} seconds", iteration)

    writer.close()


    return unclustered_counter


