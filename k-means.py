import pandas as pd
import seaborn as sns
import numpy as np
from utils import michals_algorithm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler



def display_clustering(pipe, data, cluster_algo) -> None:

    pipe.fit(data)

    pcadf = pd.DataFrame(
    pipe["preprocessor"].transform(data),
    columns=["component_1", "component_2"],
    )

    pcadf["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_
    pcadf["true_label"] = true_labels

    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(12, 8))

    scat = sns.scatterplot(
        x="component_1",
        y="component_2",
        s=50,
        data=pcadf,
        hue="predicted_cluster",
        style="true_label",
        palette="Set2",
    )

    scat.set_title(
        f"Clustering test {cluster_algo}"
    )

    # Extract unique colors used in the scatter plot
    colors = list(set(tuple(color) for color in scat.get_children()[0].get_facecolors()))
    print(len(colors))
        

    # #add the circles
    # for index, cluster_center in enumerate(pipe["clusterer"]["kmeans"].cluster_centers_):
    #     circle = plt.Circle(cluster_center, radius=0.5, edgecolor=colors[index], facecolor='None')
    #     plt.gca().add_patch(circle)



        # Find a point in the same cluster as the cluster center and use its color for the circle
    for i, cluster_center in enumerate(pipe["clusterer"]["kmeans"].cluster_centers_):
        cluster_label = i  # Cluster label corresponding to the cluster center
        
        # Find any point in the same cluster and take it's color
        cluster_point_index = np.where(pipe["clusterer"]["kmeans"].labels_ == cluster_label)[0][0]
        cluster_point_color = scat.get_children()[0].get_facecolors()[cluster_point_index]
        
       #now we find the radius of the circle

        # Find points in the same cluster
        cluster_points = pcadf[pcadf['predicted_cluster'] == cluster_label][['component_1', 'component_2']].values
        
        # Calculate distances from cluster center to all points in the cluster
        distances = cdist([cluster_center], cluster_points, 'euclidean')[0]
        
        # Find maximum distance
        max_distance = np.max(distances)


        # Plot circle around the cluster center with the same color as the cluster point
        circle = plt.Circle(cluster_center, radius=max_distance, edgecolor=cluster_point_color, facecolor='None')
        plt.gca().add_patch(circle)


    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.tight_layout()
    plt.show()


def run_michals_algorithm(preprocessor, data, true_labels):
    

    centroids = {}
    results = {}
    clusterer = Pipeline(
        [
            (
                "michal algorithm",
                michals_algorithm(
                   
                ),
            ),
        ]
    )

    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("clusterer", clusterer)
        ]
    )


def run_k_means_algorithm(preprocessor, data, true_labels):
    
    clusterer = Pipeline(
        [
            (
                "kmeans",
                KMeans(
                    n_clusters=n_clusters,
                    init="k-means++",
                    n_init=10,
                    max_iter=300,
                    random_state=42,
                ),
            ),
        ]
    )

    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("clusterer", clusterer)
        ]
    )

    display_clustering(pipe, data, "k_means")


if __name__ == "__main__":

    n_clusters = 3

    data, true_labels = make_blobs(
        n_samples=200,
        centers=n_clusters,
        cluster_std=1,
        random_state=42
    )


    preprocessor = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=2, random_state=42)),
    ]
    )

    run_k_means_algorithm(preprocessor, data, true_labels)




   


    