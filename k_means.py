from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from utils import display_clustering



def run_k_means_algorithm(preprocessor, data, n_clusters,  true_labels=None):

    algorithm_type = "kmeans"
    iterations = 10
    centroids = None

    for i in range(iterations):

        clusterer = Pipeline(
            [
                (
                    algorithm_type,
                    KMeans(
                        n_clusters=n_clusters,
                        init=(centroids if centroids is not None else 'k-means++'),
                        n_init=1,
                        max_iter=1,
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

        display_clustering(pipe, data, true_labels, algorithm_type, iteration=i)


if __name__ == "__main__":

    n_clusters = 5

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

    run_k_means_algorithm(preprocessor, data, n_clusters, true_labels)




   


    