from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from utils import display_clustering
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
MODEL_NAME = 'all-MiniLM-L6-v2'
model = SentenceTransformer(MODEL_NAME)

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def run_k_means_algorithm(preprocessor, data, n_clusters,  true_labels=None):

    algorithm_type = "kmeans"
    iterations = 1
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
                        max_iter=3,
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

        display_clustering(pipe, data, true_labels, n_clusters, algorithm_type, iteration=i)


if __name__ == "__main__":

    # n_clusters = 5

    # data, true_labels = make_blobs(
    #     n_samples=200,
    #     centers=n_clusters,
    #     cluster_std=1,
    #     random_state=42
    # )

    # # Step 1: Generate 1/10 of the original points as a uniform distribution
    # n_uniform = data.shape[0] // 10  # 1/10 of the original points

    # # Determine the range for the uniform distribution
    # data_min = np.min(data, axis=0)
    # data_max = np.max(data, axis=0)

    # # Generate uniform points
    # uniform_data = np.random.uniform(low=data_min, high=data_max, size=(n_uniform, data.shape[1]))

    # # Step 2: Concatenate the uniform points with the original data
    # new_data = np.vstack([data, uniform_data])

    # # No labels for uniform points, so we can create dummy labels if needed
    # # Concatenate the existing labels with dummy labels
    # new_labels = np.concatenate([true_labels, [-1] * n_uniform])


    #dataset = fvecs_read(r"datasets/sift/sift_base.fvecs")
    data_file = r"C:\Users\Omer\Desktop\workshop\ClusteringAlgorithm\My-code\datasets\requests\covid19-unrecognized-requests.csv"
    df = load_data(data_file)
    sentences = df['text'].tolist()
    # processing the sentences - remove some charchters and convert to lower case
    sentences = [sentence.strip('\r\n').lower() for sentence in sentences]
    # encoding from sentences to vectors
    new_data = model.encode(sentences)
    n_clusters = 68

    preprocessor = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        #("pca", PCA(n_components=2, random_state=42)),
    ]
    )

    run_k_means_algorithm(preprocessor, new_data, n_clusters)




   


    