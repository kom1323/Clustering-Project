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
from utils import display_clustering




def run_michals_algorithm(preprocessor, data, true_labels):

    algorithm_type = "michals"    

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

    display_clustering(pipe, data, true_labels, algorithm_type)




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

    run_michals_algorithm(preprocessor, data, true_labels)
