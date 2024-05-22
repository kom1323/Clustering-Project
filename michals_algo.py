from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from utils import display_clustering
from utils import parameters
import math
from michal_algo_obj import MichalAlgorithm
import numpy as np

def run_michals_algorithm(preprocessor, data, true_labels):

    algorithm_type = "michals algorithm"    

    #for k in range(parameters['k'] - 10, parameters['k'] - 6, 1):
    k = parameters['k']
    iteration = 1
    for b in np.arange(parameters['b'], parameters['b'] + 0.5, 0.05):
        for eps in np.arange(parameters['eps'] - 0.1, parameters['eps'] + 0.1, 0.05):
            iteration += 1
            clusterer = Pipeline(
                [
                    (
                        algorithm_type,
                        MichalAlgorithm(k=k,
                                        b=b,
                                        eps=eps,
                                        max_iter=30
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



            display_clustering(pipe, data, true_labels, algorithm_type, iteration=iteration)




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

    run_michals_algorithm(preprocessor, data, true_labels)
