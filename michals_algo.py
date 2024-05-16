from utils import michals_algorithm
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from utils import display_clustering
from utils import parameters
import math

def run_michals_algorithm(preprocessor, data, true_labels):

    algorithm_type = "michals algorithm"    

    
    clusterer = Pipeline(
        [
            (
                algorithm_type,
                michals_algorithm(k=parameters["k"],
                                  b=parameters["b"],
                                  eps=parameters["eps"],
                                  sample_size=int(math.log(3 * parameters['k']) / parameters['eps'] + 1),
                                  sampled_data=data
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

    display_clustering(pipe, data, true_labels, algorithm_type, iteration=1)




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
