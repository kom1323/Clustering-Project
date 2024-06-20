from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from utils import display_clustering
from utils import parameters
import math
from michal_algo_obj import MichalAlgorithm
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def run_michals_algorithm_and_graph(preprocessor, data, true_labels=None):

    algorithm_type = "New Algorithm"    

    unclustered_list = []

    #for k in range(parameters['k'] - 10, parameters['k'] - 6, 1):
    k = parameters['k']
    iteration = 1
    for b in np.arange(parameters['b'], parameters['b'] + 0.5, 0.05):
        for eps in np.arange(parameters['eps'], parameters['eps'] + 0.4, 0.05):
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



            num_unclustered = display_clustering(pipe, data, true_labels, algorithm_type, iteration=iteration)
            unclustered_list.append((iteration, num_unclustered))
    
    return unclustered_list



def find_parameters_general(preprocessor, data):

    iteration = 1
    for k in range(parameters['k'] - 5, parameters['k'] + 5, 1):
        for b in np.arange(parameters['b'], parameters['b'] + 0.5, 0.05):
            for eps in np.arange(parameters['eps'], parameters['eps'] + 0.4, 0.05):
                iteration += 1
                run_michals_algorithm_general(preprocessor, data, k, b, eps, iteration)


def run_michals_algorithm_general(preprocessor, data, k, b, eps, iteration):

    algorithm_type = "New Algorithm"
   
    clusterer = Pipeline([
        (algorithm_type, MichalAlgorithm(k=k, b=b, eps=eps, max_iter=30)),
    ])

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clusterer", clusterer)
    ])


    # Fit the pipeline to the data
    pipe.fit(data)

    # Check if the algorithm result is False
    if pipe["clusterer"][algorithm_type].result == False:
        return None

    # Transform the data using the preprocessor
    transformed_data = pipe["preprocessor"].transform(data)

    # Create a DataFrame with the transformed data
    pcadf = pd.DataFrame(transformed_data)

    # Get the cluster labels from the fitted algorithm
    predicted_labels = pipe["clusterer"][algorithm_type].labels_

    
    # Add the predicted labels to the DataFrame
    pcadf = pd.DataFrame(transformed_data)
    pcadf['predicted_cluster'] = predicted_labels

    # Calculate and print the percentage of unclustered points
    num_unclustered = np.sum(predicted_labels == -1)
    total_points = len(predicted_labels)
    percentage_unclustered = (num_unclustered / total_points) * 100

    print(f"Iteration #{iteration} Percentage of unclustered points: {percentage_unclustered:.2f}%")





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

    run_michals_algorithm_and_graph(preprocessor, data, true_labels)

    #find_parameters_general(preprocessor, data)