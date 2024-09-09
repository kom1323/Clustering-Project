from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from utils import display_clustering
from utils import parameters
import math
import time
from michal_algo_obj import MichalAlgorithm
import numpy as np
import pandas as pd
from k_means import run_k_means_algorithm
from sklearn.metrics import silhouette_score
from torch.utils.tensorboard import SummaryWriter

def run_michals_algorithm_and_graph(preprocessor, data, true_labels=None):

    algorithm_type = "New Algorithm"    

    unclustered_list = []

    #for k in range(parameters['k'] - 10, parameters['k'] - 6, 1):
    k = parameters['k']
    iteration = 1
    for b in np.arange(parameters['b'], parameters['b'] + 0.3, 0.05):
        for eps in np.arange(parameters['eps'], parameters['eps'] + 0.3, 0.05):
            iteration += 1
            clusterer = Pipeline(
                [
                    (
                        algorithm_type,
                        MichalAlgorithm(k=k,
                                        b=b,
                                        eps=eps,
                                        max_iter=parameters['num_iterations']
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

    # First run to load all the values to cache
    run_michals_algorithm_general(preprocessor, data, k=100, b=1, eps=0.03, iteration=-1)

    iteration = 1
    #for k in range(parameters['k'], parameters['k'] + 300, 50):
    for k in [100, 200]:
        for b in [2.5, 3, 3.5]:
            for eps in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03 ]:
                iteration += 1
                run_michals_algorithm_general(preprocessor, data, k, b, eps, iteration)


def run_michals_algorithm_general(preprocessor, data, k, b, eps, iteration):

    algorithm_type = "New Algorithm"
   
    clusterer = Pipeline([
        (algorithm_type, MichalAlgorithm(k=k, b=b, eps=eps, max_iter=parameters['num_iterations'])),
    ])

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clusterer", clusterer)
    ])

    

    # Fit the pipeline to the data
    pipe.fit(data)

    
    
    writer = SummaryWriter(f'logs/run-{iteration}_k-{k}_b-{b:.2f}_eps-{eps:.2f}')

    # Check if the algorithm result is True
    if pipe["clusterer"][algorithm_type].result == True:
        print("Algorithm result: ", True)
        writer.add_text('Algorithm Result', f"(k = {k}, b = {b:.2f}, eps = {eps:.3f}) True, Clusters found = {len(set(pipe['clusterer'][algorithm_type].labels_))}", iteration)
        # len(set(pipe['clusterer'][algorithm_type].labels_))
    else:
        print("Algorithm result: ", False)
        writer.add_text('Algorithm Result', f"(k = {k}, b = {b:.2f}, eps = {eps:.3f}) False", iteration)
        writer.close()
        print("-----------------------------")
        return


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

    #filter out unclustered points for silhouette calculations
    clustered_data = transformed_data[predicted_labels != -1]
    clustered_labels = predicted_labels[predicted_labels != -1]

    # Calculate the silhouette score
    print("Calculating silhoette score...")
    start_time = time.time()
    if len(set(predicted_labels)) > 1:  # At least two clusters are required
        silhouette_avg = silhouette_score(clustered_data, clustered_labels)
    else:
        silhouette_avg = -2  # In case there is only one cluster
    end_time = time.time()
    elapsed_time_minutes = (end_time - start_time) / 60
    print(f"DONE. Elapsed time: {elapsed_time_minutes:.2f} minutes")
    print("Silhouette score = ", silhouette_avg)

   
    print(f"Iteration #{iteration} Percentage of unclustered points: {percentage_unclustered:.2f}%, Clusters found: {len(set(pipe['clusterer'][algorithm_type].labels_))}")
    print("-----------------------------")
    writer.add_scalar('Percentage of Unclustered Points (%)', percentage_unclustered, iteration)
    writer.add_scalar('Silhouette Score', silhouette_avg, iteration)
    writer.close()



def calculate_error_on_test(test_data, test_labels, centroids, radius_b):
    pass




if __name__ == "__main__":

    n_clusters = 5

    data, true_labels = make_blobs(
        n_samples=200,
        centers=n_clusters,
        cluster_std=1.5,
        random_state=42
    )

    # Step 1: Generate 1/10 of the original points as a uniform distribution
    n_uniform = data.shape[0] // 20  # 1/10 of the original points

    # Determine the range for the uniform distribution
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)

    # Generate uniform points
    uniform_data = np.random.uniform(low=data_min, high=data_max, size=(n_uniform, data.shape[1]))

    # Step 2: Concatenate the uniform points with the original data
    new_data = np.vstack([data, uniform_data])

    # No labels for uniform points, so we can create dummy labels if needed
    # Concatenate the existing labels with dummy labels
    new_labels = np.concatenate([true_labels, [-1] * n_uniform])


    

    preprocessor = Pipeline(
    [
        ("scaler", MinMaxScaler()),
       # ("pca", PCA(n_components=2, random_state=42)),
    ]
    )

    run_michals_algorithm_and_graph(preprocessor, new_data, new_labels)
    run_k_means_algorithm(preprocessor, new_data, n_clusters, new_labels)
    #find_parameters_general(preprocessor, new_data)