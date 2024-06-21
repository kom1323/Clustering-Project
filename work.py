from utils import *
from k_means import run_k_means_algorithm
from michals_algo import find_parameters_general, run_michals_algorithm_and_graph
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    
    
    dataset = fvecs_read(r"datasets/sift/sift_learn.fvecs")
    sub_dataset = dataset[:200, :2]
    draw_vectors(dataset)
    

    preprocessor = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        #("pca", PCA(n_components=128, random_state=42)),
    ]
    )

    n_clusters = 10

    #run_k_means_algorithm(preprocessor, sub_dataset, n_clusters)
    find_parameters_general(preprocessor, dataset)