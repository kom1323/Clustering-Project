from utils import *
from k_means import run_k_means_algorithm
from michals_algo import run_michals_algorithm
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    
    
    dataset = fvecs_read(r"datasets/sift/sift_learn.fvecs")
    sub_dataset = dataset[:200, :2]
    draw_vectors(sub_dataset)
    

    preprocessor = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=2, random_state=42)),
    ]
    )

    n_clusters = 10

    #run_k_means_algorithm(preprocessor, sub_dataset, n_clusters)
    print(run_michals_algorithm(preprocessor, sub_dataset))