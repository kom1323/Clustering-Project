import pandas as pd
import seaborn as sns


import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler



def display_k_means(pipe, data) -> None:

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
        "Clustering test k-means"
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    n_clusters = 3

    data, true_labels = make_blobs(
        n_samples=200,
        centers=n_clusters,
        cluster_std=2.75,
        random_state=42
    )


    preprocessor = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=2, random_state=42)),
    ]
    )


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

    display_k_means(pipe, data)


    