import numpy as np
import matplotlib.pyplot as plt
import math

import pandas as pd
import seaborn as sns
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs')


parameters = {
    'eps': 500 / 2352,
    'k': 5,
    'b': 0.05,
    'num_iterations': 30
}




# for reading sift dataset
def fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv



def find_accurate_parameters(sampled_data):
    sample_size = int(math.log(3 * parameters['k']) / parameters['eps'] + 1)

    results = {}
    centroids = {}
    for k in range(parameters['k'] - 10, parameters['k'] - 6, 1):
        for b in np.arange(parameters['b'] - 0.1, parameters['b'] + 0.1, 0.05):
            for eps in np.arange(parameters['eps'] - 0.1, parameters['eps'] + 0.1, 0.05):
                michals_algorithm_example(eps, k, b, results, sampled_data, centroids, sample_size)
                
def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def michals_algorithm_example(eps, k, b, sampled_data, sample_size):
    
    result = False
    for _ in range(parameters['num_iterations']):  # iterations
        reps = []
        radii = []
        for _ in range(k + 1):
            
            random_subset_indices = np.random.choice(len(sampled_data), sample_size, replace=False)
            random_subset = sampled_data[random_subset_indices]
            random_subset_list = random_subset.tolist()
            found_any_new_representative = False

            for p_sample in random_subset_list:
                
                is_sample_new_representative = True
                max_point_to_rep_distance = 0
                
                for rep in reps:
                
                    distance_point_to_rep = dist(p_sample, rep)
                    if distance_point_to_rep > max_point_to_rep_distance:
                        max_point_to_rep_distance = distance_point_to_rep
                
                    if distance_point_to_rep <= b:
                        is_sample_new_representative = False
                        break
                if is_sample_new_representative:
                    reps.append(p_sample)
                    radii.append(max_point_to_rep_distance)
                    found_any_new_representative = True
                    break
            if not found_any_new_representative:
                break
        if len(reps) < k + 1:
            print((k, b, eps), True)
            result = True
            break
    if not result:
        print((k, b, eps), False)



def draw_vectors(vectors: np.ndarray) -> None:

    # Extract x and y coordinates
    x_values = [point[0] for point in vectors]
    y_values = [point[1] for point in vectors]


    # Create a scatter plot
    plt.scatter(x_values, y_values)

    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot')

    # Show the plot
    plt.show()




def display_clustering(pipe, data, true_labels, algorithm_type, iteration):

    
    pipe.fit(data)

    if algorithm_type == "New Algorithm":
        if pipe["clusterer"][algorithm_type].result == False:
            return


    pcadf = pd.DataFrame(
    pipe["preprocessor"].transform(data),
    columns=["component_1", "component_2"],
    )

    algorithm_labels = pipe["clusterer"][algorithm_type].labels_
    algorithm_cluster_centers = pipe["clusterer"][algorithm_type].cluster_centers_

    

    print("algo labels = ", algorithm_labels)

    for index in range(len(algorithm_labels)):
            if algorithm_labels[index] is None:
                algorithm_labels[index] = "Unclustered"


    pcadf["predicted_cluster"] = algorithm_labels
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
        f"Clustering test {algorithm_type}"
    )

    # Extract unique colors used in the scatter plot
    colors = list(set(tuple(color) for color in scat.get_children()[0].get_facecolors()))
    print(len(colors))
        

    # Find a point in the same cluster as the cluster center and use its color for the circle
    for cluster_label, cluster_center in enumerate(algorithm_cluster_centers):
        
        # Find any point in the same cluster and take it's color
        cluster_point_index = np.where(algorithm_labels == cluster_label)[0][0]
        cluster_point_color = scat.get_children()[0].get_facecolors()[cluster_point_index]
            
        #Now we find the radius of the circle
        #Find points in the same cluster
        cluster_points = pcadf[pcadf['predicted_cluster'] == cluster_label][['component_1', 'component_2']].values
        
        # Calculate distances from cluster center to all points in the cluster
        distances = cdist([cluster_center], cluster_points, 'euclidean')[0]
        
        # Find maximum distance
        max_distance = np.max(distances)

        # Plot circle around the cluster center with the same color as the cluster point
        circle = plt.Circle(cluster_center, radius=max_distance, edgecolor=cluster_point_color, facecolor='None')
        plt.gca().add_patch(circle)



   
    if algorithm_type == "New Algorithm":
         #Add algorithm legend that display parameters
        legend_labels = [f"$k$ = {pipe['clusterer'][algorithm_type].k}",
                            f"$b$ = {pipe['clusterer'][algorithm_type].b:.2f}",
                            f"$\\epsilon$ = {pipe['clusterer'][algorithm_type].eps:.2f}"]
        handles, labels = scat.get_legend_handles_labels()
        handles.extend([plt.Line2D([0], [0], label=label) for label in legend_labels])
        labels.extend(legend_labels)
        plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    else:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.tight_layout()

    

    # Write the plot to TensorBoard
    writer.add_figure('Fig1', plt.gcf(), global_step=iteration)

    # Close the plot to release memory
    plt.close()

    return algorithm_cluster_centers