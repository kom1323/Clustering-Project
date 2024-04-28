import numpy as np
import matplotlib.pyplot as plt
import math
#from torch.utils.tensorboard import SummaryWriter


#writer = SummaryWriter()
parameters = {
    'eps': 262 / 2352,
    'k': 65,
    'b': 0.8,
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


def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))



def find_accurate_parameters(sampled_data):
    sample_size = int(math.log(3 * parameters['k']) / parameters['eps'] + 1)

    results = {}
    centroids = {}
    for k in range(parameters['k'] - 10, parameters['k'] - 6, 1):
        for b in np.arange(parameters['b'] - 0.1, parameters['b'] + 0.1, 0.05):
            for eps in np.arange(parameters['eps'] - 0.1, parameters['eps'] + 0.1, 0.05):
                michals_algorithm(eps, k, b, results, sampled_data, centroids, sample_size)
                


def michals_algorithm(eps, k, b, results, sampled_data, centroids, sample_size):
    
    result = False
    for _ in range(parameters['num_iterations']):  # iterations
        reps = []
        for _ in range(k + 1):
            random_subset_indices = np.random.choice(len(sampled_data), sample_size, replace=False)
            random_subset = sampled_data[random_subset_indices]
            random_subset_list = random_subset.tolist()
            found_any_new_representative = False
            for p_sample in random_subset_list:
                is_sample_new_representative = True
                for rep in reps:
                    if dist(p_sample, rep) <= b:
                        is_sample_new_representative = False
                        break
                if is_sample_new_representative:
                    reps.append(p_sample)
                    found_any_new_representative = True
                    break
            if not found_any_new_representative:
                break
        if len(reps) < k + 1:
            print((k, b, eps), True, reps)
            result = True
            centroids[(k, b, eps)] = reps
            break
    if not result:
        print((k, b, eps), False)


    return result, reps

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