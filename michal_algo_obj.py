import numpy as np
import math
import time
from tqdm import tqdm

class MichalAlgorithm:
    def __init__(self,eps, k, b, max_iter):
        
        """
        Finds the clusters if the data is (k,b)-clusterable

        b - the max size of the cluster radius
        eps - precentage of points to remove for the data to be (k,2b)-clusterable
        sampled_size - algorithm randomly chosen ln(3k)/eps points
        max_iter - maxium number of times to perform the algorithm

        _result - True/False whether the algorithm succeeded
        _reps - the points chosen as the center of the clusters
        _radii - the radius of each cluster corresponding to _reps
        
        """
        
        self._b = b
        self._eps = eps
        self._k = k
        self._sample_size = int(math.log(3 * self._k) / self._eps + 1)
        print(f"Running (k = {k}, b = {b:.2f}, eps = {eps:.3f}, sample size = {self._sample_size})")
        self._max_iter = max_iter

        self._result = False
        self._reps = None
        self._labels = None



    @property
    def result(self):
        return self._result

    @property
    def cluster_centers_(self):
        return self._reps

    @property
    def b(self):
        return self._b
    
    @property
    def k(self):
        return self._k

    @property
    def eps(self):
        return self._eps

    @property
    def num_clusters(self):
        return self._num_clusters

    @property
    def radii(self):
        return self._radii

    def find_labels(self, points):
        labels = []
        for point in tqdm(points, desc="Assigning labels"):
            distances = [self.dist(point, centroid) for centroid in self._reps]
            closest_centroid = np.argmin(distances)
            if distances[closest_centroid] <= self._b:
                labels.append(closest_centroid)
            else:
                labels.append(-1)

        self._labels = np.array(labels)
    

    @property
    def labels_(self):
        return self._labels
    

    def dist(self,p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))


    def fit(self, X, y=None, sample_weight=None):
        start_time = time.time()  # Record the start time

        result = False
        iterations =  None
        for iters in range(self._max_iter):  # iterations
            reps = []
            for _ in range(self._k + 1):
                
                random_subset_indices = np.random.choice(len(X), self._sample_size, replace=False)
                random_subset = X[random_subset_indices]
                random_subset_list = random_subset.tolist()
                found_any_new_representative = False

                for p_sample in random_subset_list:
                    is_sample_new_representative = True                    
                    for rep in reps:
                        distance_point_to_rep = self.dist(p_sample, rep)
                        if distance_point_to_rep <= self._b:
                            is_sample_new_representative = False
                            break
                    if is_sample_new_representative:
                        reps.append(p_sample)
                        found_any_new_representative = True
                        break
                if not found_any_new_representative:
                    break
            if len(reps) < self._k + 1:
                result = True
                iterations = iters
                break
        if not result:
            return
            
        self._result = result
        self._reps = reps


        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time in seconds
        minutes, seconds = divmod(elapsed_time, 60)  # Convert to minutes and seconds
        print(f"Time taken for Michal's Algo (k={self._k}, b={self._b}, eps={self._eps}) for {iterations} iterations in {int(minutes)} minutes and {seconds:.2f} seconds")
        self.find_labels(X)
        print("DONE labeling")
        


        
