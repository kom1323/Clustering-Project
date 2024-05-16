import numpy as np


class MichalAlgorithmObject:
    def __init__(self, result, reps, radii ,b, eps, num_clusters):
        self._result = result
        self._reps = reps
        self._b = b
        self._eps = eps
        self._num_clusters = num_clusters
        self._radii = radii

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
    def eps(self):
        return self._eps

    @property
    def num_clusters(self):
        return self._num_clusters

    @property
    def radii(self):
        return self._radii

    @property
    def labels_(self, points):
        labels = []
        for point in points:
            labeled = False
            for i, centroid in enumerate(self.reps):
                distance = np.linalg.norm(np.array(point) - np.array(centroid))
                if distance <= self.radii[i]:
                    labels.append(i)
                    labeled = True
                    break
            if not labeled:
                labels.append(None)
        return np.array(labels)

        
