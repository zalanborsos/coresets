from sklearn.utils import check_random_state
from sklearn.cluster.k_means_ import _init_centroids
from sklearn.utils.extmath import row_norms
from algorithms import weighted_kmeans_
import numpy as np


class WeightedKMeans:
    """
    Implementation of k-Means that supports weighted data set. Since extending sklearn's k-Means is not
    straightforward, this implementation only relies on sklearn's _init_centroids, the E and M steps are
    implemented in C++ for efficiency.
    This implementation does not use any heuristics for improving performance.

    Parameters
    ----------
        The parameters are a subset of sklearn.cluster.k_means_.KMeans, please refer to it for more details.
    """

    def __init__(self, n_clusters=10, init="k-means++", n_init=3, n_iter=3, max_iter=10, tol=1e-4, random_state=None):
        self.init = init
        self.n_init = n_init
        self.n_iter = n_iter
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.tol = tol
        self.random_state = check_random_state(random_state)
        self.centers = None
        self.inertia = -1

    def fit(self, X, w=None):
        if w is None:
            w = np.ones(X.shape[0])
        elif X.shape[0] != w.shape[0]:
            raise ValueError("The number of weights must match the number of data points.")
        x_squared_norms = row_norms(X, squared=True)
        self.centers = None

        for it in range(self.n_iter):
            best_centers, best_inertia = None, -1
            for init_it in range(self.n_init):
                # initialization could be extended to consider weights
                centers = _init_centroids(X, self.n_clusters, self.init, random_state=self.random_state,
                                          x_squared_norms=x_squared_norms)
                assignment, inertia = weighted_kmeans_.assignment_inertia(X, centers)
                if best_inertia == -1 or w.dot(inertia) < best_inertia:
                    best_centers = centers
                    best_inertia = w.dot(inertia)

            centers = best_centers

            inertia = np.full((X.shape[0]), np.inf)

            for it in range(self.max_iter):
                # E-step
                assignment, new_inertia = weighted_kmeans_.assignment_inertia(X, centers)

                # M-step
                centers = weighted_kmeans_.update_centers(X, w, centers, assignment)

                if w.dot(inertia - new_inertia) <= self.tol:
                    break
                inertia = new_inertia

            if self.centers is None or w.dot(self.inertia - new_inertia) > 0:
                self.inertia = new_inertia
                self.centers = centers

    def predict(self, X):

        if self.centers is None:
            raise ValueError("You must fit the model before prediction.")

        if X.shape[1] != self.centers.shape[1]:
            raise ValueError("Incompatible data dimension.")

        return weighted_kmeans_.assignment_inertia(X, self.centers)
