from sklearn.cluster.k_means_ import _init_centroids
from sklearn.utils.extmath import row_norms

from coresets import Coreset
import numpy as np
import sensitivity


class KMeansCoreset(Coreset):
    def __init__(self, X, w=None, n_clusters=10, init="k-means++", random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        super(KMeansCoreset, self).__init__(X, w, random_state)

    def calc_sampling_distribution(self):
        x_squared_norms = row_norms(self.X, squared=True)
        centers = _init_centroids(self.X, self.n_clusters, self.init, random_state=self.random_state,
                                  x_squared_norms=x_squared_norms)
        sens = sensitivity.kmeans_sensitivity(self.X, self.w, centers, max(np.log(self.n_clusters), 1))
        self.p = sens / np.sum(sens)


class KMeansLightweightCoreset(Coreset):
    def __init__(self, X, w=None, random_state=None):
        super(KMeansLightweightCoreset, self).__init__(X, w, random_state)

    def calc_sampling_distribution(self):
        weighted_data = self.X * self.w[:, np.newaxis]
        data_mean = np.sum(weighted_data, axis=0) / np.sum(self.w)
        self.p = np.sum((weighted_data - data_mean[np.newaxis, :]) ** 2, axis=1) * self.w
        if np.sum(self.p) > 0:
            self.p = self.p / np.sum(self.p) * 0.5 + 0.5 / np.sum(self.w)
        else:
            self.p = np.ones(self.n_samples) / np.sum(self.w)

        # normalize in order to avoid numerical errors
        self.p /= np.sum(self.p)


class KMeansUniform(Coreset):
    def __init__(self, X, w=None, random_state=None):
        super(KMeansUniform, self).__init__(X, w, random_state)

    def calc_sampling_distribution(self):
        self.p = np.ones(self.n_samples) / self.n_samples
        self.p /= np.sum(self.p)
