from sklearn.cluster.k_means_ import _init_centroids
from sklearn.utils.extmath import row_norms

from coresets import CoresetGenerator
import numpy as np
import sensitivity

class KMeansCoresetGenerator(CoresetGenerator):
    def __init__(self, X, n_clusters=10, init="k-means++", random_state=None):
        super(KMeansCoresetGenerator, self).__init__(X, random_state)
        self.n_clusters = n_clusters
        self.init = init
        self.calc_sampling_distribution()

    def calc_sampling_distribution(self):
        x_squared_norms = row_norms(self.X, squared=True)
        centers = _init_centroids(self.X, self.n_clusters, self.init, random_state=self.random_state,
                                  x_squared_norms=x_squared_norms)
        sens = sensitivity.kmeans_sensitivity(self.X, centers, max(np.log(self.n_clusters), 1))
        self.p = sens / np.sum(sens)


class KMeansLightweightCoresetGenerator(CoresetGenerator):
    def __init__(self, X, random_state=None):
        super(KMeansLightweightCoresetGenerator, self).__init__(X, random_state)
        self.calc_sampling_distribution()

    def calc_sampling_distribution(self):
        data_mean = np.mean(self.X, axis=0)
        self.p = np.sum((self.X - data_mean[np.newaxis, :]) ** 2, axis=1)
        if np.sum(self.p) > 0:
            self.p = self.p / np.sum(self.p) * 0.5 + 0.5 / self.n_samples
        else:
            self.p = np.ones(self.n_samples) / self.n_samples

        # normalize in order to avoid numerical errors
        self.p /= np.sum(self.p)


class KMeansUniformGenerator(CoresetGenerator):
    def __init__(self, X, random_state=None):
        super(KMeansUniformGenerator, self).__init__(X, random_state)
        self.calc_sampling_distribution()

    def calc_sampling_distribution(self):
        self.p = np.ones(self.n_samples) / self.n_samples
        self.p /= np.sum(self.p)
