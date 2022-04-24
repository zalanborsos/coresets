from sklearn.utils.extmath import row_norms
from utils.kmeans_init import _init_centroids

from coresets import Coreset
import numpy as np
from coresets import sensitivity


class KMeansCoreset(Coreset):
    """
    Class for generating k-Means coreset based on the sensitivity framework
    with importance sampling [1].

    Parameters
    ----------
        X : ndarray, shape (n_points, n_dims)
            The data set to generate coreset from.
        w : ndarray, shape (n_points), optional
            The weights of the data points. This allows generating coresets from a
            weighted data set, for example generating coreset of a coreset. If None,
            the data is treated as unweighted and w will be replaced by all ones array.
        n_clusters : int
            Number of clusters used for the initialization step.
        init : for avaiable types, please refer to sklearn.cluster.k_means_._init_centroids
            Method for initialization
        random_state : int, RandomState instance or None, optional (default=None)

    References
    ----------
        [1] Bachem, O., Lucic, M., & Krause, A. (2017). Practical coreset constructions
        for machine learning. arXiv preprint arXiv:1703.06476.
    """

    def __init__(self, X, w=None, n_clusters=10, init="k-means++", random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        super(KMeansCoreset, self).__init__(X, w, random_state)

    def calc_sampling_distribution(self):
        x_squared_norms = row_norms(self.X, squared=True)
        centers = _init_centroids(self.X, self.n_clusters, self.init, x_squared_norms=x_squared_norms,
                                  random_state=self.random_state)
        sens = sensitivity.kmeans_sensitivity(self.X, self.w, centers, max(np.log(self.n_clusters), 1))
        self.p = sens / np.sum(sens)


class KMeansLightweightCoreset(Coreset):
    """
       Class for generating k-Means coreset based on the importance sampling scheme of [1]

       Parameters
       ----------
           X : ndarray, shape (n_points, n_dims)
               The data set to generate coreset from.
           w : ndarray, shape (n_points), optional
               The weights of the data points. This allows generating coresets from a
               weighted data set, for example generating coreset of a coreset. If None,
               the data is treated as unweighted and w will be replaced by all ones array.
           random_state : int, RandomState instance or None, optional (default=None)

       References
       ----------
           [1] Bachem, O., Lucic, M., & Krause, A. (2017). Scalable and distributed
           clustering via lightweight coresets. arXiv preprint arXiv:1702.08248.
       """

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


class KMeansUniformCoreset(Coreset):
    """
       Class for generating uniform subsamples of the data.

       Parameters
       ----------
           X : ndarray, shape (n_points, n_dims)
               The data set to generate coreset from.
           w : ndarray, shape (n_points), optional
               The weights of the data points. This allows generating coresets from a
               weighted data set, for example generating coreset of a coreset. If None,
               the data is treated as unweighted and w will be replaced by all ones array.
           random_state : int, RandomState instance or None, optional (default=None)

       References
       ----------
           [1] Bachem, O., Lucic, M., & Krause, A. (2017). Scalable and distributed
           clustering via lightweight coresets. arXiv preprint arXiv:1702.08248.
       """

    def __init__(self, X, w=None, random_state=None):
        super(KMeansUniformCoreset, self).__init__(X, w, random_state)

    def calc_sampling_distribution(self):
        self.p = self.w
        self.p /= np.sum(self.p)
