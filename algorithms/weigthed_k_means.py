from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
from sklearn.cluster.k_means_ import _init_centroids
from sklearn.utils.extmath import row_norms


class KMeans():
    def __init__(self, X, n_clusters=10, init="k-means++", n_init=5, random_state=None):
        self.X = X
        self.random_state = check_random_state(random_state)
        self.n_init = n_init
        self.init = init
        self.n_clusters = n_clusters

    def fit(self, X, w):
        x_squared_norms = row_norms(self.X, squared=True)
        for it in range(self.n_init):
            # k-means++ can be extended to consider weights
            centers = _init_centroids(self.X, self.n_clusters, self.init, random_state=self.random_state,
                                      x_squared_norms=x_squared_norms)

