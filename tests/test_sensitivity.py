from __future__ import division, absolute_import

import pytest
from sklearn.datasets import load_iris
import numpy as np

from coresets import *


class TestSensitivity(object):

    @pytest.fixture
    def gen_data(self):
        X, _ = load_iris(return_X_y=True)
        centers = X[np.random.choice(np.arange(X.shape[0]), size=5)]
        return X, centers

    def test_kmeans_sensitivity(self, gen_data):
        # test kmeans sensitivity
        # make sure that the efficient C++ implementation gives the same results as Python
        X, centers = gen_data
        alpha = 1
        sensitivity = kmeans_sensitivity(X, np.ones(X.shape[0]), centers, alpha)

        # calc sensitivity in Python to check Cython impl
        dists = np.zeros((X.shape[0], centers.shape[0]))
        for i, x in enumerate(X):
            for j, c in enumerate(centers):
                dists[i, j] = np.sum((x - c) ** 2)
        assigns = np.argmin(dists, axis=1)
        dists = np.min(dists, axis=1)
        cnts = np.bincount(assigns)
        total = np.mean(dists)
        cluster_tot = np.zeros(centers.shape[0])
        for i, c in enumerate(centers):
            cluster_tot[i] = np.sum(dists[np.where(assigns == i)[0]])
        sensitivity2 = 2 * alpha * dists / total + 4 * alpha / (total * cnts[assigns]) * cluster_tot[assigns] + 4. * \
                       X.shape[0] / cnts[assigns]
        assert np.allclose(sensitivity, sensitivity2)
