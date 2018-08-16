from __future__ import division, absolute_import

import pytest
import numpy as np
from algorithms import weighted_kmeans


class TestWeightedKmeans(object):

    def test_kmeans_seeding(self):
        # test that seeding with 2 clusters for 2 points achieves 0 cost
        X = np.array([[0, 0], [10, 10]])
        w = np.array([1, 1])
        km = weighted_kmeans.WeightedKMeans(n_clusters=2)
        km.fit(X, w)
        y, dists = km.predict(X)
        assert len(y) == 2
        assert y[0] != y[1]
        assert np.allclose(dists, 0)

    def test_kmeans_weighted(self):
        # test k-Means with weighted data and 1 center
        X = np.array([[0, 0], [10, 10]])
        w = np.array([100, 1])
        km = weighted_kmeans.WeightedKMeans(n_clusters=1)
        km.fit(X, w)
        y, dists = km.predict(X)

        expected_center = X[1] / 101
        assert len(y) == 2
        assert y[0] == y[1]
        assert np.allclose(km.centers, expected_center)
        assert np.allclose(dists[0], expected_center.dot(expected_center))
        assert np.allclose(dists[1], (X[1] - expected_center).dot(X[1] - expected_center))
