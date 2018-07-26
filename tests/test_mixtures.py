from __future__ import division, absolute_import

from sklearn import datasets

import pytest
from algorithms import WeightedGaussianMixture, WeightedBayesianGaussianMixture


class TestMixtures(object):

    def test_mixtures(self):
        iris = datasets.load_iris()

        wgm = WeightedGaussianMixture()
        wgm.fit(iris.data)
        wgm.predict(iris.data)

        wbgm = WeightedBayesianGaussianMixture()
        wbgm.fit(iris.data)
        wbgm.predict(iris.data)
