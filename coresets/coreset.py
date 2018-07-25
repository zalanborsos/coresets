from __future__ import print_function, absolute_import, division
import abc
import numpy as np
from sklearn.utils import check_array, check_random_state


class Coreset(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, X, w=None, random_state=None):
        X = check_array(X, accept_sparse="csr", order='C',
                        dtype=[np.float64, np.float32])
        self.X = X
        self.w = w if w is not None else np.ones(X.shape[0])
        self.n_samples = X.shape[0]
        self.random_state = check_random_state(random_state)
        self.calc_sampling_distribution()

    @abc.abstractmethod
    def calc_sampling_distribution(self):
        pass

    def generate_coreset(self, size):
        ind = np.random.choice(self.n_samples, size=size, p=self.p)
        return self.X[ind], 1. / (size * self.p[ind])
