from __future__ import print_function, absolute_import, division
import abc
import numpy as np
from sklearn.utils import check_array, check_random_state


class Coreset(object):
    """
    Abstract class for coresets.

    Parameters
    ----------
    X : ndarray, shape (n_points, n_dims)
        The data set to generate coreset from.
    w : ndarray, shape (n_points), optional
        The weights of the data points. This allows generating coresets from a
        weighted data set, for example generating coreset of a coreset. If None,
        the data is treated as unweighted and w will be replaced by all ones array.
    random_state : int, RandomState instance or None, optional (default=None)
    """
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
        """
        Calculates the coreset importance sampling distribution.
        """
        pass

    def generate_coreset(self, size):
        """
        Generates a coreset of the data set.

        Parameters
        ----------
        size : int
            The size of the coreset to generate.

        """
        ind = np.random.choice(self.n_samples, size=size, p=self.p)
        return self.X[ind], 1. / (size * self.p[ind])
