import numpy as np
from sklearn.mixture import BayesianGaussianMixture


class WeightedBayesianGaussianMixture(BayesianGaussianMixture):
    """
    Extends sklearn.mixture.BayesianGaussianMixture to support weighted data set.
    Its methods and attributes are identical to the parent's, except for
    the fit() method.

    Parameters
    ----------
        See sklearn.mixture.BayesianGaussianMixture
    """

    def __init__(self, n_components=1, covariance_type='full', tol=1e-5,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weight_concentration_prior_type='dirichlet_process',
                 weight_concentration_prior=None,
                 mean_precision_prior=None, mean_prior=None,
                 degrees_of_freedom_prior=None, covariance_prior=None,
                 random_state=None, warm_start=False, verbose=0,
                 verbose_interval=10):
        super(WeightedBayesianGaussianMixture, self).__init__(
            n_components=n_components, covariance_type=covariance_type, tol=tol,
            reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params,
            weight_concentration_prior_type=weight_concentration_prior_type,
            weight_concentration_prior=weight_concentration_prior,
            mean_precision_prior=mean_precision_prior, mean_prior=mean_prior,
            degrees_of_freedom_prior=degrees_of_freedom_prior, covariance_prior=covariance_prior,
            random_state=random_state, warm_start=warm_start, verbose=verbose,
            verbose_interval=verbose_interval)
        self.weights = None

    def _compute_lower_bound(self, log_resp, log_prob_norm):
        return super(WeightedBayesianGaussianMixture, self)._compute_lower_bound(log_resp, log_prob_norm) \
               + np.sum(np.exp(log_resp) * (1 - np.exp(self.log_weight_mat)) * log_resp)

    def _initialize(self, X, resp):
        self.weight_mat = self.weights.repeat(self.n_components).reshape(X.shape[0], self.n_components)
        self.log_weight_mat = np.log(self.weight_mat)
        resp_w = resp * self.weight_mat
        super(WeightedBayesianGaussianMixture, self)._initialize(X, resp_w)

    def fit(self, X, weights=None, y=None):
        if weights is None:
            weights = np.ones(X.shape[0])
        if X.shape[0] != weights.shape[0]:
            raise ValueError("The number of weights must match the number of data points.")
        self.weights = weights
        super(WeightedBayesianGaussianMixture, self).fit(X, y)

    def _e_step(self, X):
        log_prob_norm, log_responsibility = super(WeightedBayesianGaussianMixture, self)._e_step(X)
        return log_prob_norm, log_responsibility + self.log_weight_mat
