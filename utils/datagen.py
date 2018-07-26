import numpy as np


def crp(n, alpha, d=2, sigma_sq=20, sigma_0_sq=5000):
    """
    Generates samples from a DPMM with Gaussian likelihood and fixed cluster covariance using
    the Chinese restaurant process.

    The mean prior is also Gaussian with mean 0 and variance sigma_0_sq.

    Parameters
    ----------
    n : int
        number of samples to generate
    alpha : float
        weight concentration parameter
    d : int
        dimension of the model
    sigma_sq : float
        the fixed cluster variance for the likelihood
    sigma_0_sq : float
        the variance of the mean prior

    Returns
    -------
    points : ndarray
        generated points
    nr_clusters : int
        number of generated clusters
    assignment : ndarray
        assignment of each point to cluster

    """
    mu_mean = np.zeros(d)
    cov_mean = np.eye(d) * sigma_0_sq
    cov_points = np.eye(d) * sigma_sq
    mu = np.random.multivariate_normal(mu_mean, cov_mean)
    points = [np.random.multivariate_normal(mu, cov_points)]
    assignment = np.zeros(n + 1).astype(int)
    assignment[1] = 1
    centers = [mu]
    members = [alpha, 1]
    total = alpha + 1
    nr_clusters = 1

    for i in range(2, n + 1):
        threshold = np.random.rand() * total
        ind = 0
        ccnt = members[ind]
        while ccnt < threshold:
            ind += 1
            ccnt += members[ind]
        if ind > 0:
            members[ind] += 1
            assignment[i] = ind
            mean = centers[ind - 1]
            total += 1
        else:
            mean = np.random.multivariate_normal(mu_mean, cov_mean)
            nr_clusters += 1
            total += 1
            centers.append(mean)
            members.append(1)
            assignment[i] = nr_clusters

        points.append(np.random.multivariate_normal(mean, cov_points))

    points = np.array(points)
    return points, nr_clusters, assignment[1:] - 1
