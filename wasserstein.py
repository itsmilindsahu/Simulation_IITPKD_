"""
wasserstein.py
==============
Computes the Wasserstein-2 distance between 1D probability distributions.

Paper reference: Section III-a  (Monge-Kantorovich distance)
Key formula:  W2(mu, nu) = sqrt( integral_0^1 (F^{-1}_mu(t) - F^{-1}_nu(t))^2 dt )
i.e. the L2 distance between the two quantile (inverse-CDF) functions.
For 1D distributions this has a clean closed form using quantile functions.
"""

import numpy as np


def quantile_function(samples, t_grid):
    """
    Estimate the quantile function F^{-1}(t) from a sample of observations.
    We sort the samples and interpolate at the requested quantile levels.

    Args:
        samples  : 1D numpy array of draws from the distribution
        t_grid   : 1D array of quantile levels in (0,1) where we evaluate F^{-1}

    Returns:
        q_vals   : array of quantile values, same length as t_grid
    """
    # Sort the samples – that gives us the empirical quantile function
    sorted_s = np.sort(samples)
    n = len(sorted_s)

    # Map each sorted sample to its quantile level  k/(n-1)
    levels = np.linspace(0, 1, n)

    # Linearly interpolate to get quantile values at the requested t_grid points
    q_vals = np.interp(t_grid, levels, sorted_s)
    return q_vals


def wasserstein2(samples_mu, samples_nu, n_grid=200):
    """
    Compute W2(mu, nu) using the quantile-function formula.

    Paper eq. (2):  W2(mu, nu) = T2(mu, nu)^{1/2}
    Paper eq. (24): W2^2(mu,nu) = E[ (F^{-1}_mu(U) - F^{-1}_nu(U))^2 ]
                                = integral_0^1 (q_mu(t) - q_nu(t))^2 dt

    We approximate the integral with a uniform grid of n_grid points.

    Args:
        samples_mu : draws from distribution mu
        samples_nu : draws from distribution nu
        n_grid     : number of quadrature points for the integral

    Returns:
        w2_dist    : scalar W2 distance
    """
    # Uniform grid on (0,1) – avoid 0 and 1 to stay inside the quantile domain
    t_grid = np.linspace(0.01, 0.99, n_grid)

    # Evaluate both quantile functions on the same grid (optimal coupling trick)
    q_mu = quantile_function(samples_mu, t_grid)
    q_nu = quantile_function(samples_nu, t_grid)

    # Numerical integral: (1/n_grid) * sum (q_mu - q_nu)^2  ≈  integral
    w2_squared = np.mean((q_mu - q_nu) ** 2)

    return np.sqrt(w2_squared)


def pairwise_w2_matrix(distributions, n_grid=200):
    """
    Build the full n x n matrix of pairwise W2 distances.

    Each entry D[i,j] = W2(mu_i, mu_j).
    The matrix is symmetric and has zeros on the diagonal.

    Args:
        distributions : list of 1D numpy arrays (samples from each distribution)
        n_grid        : quadrature points passed to wasserstein2()

    Returns:
        D             : (n, n) symmetric distance matrix
    """
    n = len(distributions)
    D = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):       # only upper triangle, then mirror
            d = wasserstein2(distributions[i], distributions[j], n_grid)
            D[i, j] = d
            D[j, i] = d                 # W2 is symmetric

    return D
