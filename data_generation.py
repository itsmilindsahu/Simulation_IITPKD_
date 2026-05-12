"""
data_generation.py
==================
Generates synthetic distributions for the simulation study.

Each distribution nu_i is constructed per Section VI-A of the paper:
  1. Sample mu_i ~ U(0.3, 0.7),  sigma_i ~ U(0.001, 0.2)
  2. Base density: f_i = N(mu_i, sigma_i^2) on [0,1]
  3. Perturb:      g_i(x) = f_i(x) * exp(Z_i(x)),  Z_i ~ GP(0, Matern-5/2)
  4. Normalise:    nu_i has density g_i / integral(g_i)

Target function (Section VI-A):
  F(nu) = m1(nu) / (0.05 + sqrt(m2(nu) - m1(nu)^2))
"""

import numpy as np
from scipy.stats import norm
from scipy.integrate import trapezoid


def matern52_cov(x1, x2, length_scale=0.2):
    """
    Matern-5/2 covariance evaluated on two 1D grids.

    K(r) = (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)
    """
    r = np.abs(x1[:, None] - x2[None, :])
    s = np.sqrt(5) * r / length_scale
    C = (1 + s + s**2 / 3) * np.exp(-s)
    return C


def generate_one_distribution(rng, n_points=100, n_grid=200):
    mu_i    = rng.uniform(0.3, 0.7)
    sigma_i = rng.uniform(0.001, 0.2)

    x_grid = np.linspace(0, 1, n_grid)
    f_i    = norm.pdf(x_grid, loc=mu_i, scale=sigma_i)

    C   = matern52_cov(x_grid, x_grid, length_scale=0.2)
    C  += 1e-8 * np.eye(n_grid)
    L   = np.linalg.cholesky(C)
    z_i = L @ rng.standard_normal(n_grid)

    g_i        = f_i * np.exp(z_i)
    norm_const = trapezoid(g_i, x_grid)
    density    = g_i / norm_const

    cdf     = np.cumsum(density) * (x_grid[1] - x_grid[0])
    cdf     = np.clip(cdf, 0, 1)
    u_vals  = rng.uniform(0, 1, n_points)
    samples = np.interp(u_vals, cdf, x_grid)

    return samples, density, x_grid


def target_function(samples):
    m1 = np.mean(samples)
    m2 = np.mean(samples ** 2)
    variance = max(0.0, m2 - m1 ** 2)
    return m1 / (0.05 + np.sqrt(variance))


def generate_dataset(n, rng_seed=0, n_points=500, n_grid=200):
    rng = np.random.default_rng(rng_seed)
    distributions, y_values, densities = [], [], []
    for _ in range(n):
        samples, density, x_grid = generate_one_distribution(rng, n_points, n_grid)
        distributions.append(samples)
        y_values.append(target_function(samples))
        densities.append((density, x_grid))
    return distributions, np.array(y_values), densities