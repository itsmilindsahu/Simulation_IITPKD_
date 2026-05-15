"""
kernels.py
==========
Positive-definite covariance kernels built from the Wasserstein distance.

Paper reference: Section IV – Gaussian Process Models for Distribution Inputs

Two kernel families implemented:
  1. Power-exponential (stationary) – eq. (14)   K(mu,nu) = sigma^2 * exp(-W2^{2H} / ell)
  2. Fractional Brownian Motion     – eq. (8)    K(mu,nu) = 0.5*(W2^{2H}(o,mu) + W2^{2H}(o,nu) - W2^{2H}(mu,nu))

Both are proven positive-definite via Theorem IV.2 / IV.1 in the paper.
"""

import numpy as np


# ── 1. Power-Exponential stationary kernel ──────────────────────────────────

def kernel_power_exp(D, sigma2, ell, H):
    """
    Stationary power-exponential kernel evaluated on a pre-computed distance matrix.

    Paper eq. (13)/(14):  K(mu,nu) = sigma^2 * exp( - W2^{2H}(mu,nu) / ell )

    This is a completely-monotone function F(x) = sigma^2 * exp(-x/ell)
    composed with the negative-definite kernel W2^{2H}, so by Theorem IV.2
    it is positive-definite.

    Special cases:
      H = 0.5  -->  Laplace  / Ornstein-Uhlenbeck style
      H = 1.0  -->  Gaussian / squared-exponential style

    Args:
        D      : (n, n) pairwise W2 distance matrix
        sigma2 : variance  (output scale^2),  > 0
        ell    : length-scale,                > 0
        H      : smoothness exponent in (0, 1]

    Returns:
        K      : (n, n) covariance matrix
    """
    # Raise W2 distances to the power 2H  (kernel argument is W2^{2H})
    D2H = D ** (2 * H)

    # Apply the completely-monotone function  F(x) = sigma^2 * exp(-x / ell)
    K = sigma2 * np.exp(-D2H / ell)

    return K


def kernel_power_exp_single(d, sigma2, ell, H):
    """
    Same formula but for a single scalar distance d.
    Used when predicting at a new test distribution.
    """
    return sigma2 * np.exp(-(d ** (2 * H)) / ell)


# ── 2. Fractional Brownian Motion kernel ────────────────────────────────────

def kernel_fbm(D, D_origin, sigma2, H):
    """
    Fractional Brownian Motion kernel with distribution inputs.

    Paper eq. (8):
      K_{H,mu0}(mu, nu) = 0.5 * ( W2^{2H}(mu0, mu)
                                 + W2^{2H}(mu0, nu)
                                 - W2^{2H}(mu,  nu) )

    This has STATIONARY INCREMENTS (not stationary) w.r.t. W2.
    The origin distribution mu0 plays the role of 0 in classical fBm.

    Proven positive-definite by Theorem IV.1 via Schoenberg's theorem
    + the negative-definiteness of W2^{2H} (Theorem IV.3).

    Args:
        D        : (n, n) pairwise W2 distances between training distributions
        D_origin : (n,)  W2 distances from each training dist to the origin mu0
        sigma2   : overall variance scale
        H        : Hurst exponent in (0, 1)   (NOT 0 or 1 – those are degenerate)

    Returns:
        K        : (n, n) covariance matrix
    """
    n = len(D_origin)

    # Broadcast: d_origin[i] contributes to row i and d_origin[j] to col j
    d_i = D_origin[:, None] ** (2 * H)   # shape (n,1)
    d_j = D_origin[None, :] ** (2 * H)   # shape (1,n)
    d_ij = D ** (2 * H)                  # shape (n,n)

    K = sigma2 * 0.5 * (d_i + d_j - d_ij)

    return K


# ── 3. Nugget / noise addition ───────────────────────────────────────────────

def add_nugget(K, delta):
    """
    Add a nugget (diagonal noise) to the covariance matrix.

    Paper eq. (23): K_{...,delta}(nu1,nu2) = K(nu1,nu2) + delta * 1{W2(nu1,nu2)=0}

    This corresponds to the delta term on the diagonal of K.
    It handles:
      - Numerical stabilisation (avoids near-singular matrices)
      - Gaussian observation noise / two-stage sampling uncertainty (Section VI-B)

    Args:
        K     : (n, n) covariance matrix
        delta : nugget variance  >= 0

    Returns:
        K_noisy : (n, n) regularised covariance matrix
    """
    return K + delta * np.eye(len(K))
