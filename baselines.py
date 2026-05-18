"""
baselines.py
============
Projection-based GP baselines for comparison.

Paper reference: Section VI-A – "Legendre" and "PCA" models

Instead of using W2 distance directly, these models:
  1. Project each distribution onto a finite-dimensional feature vector
  2. Fit a standard GP on those feature vectors

Two projection methods:
  - Legendre : project density onto normalised Legendre polynomials (eq. in paper)
  - PCA      : project discretised density onto top-k principal components

Both then fit the same power-exponential GP on the projected features using
the standard Euclidean distance (not Wasserstein).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve


# ── Legendre projection ───────────────────────────────────────────────────────

def legendre_poly(i, x):
    """
    Evaluate the i-th normalised Legendre polynomial at points x in [0,1].

    The paper uses  a_i(nu) = integral_0^1 f_nu(t) * p_i(t) dt
    where p_i is normalised so that integral_0^1 p_i^2 = 1.

    We use the standard Legendre polynomials (defined on [-1,1]) shifted to [0,1].
    """
    # Map [0,1] to [-1,1]
    t = 2 * x - 1

    if i == 0:
        return np.ones_like(t)          # P_0(t) = 1,   norm = sqrt(2) -> normalise
    elif i == 1:
        return np.sqrt(3) * t           # P_1(t) = t,   norm factor included
    elif i == 2:
        return np.sqrt(5) * (3*t**2 - 1) / 2
    elif i == 3:
        return np.sqrt(7) * (5*t**3 - 3*t) / 2
    elif i == 4:
        return np.sqrt(9) * (35*t**4 - 30*t**2 + 3) / 8
    else:
        # Recurse via three-term recurrence (stable for moderate i)
        p_prev2 = legendre_poly(i-2, x)
        p_prev1 = legendre_poly(i-1, x)
        n = i
        return ((2*n-1)*t*p_prev1 - (n-1)*p_prev2) / n


def legendre_features(samples, order, x_grid=None, n_grid=100):
    """
    Compute Legendre projection features a_0..a_{order-1} for a distribution.

    Paper eq.: a_i(nu) = integral_0^1 f_nu(t) * p_i(t) dt
    Approximated here via empirical expectation: a_i ≈ (1/N) sum_k p_i(x_k).

    Args:
        samples  : 1D array of draws from the distribution
        order    : number of Legendre coefficients
        x_grid   : not used here (samples already available)
        n_grid   : unused placeholder

    Returns:
        feats    : (order,) feature vector
    """
    # Compute a_i = E_{nu}[p_i(X)]  by Monte-Carlo (samples are draws from nu)
    feats = np.array([np.mean(legendre_poly(i, samples)) for i in range(order)])
    return feats


# ── PCA projection ────────────────────────────────────────────────────────────

def compute_pca_components(densities, n_components, n_grid=100):
    """
    Compute PCA components from a list of discretised densities.

    Paper Section VI-A: discretise f_nu_i at d=100 equally spaced points,
    then take top-o principal components of the resulting matrix.

    Args:
        densities     : list of (density_array, x_grid) tuples
        n_components  : number of principal components to keep
        n_grid        : grid size for resampling

    Returns:
        components    : (n_components, n_grid) array of PC vectors
        x_grid_out    : (n_grid,) common x-grid used
    """
    x_out = np.linspace(0, 1, n_grid)

    # Resample each density onto the common grid
    matrix = []
    for (dens, xg) in densities:
        resampled = np.interp(x_out, xg, dens)
        matrix.append(resampled)

    V = np.array(matrix)            # shape (n_distributions, n_grid)

    # Centre the matrix (subtract mean density)
    mean_row = V.mean(axis=0)
    V_centred = V - mean_row

    # SVD to get principal components
    _, _, Vt = np.linalg.svd(V_centred, full_matrices=False)
    components = Vt[:n_components]  # shape (n_components, n_grid)

    return components, x_out, mean_row


def pca_features(samples, components, x_grid_pca, mean_row, n_grid=100):
    """
    Project a distribution onto pre-fitted PCA components.

    Paper: a_i(nu) = (1/d) * sum_{j=0}^{d-1} f_nu(j/(d-1)) * (w_i)_j

    Args:
        samples     : draws from the distribution (used to estimate density)
        components  : (n_components, n_grid) from compute_pca_components
        x_grid_pca  : x-grid used when computing PCA
        mean_row    : mean density subtracted before PCA
        n_grid      : grid resolution

    Returns:
        feats       : (n_components,) feature vector
    """
    # Estimate density via histogram on the PCA x-grid
    x_min, x_max = x_grid_pca[0], x_grid_pca[-1]
    counts, _ = np.histogram(samples, bins=n_grid, range=(x_min, x_max), density=True)

    # Centre
    counts_c = counts - mean_row

    # Project: a_i = (1/n_grid) * counts . w_i
    feats = components @ counts_c / n_grid

    return feats


# ── Generic GP on Euclidean features ─────────────────────────────────────────

def euclid_kernel(X, sigma2, ell_vec, H):
    """
    Power-exponential kernel on a feature matrix using per-dimension length-scales.

    K(x, x') = sigma^2 * exp( - sum_d (|x_d - x'_d| / ell_d)^{2H} )

    This is the standard kernel used for Legendre and PCA models.

    Args:
        X       : (n, p) feature matrix
        sigma2  : variance
        ell_vec : (p,) length-scales, one per feature dimension
        H       : smoothness exponent

    Returns:
        K       : (n, n) covariance matrix
    """
    n, p = X.shape
    K = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            diff  = np.abs(X[i] - X[j]) / ell_vec
            k_val = sigma2 * np.exp(-np.sum(diff ** (2 * H)))
            K[i, j] = k_val
            K[j, i] = k_val

    return K


def fit_baseline_gp(X_train, y_train, delta=1e-6):
    """
    Fit a GP on feature vectors by MLE (same objective as the distribution GP).

    Parameters: sigma2, one global length-scale ell, and H.
    (Simplified from paper which uses one ell per dimension, to keep code small.)

    Args:
        X_train  : (n, p) training feature matrix
        y_train  : (n,)   training outputs

    Returns:
        params   : dict with 'sigma2', 'ell', 'H'
    """
    def nll(raw):
        s2  = np.exp(raw[0])
        ell = np.exp(raw[1])
        H   = 1 / (1 + np.exp(-raw[2]))
        n   = len(y_train)

        ell_vec = np.full(X_train.shape[1], ell)
        K = euclid_kernel(X_train, s2, ell_vec, H) + (delta + 1e-8) * np.eye(n)

        try:
            c, low = cho_factor(K)
        except Exception:
            return 1e10

        ld    = 2 * np.sum(np.log(np.diag(c)))
        alpha = cho_solve((c, low), y_train)
        return (ld + y_train @ alpha) / n

    rng  = np.random.default_rng(7)
    best = (np.inf, None)
    for _ in range(5):
        x0  = rng.uniform([-2, -2, -2], [2, 2, 2])
        res = minimize(nll, x0, method='L-BFGS-B', options={'maxiter': 200})
        if res.fun < best[0]:
            best = (res.fun, res.x)

    r      = best[1]
    return {'sigma2': float(np.exp(r[0])),
            'ell':    float(np.exp(r[1])),
            'H':      float(1 / (1 + np.exp(-r[2])))}


def predict_baseline(x_test, X_train, y_train, params, delta=1e-6):
    """
    Kriging prediction for the Euclidean feature-based GP.

    Same formula as eq. (20) in the paper, but with Euclidean kernel.

    Returns: (mean, variance)
    """
    s2  = params['sigma2']
    ell = params['ell']
    H   = params['H']
    n   = len(y_train)

    ell_vec = np.full(X_train.shape[1], ell)

    K_train = euclid_kernel(X_train, s2, ell_vec, H) + (delta + 1e-8) * np.eye(n)

    # Cross-covariance r(x*)
    r = np.array([
        s2 * np.exp(-np.sum((np.abs(x_test - X_train[i]) / ell_vec)**(2*H)))
        for i in range(n)
    ])

    c, low = cho_factor(K_train)
    alpha  = cho_solve((c, low), y_train)
    y_pred = float(r @ alpha)

    v     = cho_solve((c, low), r)
    y_var = float(max(0.0, s2 - r @ v))

    return y_pred, y_var
