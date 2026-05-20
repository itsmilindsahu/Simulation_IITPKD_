"""
gp_regression.py
================
Gaussian Process regression (Kriging) for distribution inputs.

Implements the model from Bachoc et al. (2018), Section V-A.
Key equations:
  - MLE objective:  L(theta) = (1/n) * [log det R_theta + y^T R_theta^{-1} y]   eq. (19)
  - Posterior mean: Yhat(mu*) = r_theta(mu*)^T R_theta^{-1} y                   eq. (20)
  - Posterior var:  Var(mu*)  = K(mu*,mu*) - r^T R^{-1} r                       eq. (20)

Parameters theta = (sigma^2, ell, H) are optimised in log/logit space so that
scipy.minimize can treat the problem as unconstrained.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

from kernels import kernel_power_exp, kernel_power_exp_single, add_nugget
from wasserstein import wasserstein2


def neg_log_likelihood(params, D_train, y_train, delta=1e-6):
    """
    Normalised negative log-likelihood for the power-exponential kernel.

    Args:
        params   : (3,) array [log_sigma2, log_ell, logit_H]  (unconstrained)
        D_train  : (n,n) pairwise W2 distance matrix for training points
        y_train  : (n,)  observed output values
        delta    : small nugget added for numerical stability

    Returns:
        nll      : scalar negative log-likelihood value
    """
    sigma2 = np.exp(params[0])
    ell    = np.exp(params[1])
    H      = 1 / (1 + np.exp(-params[2]))   # sigmoid maps R -> (0,1)

    n = len(y_train)

    K = kernel_power_exp(D_train, sigma2, ell, H)
    K = add_nugget(K, delta)

    try:
        c, low = cho_factor(K)
    except np.linalg.LinAlgError:
        # Matrix isn't PD at this parameter point; skip with a large penalty
        return 1e10

    # log det K = 2 * sum(log diag(L))  via Cholesky
    log_det = 2 * np.sum(np.log(np.diag(c)))

    alpha = cho_solve((c, low), y_train)
    quad  = y_train @ alpha

    nll = (log_det + quad) / n

    return nll


def fit_gp(D_train, y_train, delta=1e-6, n_restarts=5):
    """
    Estimate kernel parameters theta = (sigma^2, ell, H) by Maximum Likelihood.

    Multiple random restarts reduce the chance of landing in a bad local minimum,
    which matters because the NLL surface for GP kernels can be multimodal.

    Args:
        D_train    : (n,n) pairwise W2 distance matrix
        y_train    : (n,)  observed outputs
        delta      : nugget for numerical stability
        n_restarts : number of random restarts for the optimiser

    Returns:
        best_params : dict with keys 'sigma2', 'ell', 'H'
    """
    best_nll = np.inf
    best_raw  = None

    rng = np.random.default_rng(42)

    for _ in range(n_restarts):
        x0 = rng.uniform([-2, -2, -2], [2, 2, 2])

        res = minimize(
            neg_log_likelihood,
            x0,
            args=(D_train, y_train, delta),
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-10}
        )

        if res.fun < best_nll:
            best_nll = res.fun
            best_raw  = res.x

    sigma2 = float(np.exp(best_raw[0]))
    ell    = float(np.exp(best_raw[1]))
    H      = float(1 / (1 + np.exp(-best_raw[2])))

    return {'sigma2': sigma2, 'ell': ell, 'H': H}


def predict(dist_test, dists_train, y_train, params, delta=1e-6, D_train=None):
    """
    Predict Y at a new distribution using Kriging (posterior mean + variance).

    Args:
        dist_test   : 1D array – samples from the new test distribution
        dists_train : list of 1D arrays – training distributions
        y_train     : (n,)  training outputs
        params      : dict with 'sigma2', 'ell', 'H'  (from fit_gp)
        delta       : nugget (must match what was used during training)
        D_train     : (n,n) pre-computed training distance matrix (optional, speeds things up)

    Returns:
        y_pred  : scalar predicted mean
        y_var   : scalar predicted variance (>= 0)
    """
    sigma2 = params['sigma2']
    ell    = params['ell']
    H      = params['H']
    n      = len(dists_train)

    if D_train is None:
        from wasserstein import pairwise_w2_matrix
        D_train = pairwise_w2_matrix(dists_train)

    K_train = kernel_power_exp(D_train, sigma2, ell, H)
    K_train = add_nugget(K_train, delta)

    # Cross-covariance vector: K(mu*, mu_i) for each training point
    r = np.array([
        kernel_power_exp_single(wasserstein2(dist_test, dists_train[i]), sigma2, ell, H)
        for i in range(n)
    ])

    try:
        c, low = cho_factor(K_train)
    except np.linalg.LinAlgError:
        # Rare but possible when delta is too small relative to condition number;
        # fall back to a safe zero prediction rather than crashing.
        return 0.0, float(kernel_power_exp_single(0.0, sigma2, ell, H))

    alpha  = cho_solve((c, low), y_train)
    y_pred = float(r @ alpha)

    v      = cho_solve((c, low), r)
    k_star = kernel_power_exp_single(0.0, sigma2, ell, H)  # K(mu*,mu*) = sigma^2
    y_var  = float(max(0.0, k_star - r @ v))               # clamp numerical noise

    return y_pred, y_var
