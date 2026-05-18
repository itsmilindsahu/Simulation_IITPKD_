"""
run_simulation.py
=================
Main script that replicates Table I from the paper.

Paper reference: Section VI-A

Experiment:
  - n=100   training distributions
  - nt=500  test distributions  (paper uses 500)
  - Compare three GP models: "distribution" (W2-based), "Legendre", "PCA"
  - Quality metrics: RMSE and CIR_0.9

Expected results (from paper Table I):
  model            RMSE    CIR_0.9
  distribution     0.094   0.92
  Legendre ord 5   0.49    0.92
  Legendre ord 10  0.34    0.89
  PCA ord 5        0.63    0.82
  PCA ord 10       0.52    0.87

NOTE: We use n=100 train / nt=100 test (not 500) to keep runtime reasonable.
      Results will approximate but not exactly match the paper.
      Increase N_TEST to 500 for full replication (takes ~30 min).

Usage:
  python run_simulation.py
"""

import numpy as np
import time
from scipy.stats import norm

# Our modules
from data_generation import generate_dataset
from wasserstein import pairwise_w2_matrix, wasserstein2
from kernels import kernel_power_exp, add_nugget
from gp_regression import fit_gp, predict
from baselines import (
    legendre_features, compute_pca_components, pca_features,
    fit_baseline_gp, predict_baseline
)

# ── Config ────────────────────────────────────────────────────────────────────
N_TRAIN   = 100    # training distributions (paper uses 100)
N_TEST    = 100    # test distributions    (paper uses 500; raise for full replication)
N_SAMPLES = 500    # samples per distribution
ORDERS    = [5]    # Legendre / PCA orders to test (paper tests 5, 10, 15)
DELTA     = 1e-4   # nugget variance
ALPHA_CIR = 0.9    # confidence level for CIR metric


def cir(y_true, y_pred, y_std, alpha=0.9):
    """
    Confidence Interval Ratio – fraction of test points where |error| <= q_alpha * std.

    Paper eq. just below Table I:
      CIR_alpha = (1/nt) * sum_i  1{ |F(nu_{t,i}) - Fhat(nu_{t,i})| <= q_alpha * sigmahat }
    where q_alpha is the (1/2 + alpha/2) quantile of N(0,1).

    Should be close to alpha for a well-calibrated GP.
    """
    q_alpha = norm.ppf(0.5 + alpha / 2)         # e.g. ~1.645 for alpha=0.9
    inside  = np.abs(y_true - y_pred) <= q_alpha * y_std
    return float(np.mean(inside))


def rmse(y_true, y_pred):
    """Root Mean Squared Error – should be as small as possible."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ── 1. Generate data ──────────────────────────────────────────────────────────
print("=" * 60)
print("Replication of Table I  –  Bachoc et al. 2018")
print("=" * 60)

print(f"\n[1/5] Generating {N_TRAIN} training distributions...")
t0 = time.time()
train_dists, y_train, train_dens = generate_dataset(N_TRAIN, rng_seed=0, n_points=N_SAMPLES)
print(f"      Done in {time.time()-t0:.1f}s")

print(f"[2/5] Generating {N_TEST} test distributions...")
t0 = time.time()
test_dists, y_test, test_dens = generate_dataset(N_TEST, rng_seed=99, n_points=N_SAMPLES)
print(f"      Done in {time.time()-t0:.1f}s")

# ── 2. Precompute W2 distance matrices ────────────────────────────────────────
print("[3/5] Computing pairwise W2 distances (training)...")
t0 = time.time()
D_train = pairwise_w2_matrix(train_dists)
print(f"      Done in {time.time()-t0:.1f}s")

# ── 3. "distribution" GP model ────────────────────────────────────────────────
print("\n[4/5] Fitting 'distribution' GP (W2-based)...")
t0 = time.time()
params_dist = fit_gp(D_train, y_train, delta=DELTA, n_restarts=3)
print(f"      MLE params: sigma2={params_dist['sigma2']:.4f}  "
      f"ell={params_dist['ell']:.4f}  H={params_dist['H']:.4f}")
print(f"      Fit time: {time.time()-t0:.1f}s")
# print(f"      DEBUG D_train range: {D_train.min():.4f} – {D_train.max():.4f}")

print("      Predicting on test set...")
t0 = time.time()
preds_dist = []
vars_dist  = []
for i, td in enumerate(test_dists):
    yp, yv = predict(td, train_dists, y_train, params_dist, delta=DELTA, D_train=D_train)
    preds_dist.append(yp)
    vars_dist.append(yv)
    if (i + 1) % 20 == 0:
        print(f"      {i+1}/{N_TEST} done...")

preds_dist = np.array(preds_dist)
# TODO: check whether clipping negative variances here is hiding a kernel issue
stds_dist  = np.sqrt(np.maximum(0, np.array(vars_dist)))
print(f"      Prediction time: {time.time()-t0:.1f}s")

rmse_dist = rmse(y_test, preds_dist)
cir_dist  = cir(y_test, preds_dist, stds_dist, ALPHA_CIR)
print(f"\n  >> 'distribution'  RMSE={rmse_dist:.3f}   CIR_0.9={cir_dist:.2f}")

# ── 4. "Legendre" GP models ───────────────────────────────────────────────────
print("\n[5/5] Fitting 'Legendre' and 'PCA' baselines...")
results = {'distribution': {'RMSE': rmse_dist, 'CIR_0.9': cir_dist}}

for order in ORDERS:
    print(f"\n  -- Legendre order={order} --")

    # Build Legendre feature matrix for training set
    X_leg_train = np.array([legendre_features(d, order) for d in train_dists])
    X_leg_test  = np.array([legendre_features(d, order) for d in test_dists])

    params_leg = fit_baseline_gp(X_leg_train, y_train, delta=DELTA)
    print(f"     MLE: sigma2={params_leg['sigma2']:.4f}  ell={params_leg['ell']:.4f}  H={params_leg['H']:.4f}")

    preds_leg, vars_leg = [], []
    for i in range(N_TEST):
        yp, yv = predict_baseline(X_leg_test[i], X_leg_train, y_train, params_leg, delta=DELTA)
        preds_leg.append(yp)
        vars_leg.append(yv)

    preds_leg = np.array(preds_leg)
    stds_leg  = np.sqrt(np.maximum(0, np.array(vars_leg)))

    r = rmse(y_test, preds_leg)
    c = cir(y_test, preds_leg, stds_leg, ALPHA_CIR)
    results[f'Legendre ord {order}'] = {'RMSE': r, 'CIR_0.9': c}
    print(f"  >> Legendre ord {order}  RMSE={r:.3f}   CIR_0.9={c:.2f}")

# ── PCA baselines ─────────────────────────────────────────────────────────────
for order in ORDERS:
    print(f"\n  -- PCA order={order} --")

    comps, xg, mean_row = compute_pca_components(train_dens, n_components=order)

    X_pca_train = np.array([pca_features(d, comps, xg, mean_row) for d in train_dists])
    X_pca_test  = np.array([pca_features(d, comps, xg, mean_row) for d in test_dists])

    params_pca = fit_baseline_gp(X_pca_train, y_train, delta=DELTA)
    print(f"     MLE: sigma2={params_pca['sigma2']:.4f}  ell={params_pca['ell']:.4f}  H={params_pca['H']:.4f}")

    preds_pca, vars_pca = [], []
    for i in range(N_TEST):
        yp, yv = predict_baseline(X_pca_test[i], X_pca_train, y_train, params_pca, delta=DELTA)
        preds_pca.append(yp)
        vars_pca.append(yv)

    preds_pca = np.array(preds_pca)
    stds_pca  = np.sqrt(np.maximum(0, np.array(vars_pca)))

    r = rmse(y_test, preds_pca)
    c = cir(y_test, preds_pca, stds_pca, ALPHA_CIR)
    results[f'PCA ord {order}'] = {'RMSE': r, 'CIR_0.9': c}
    print(f"  >> PCA ord {order}  RMSE={r:.3f}   CIR_0.9={c:.2f}")

# ── Final Table ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTS  (replicated Table I)")
print("=" * 60)
print(f"{'Model':<22}  {'RMSE':>6}  {'CIR_0.9':>8}")
print("-" * 42)
for model, vals in results.items():
    print(f"{model:<22}  {vals['RMSE']:>6.3f}  {vals['CIR_0.9']:>8.2f}")

print("\nPaper Table I values for reference:")
print(f"{'distribution':<22}  {'0.094':>6}  {'0.92':>8}")
print(f"{'Legendre ord 5':<22}  {'0.49':>6}  {'0.92':>8}")
print(f"{'PCA ord 5':<22}  {'0.63':>6}  {'0.82':>8}")
print("=" * 60)
print("\nDone! The 'distribution' GP (W2-based) should clearly outperform baselines.")
