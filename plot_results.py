"""
plot_results.py
===============
Visualise the simulation results and distributions.

Produces:
  1. Figure 1 replica  – density plots of 10 sample training distributions
  2. Prediction scatter – true vs predicted for the 'distribution' model
  3. RMSE bar chart     – comparison across models

Run this after run_simulation.py, or call it standalone (it regenerates data).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')       # no display needed; saves to file
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from data_generation import generate_dataset
from wasserstein import pairwise_w2_matrix, wasserstein2
from gp_regression import fit_gp, predict


# ── Reproduce Figure 1: density plots ────────────────────────────────────────

def plot_densities(densities, n_show=10, save_path='fig1_densities.png'):
    """
    Plot density functions of the first n_show training distributions.
    Replicates Figure 1 of the paper.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    colors = plt.cm.tab10(np.linspace(0, 1, n_show))
    for i in range(n_show):
        dens, xg = densities[i]
        ax.plot(xg, dens, color=colors[i], alpha=0.8, linewidth=1.2)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Figure 1 – Sample training distributions (10 of 100)', fontsize=11)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Scatter plot: true vs predicted ──────────────────────────────────────────

def plot_scatter(y_true, y_pred, y_std, model_name, save_path='scatter.png'):
    """
    True vs predicted scatter with ±1.645*std error bars (90% CI).
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.errorbar(y_true, y_pred, yerr=1.645 * y_std,
                fmt='o', alpha=0.4, markersize=3, elinewidth=0.5, color='steelblue')

    # Perfect-prediction diagonal
    lo = min(y_true.min(), y_pred.min()) - 0.2
    hi = max(y_true.max(), y_pred.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect prediction')

    ax.set_xlabel('True F(ν)', fontsize=12)
    ax.set_ylabel('Predicted F̂(ν)', fontsize=12)
    ax.set_title(f'{model_name} – True vs Predicted', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Bar chart: RMSE comparison ────────────────────────────────────────────────

def plot_rmse_bar(results_dict, save_path='rmse_comparison.png'):
    """
    Bar chart comparing RMSE across models.
    Highlights the 'distribution' bar in a different colour.
    """
    models = list(results_dict.keys())
    rmses  = [results_dict[m]['RMSE'] for m in models]
    colors = ['#e74c3c' if 'distribution' in m else '#3498db' for m in models]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(len(models)), rmses, color=colors, edgecolor='k', linewidth=0.5)
    ax.bar_label(bars, fmt='%.3f', padding=2, fontsize=9)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('RMSE Comparison – Table I Replica', fontsize=11)
    ax.set_ylim(0, max(rmses) * 1.25)
    ax.grid(True, axis='y', alpha=0.3)

    # Legend patch
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#e74c3c', label='Distribution (W2)'),
                        Patch(color='#3498db', label='Projection baselines')],
              fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Error distribution histogram ─────────────────────────────────────────────

def plot_error_histogram(y_true, y_pred, model_name, save_path='error_histogram.png'):
    """
    Histogram of absolute prediction errors.
    """
    errors = np.abs(y_true - y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(errors, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(errors):.4f}')
    ax.axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median = {np.median(errors):.4f}')
    
    ax.set_xlabel('Absolute Error |y_true - y_pred|', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{model_name} – Prediction Error Distribution', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── CIR comparison bar chart ──────────────────────────────────────────────────

def plot_cir_bar(results_dict, save_path='cir_comparison.png'):
    """
    Bar chart comparing CIR (Coverage Index Rate at 90%) across models.
    """
    models = list(results_dict.keys())
    cirs   = [results_dict[m]['CIR'] for m in models]
    colors = ['#e74c3c' if 'distribution' in m else '#3498db' for m in models]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(len(models)), cirs, color=colors, edgecolor='k', linewidth=0.5)
    ax.bar_label(bars, fmt='%.3f', padding=2, fontsize=9)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('CIR (90%)', fontsize=12)
    ax.set_title('Coverage Index Rate Comparison', fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.9, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='Target = 0.90')
    ax.grid(True, axis='y', alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#e74c3c', label='Distribution (W2)'),
                        Patch(color='#3498db', label='Projection baselines')],
              fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Residuals plot ────────────────────────────────────────────────────────────

def plot_residuals(y_true, y_pred, model_name, save_path='residuals.png'):
    """
    Residuals vs predicted values with horizontal line at zero.
    """
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.5, s=40, color='steelblue', edgecolor='k', linewidth=0.3)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    
    ax.set_xlabel('Predicted F̂(ν)', fontsize=12)
    ax.set_ylabel('Residuals (True - Predicted)', fontsize=12)
    ax.set_title(f'{model_name} – Residual Plot', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── QQ plot for normality check ───────────────────────────────────────────────

def plot_qq(y_true, y_pred, y_std, model_name, save_path='qq_plot.png'):
    """
    Q-Q plot: normalized residuals vs standard normal quantiles.
    """
    from scipy import stats
    
    normalized_residuals = (y_true - y_pred) / (y_std + 1e-8)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    stats.probplot(normalized_residuals, dist="norm", plot=ax)
    
    ax.set_title(f'{model_name} – Q-Q Plot (Normality Check)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    from baselines import (legendre_features, compute_pca_components,
                            pca_features, fit_baseline_gp, predict_baseline)
    from scipy.stats import norm

    N = 60    # smaller for fast plotting demo
    print("Generating data...")
    train_d, y_tr, train_dens = generate_dataset(N,  rng_seed=0)
    test_d,  y_te, test_dens  = generate_dataset(40, rng_seed=5)

    # Figure 1
    print("Plotting densities (Figure 1 replica)...")
    plot_densities(train_dens, n_show=10, save_path='fig1_densities.png')

    # Fit distribution GP
    print("Fitting distribution GP...")
    D_tr = pairwise_w2_matrix(train_d)
    p    = fit_gp(D_tr, y_tr, delta=1e-4, n_restarts=2)
    print(f"  params: {p}")

    preds, stds = [], []
    for td in test_d:
        yp, yv = predict(td, train_d, y_tr, p, D_train=D_tr)
        preds.append(yp); stds.append(np.sqrt(max(0, yv)))

    preds = np.array(preds); stds = np.array(stds)

    # Plot 2: Scatter plot for distribution GP
    print("Plotting scatter (true vs predicted)...")
    plot_scatter(y_te, preds, stds, 'Distribution GP (W2)', save_path='scatter_dist.png')

    # Plot 4: Error histogram for distribution GP
    print("Plotting error histogram...")
    plot_error_histogram(y_te, preds, 'Distribution GP (W2)', save_path='error_histogram.png')

    # Plot 5: Residuals plot for distribution GP
    print("Plotting residuals...")
    plot_residuals(y_te, preds, 'Distribution GP (W2)', save_path='residuals_plot.png')

    # Plot 6: Q-Q plot for normality check
    print("Plotting Q-Q plot...")
    plot_qq(y_te, preds, stds, 'Distribution GP (W2)', save_path='qq_plot.png')

    # Quick Legendre baseline
    print("Fitting Legendre baseline...")
    Xl_tr = np.array([legendre_features(d, 5) for d in train_d])
    Xl_te = np.array([legendre_features(d, 5) for d in test_d])
    pl    = fit_baseline_gp(Xl_tr, y_tr)
    pl_preds = np.array([predict_baseline(Xl_te[i], Xl_tr, y_tr, pl)[0]
                          for i in range(len(test_d))])

    def rmse(a, b): return float(np.sqrt(np.mean((a-b)**2)))
    def cir(yt, yp, ys, a=0.9):
        q = norm.ppf(0.5+a/2)
        return float(np.mean(np.abs(yt-yp) <= q*ys))

    # Plot 3 & additional data: RMSE and CIR comparison
    dist_std = np.array([np.sqrt(max(0, v)) for v in stds])
    pl_stds = np.array([1.0] * len(pl_preds))  # dummy stds for baseline
    
    results = {
        'distribution':  {'RMSE': rmse(y_te, preds), 'CIR': cir(y_te, preds, dist_std)},
        'Legendre ord 5': {'RMSE': rmse(y_te, pl_preds), 'CIR': cir(y_te, pl_preds, pl_stds)},
    }

    print("Plotting RMSE comparison...")
    plot_rmse_bar(results, save_path='rmse_comparison.png')

    print("Plotting CIR comparison...")
    plot_cir_bar(results, save_path='cir_comparison.png')

    print("\n" + "="*50)
    print("All 6 plots generated successfully:")
    print("  1. fig1_densities.png      – Sample training distributions")
    print("  2. scatter_dist.png        – True vs predicted scatter")
    print("  3. rmse_comparison.png     – RMSE bar chart")
    print("  4. error_histogram.png     – Prediction error distribution")
    print("  5. residuals_plot.png      – Residuals plot")
    print("  6. qq_plot.png             – Q-Q plot for normality")
    print("="*50)
