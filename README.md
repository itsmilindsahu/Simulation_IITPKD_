# Gaussian Process Regression for Distribution Inputs
## IITPKD Summer Internship 2026 – Simulation Study

Replication of the simulation experiments from *"A Gaussian Process Regression Model for Distribution Inputs"* by Bachoc et al. (2018).

---

## Project Structure

| Module | Purpose |
|--------|---------|
| `data_generation.py` | Synthetic distribution generation with GP perturbations |
| `gp_regression.py` | GP model: MLE fitting and Kriging prediction |
| `kernels.py` | Power-exponential kernel and utilities |
| `wasserstein.py` | Wasserstein-2 distance computations |
| `baselines.py` | Legendre and PCA projection baselines |
| `plot_results.py` | Diagnostic and comparison plots |
| `run_simulation.py` | Main entry point — replicates Table I |

---

## Installation

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Dependencies: `numpy>=1.23`, `scipy>=1.9`, `matplotlib>=3.6`

---

## Usage

```bash
python run_simulation.py   # Table I replication
python plot_results.py     # generate diagnostic plots
```

Results will vary slightly between runs due to random seed and optimizer non-determinism.
See paper Table I for reference values (n=100 train, n=500 test).

**Generated plots:** `fig1_densities.png`, `scatter_dist.png`, `rmse_comparison.png`,
`error_histogram.png`, `residuals_plot.png`, `qq_plot.png`

---

## Troubleshooting

**`AttributeError: module 'numpy' has no attribute 'trapezoid'`**
→ `pip install "scipy>=1.9"`

**Slow runtime**
→ Reduce `N_TEST` or `n_restarts` in the respective config variables.

---

## References

1. Bachoc, F., et al. (2018). "A Gaussian Process Regression Model for Distribution Inputs."
2. Villani, C. (2009). *Optimal Transport: Old and New*. Springer.
3. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.

---

**Institution**: IIT Palakkad | **Supervisor**: Dr. Tanmay Sahoo | **Updated**: June 2026
