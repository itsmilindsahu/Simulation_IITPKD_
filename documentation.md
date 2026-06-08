# Simulation Study Documentation
## Replicating Bachoc et al. (2018) — *A Gaussian Process Regression Model for Distribution Inputs*
### IITPKD Summer Internship 2026 | Supervisor: Dr. Tanmay Sahoo

---

## What We Are Replicating

The paper proposes using Gaussian Processes (GP) to predict a scalar output when the **inputs are probability distributions** (not vectors). The core idea: replace the usual Euclidean distance between points with the **Wasserstein-2 (W₂) distance** between distributions when building the GP covariance kernel.

The simulation (Section VI of the paper) compares three models on 100 training distributions and 500 test distributions:
- **"distribution"** — GP with W₂-based kernel (their method)
- **"Legendre"** — GP on Legendre polynomial projections of the density
- **"PCA"** — GP on principal components of the discretised density

The paper's result (Table I): the distribution model achieves RMSE = 0.094, while Legendre and PCA stay well above 0.29 at best.

Our replication result: **RMSE ≈ 0.169**, confirming the paper's qualitative claim that the W₂-based GP strongly outperforms projection-based methods.

---

## Repository Structure

```
Simulation_IITPKD_/
├── data_generation.py   — Generate synthetic distributions (Section VI-A of paper)
├── wasserstein.py       — W₂ distance via quantile functions (Section III-a)
├── kernels.py           — Power-exponential kernel family (eq. 14 of paper)
├── gp_regression.py     — MLE parameter fitting + Kriging prediction (Section V-A)
├── baselines.py         — Legendre and PCA baseline models
├── run_simulation.py    — Main pipeline: generates Table I results
└── plot_results.py      — 6 publication-quality diagnostic plots
```

---

## Step 1: Generating Distributions — `data_generation.py`

**What the paper says (Section VI-A):**
Each distribution νᵢ is built by:
1. Sampling μᵢ ~ Uniform(0.3, 0.7) and σᵢ ~ Uniform(0.001, 0.2)
2. Taking the Gaussian density fᵢ = N(μᵢ, σᵢ²) on [0, 1]
3. Perturbing it with a GP sample: gᵢ(x) = fᵢ(x) · exp(Zᵢ(x)), where Zᵢ ~ GP(0, Matérn-5/2)
4. Normalising: νᵢ has density gᵢ / (∫gᵢ dx)

This gives distributions that are not restricted to a parametric family — they have random asymmetries and shapes, making linear projections suboptimal.

**The Matérn-5/2 kernel used for perturbation:**

```python
def matern52_cov(x1, x2, length_scale=0.2):
    r = np.abs(x1[:, None] - x2[None, :])      # pairwise distances
    s = np.sqrt(5) * r / length_scale
    C = (1 + s + s**2 / 3) * np.exp(-s)        # Matern-5/2 formula
    return C
```

Why Matérn-5/2? It produces smooth but not infinitely differentiable paths — matches the paper's choice of σ=1, ℓ=0.2.

**Sampling a GP and building the perturbed density:**

```python
C = matern52_cov(x_grid, x_grid, length_scale=0.2)
C += 1e-8 * np.eye(n_grid)          # small nugget for numerical stability
L = np.linalg.cholesky(C)           # Cholesky decomposition: C = L Lᵀ
z_i = L @ rng.standard_normal(n_grid)   # one GP sample: z ~ N(0, C)

g_i = f_i * np.exp(z_i)            # perturbed (always positive)
norm_const = trapezoid(g_i, x_grid) # integrate to normalise
density = g_i / norm_const
```

Key insight: multiplying by `exp(z_i)` keeps the density positive everywhere. The `trapezoid` rule (scipy) approximates the normalisation integral numerically.

**The target function F(ν):**

```python
def target_function(samples):
    m1 = np.mean(samples)           # first moment = mean
    m2 = np.mean(samples ** 2)      # second moment
    variance = max(0.0, m2 - m1**2) # clamp to avoid sqrt of negative
    F_val = m1 / (0.05 + np.sqrt(variance))
    return F_val
```

This is equation F(ν) = m₁(ν) / (0.05 + √(m₂(ν) − m₁(ν)²)) directly from the paper. It mixes the mean and standard deviation in a nonlinear way — a function that W₂ measures similarity for naturally, but L² projections of densities do not.

---

## Step 2: Wasserstein-2 Distance — `wasserstein.py`

**The key mathematical fact (paper eq. 2 and 24):**

For 1D distributions, W₂ has a clean closed form:

W₂²(μ, ν) = ∫₀¹ (F⁻¹_μ(t) − F⁻¹_ν(t))² dt

i.e. the L² distance between the **quantile functions** (inverse CDFs), not the densities. This is the optimal coupling result: evaluating both quantile functions at the same uniform u gives the cheapest transport plan.

**Estimating the quantile function from samples:**

```python
def quantile_function(samples, t_grid):
    sorted_s = np.sort(samples)         # sorted samples = empirical quantile function
    n = len(sorted_s)
    levels = np.linspace(0, 1, n)       # quantile levels k/(n-1)
    q_vals = np.interp(t_grid, levels, sorted_s)   # interpolate at requested t
    return q_vals
```

Sorting the samples directly gives the empirical quantile function — no density estimation needed.

**Computing W₂:**

```python
def wasserstein2(samples_mu, samples_nu, n_grid=200):
    t_grid = np.linspace(0.01, 0.99, n_grid)   # avoid endpoints
    q_mu = quantile_function(samples_mu, t_grid)
    q_nu = quantile_function(samples_nu, t_grid)
    w2_squared = np.mean((q_mu - q_nu) ** 2)   # discrete integral
    return np.sqrt(w2_squared)
```

Both quantile functions are evaluated on the **same** t_grid — this is exactly the optimal coupling from the paper. The mean replaces the continuous integral.

**Building the full pairwise distance matrix:**

```python
def pairwise_w2_matrix(distributions, n_grid=200):
    n = len(distributions)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):           # upper triangle only
            d = wasserstein2(distributions[i], distributions[j], n_grid)
            D[i, j] = d
            D[j, i] = d                     # W₂ is symmetric
    return D
```

This is an O(N²) operation — for N=100 training distributions it computes 4,950 pairwise distances. This matrix is then used by the kernel to build the GP covariance matrix.

---

## Step 3: The Kernel — `kernels.py`

**Paper eq. (14) — Power-exponential kernel:**

K_{σ², ℓ, H}(μ, ν) = σ² · exp( −W₂(μ, ν)^{2H} / ℓ )

Three parameters: σ² (variance), ℓ (length scale), H ∈ (0,1] (smoothness exponent).

- H = 1/2 → Laplace / exponential kernel
- H = 1 → Gaussian / squared-exponential kernel

The paper proves (Theorem IV.2) this is a valid positive-definite kernel on the Wasserstein space for any completely monotone function F, which `exp(−·)` satisfies.

**Implementation:**

```python
def kernel_power_exp(D, sigma2, ell, H):
    # D is the (n,n) pairwise W2 distance matrix
    K = sigma2 * np.exp(-(D ** (2*H)) / ell)
    return K

def add_nugget(K, delta=1e-6):
    # Small diagonal addition for numerical positive-definiteness
    return K + delta * np.eye(len(K))
```

The nugget (δ on the diagonal) ensures the covariance matrix is strictly positive-definite, preventing Cholesky failures during inversion.

---

## Step 4: GP Regression — `gp_regression.py`

### 4a. Maximum Likelihood Estimation

**Paper eq. (19) — the negative log-likelihood to minimise:**

L_θ = (1/n) · [log det R_θ + yᵀ R_θ⁻¹ y]

where R_θ = [K_θ(μᵢ, μⱼ)]_{i,j} is the n×n training covariance matrix.

**Key trick — unconstrained parametrisation:**

```python
sigma2 = np.exp(params[0])              # always positive
ell    = np.exp(params[1])              # always positive
H      = 1 / (1 + np.exp(-params[2]))  # sigmoid maps R → (0,1)
```

We optimise over (log σ², log ℓ, logit H) — all unconstrained reals — then transform back. This lets `scipy.minimize` (L-BFGS-B) work without explicit bounds.

**Computing the NLL efficiently using Cholesky:**

```python
def neg_log_likelihood(params, D_train, y_train, delta=1e-6):
    # ... build K ...
    c, low = cho_factor(K)                         # K = L Lᵀ
    log_det = 2 * np.sum(np.log(np.diag(c)))       # log det K = 2 Σ log L_ii
    alpha = cho_solve((c, low), y_train)            # K⁻¹ y via triangular solves
    quad = y_train @ alpha                          # yᵀ K⁻¹ y
    nll = (log_det + quad) / n
    return nll
```

Cholesky is preferred over direct inversion: it costs O(N³) once, then O(N²) per solve, and is numerically stable.

**Multi-start optimisation:**

```python
def fit_gp(D_train, y_train, n_restarts=5):
    best_nll = np.inf
    for _ in range(n_restarts):
        x0 = rng.uniform([-2, -2, -2], [2, 2, 2])   # random start
        res = minimize(neg_log_likelihood, x0, ...)
        if res.fun < best_nll:
            best_nll = res.fun
            best_raw = res.x
    return decode(best_raw)   # sigma2, ell, H
```

Multiple restarts guard against local minima in the non-convex likelihood surface.

### 4b. Kriging Prediction

**Paper eq. (20) — posterior mean:**

Ŷ_θ(μ*) = r_θ(μ*)ᵀ R_θ⁻¹ y

where r_θ(μ*)ᵢ = K_θ(μ*, μᵢ) is the cross-covariance vector between the test point and all training points.

**Posterior variance (uncertainty quantification):**

Var_θ(μ*) = K_θ(μ*, μ*) − r_θ(μ*)ᵀ R_θ⁻¹ r_θ(μ*)

```python
def predict(dist_test, dists_train, y_train, params, D_train):
    # Cross-covariance: distance from test to each training distribution
    r = np.array([
        kernel_power_exp_single(wasserstein2(dist_test, dists_train[i]), ...)
        for i in range(n)
    ])
    c, low = cho_factor(K_train)
    alpha = cho_solve((c, low), y_train)   # K⁻¹ y
    y_pred = float(r @ alpha)              # posterior mean

    v = cho_solve((c, low), r)             # K⁻¹ r
    y_var = float(max(0.0, sigma2 - r @ v))  # posterior variance, clamped ≥ 0
    return y_pred, y_var
```

The clamping to 0 handles tiny negative values from floating-point rounding.

---

## Step 5: Baseline Models — `baselines.py`

### Legendre Projection
For each distribution νᵢ with density f_νᵢ on [0, 1], compute projections onto normalised Legendre polynomials p₀, p₁, ..., p_{o−1}:

aᵢₖ = ∫₀¹ f_νᵢ(t) · pₖ(t) dt

The GP then operates on the o-dimensional feature vector (aᵢ₀, ..., aᵢ,_{o-1}) with a standard power-exponential kernel. Tested at orders 5, 10, 15.

### PCA Projection
Discretise each density onto a grid of d=100 points, then take the first o principal components. The GP operates on the PCA projection vector. Same kernel family as Legendre.

**Why these fail:** Both projections measure L² similarity between density functions. Two distributions with similar means but very different variances will have very different L² projections, even though they may be close in W₂ distance (and therefore have similar target values F(ν)).

---

## Step 6: Metrics — `run_simulation.py`

**RMSE (Root Mean Squared Error):**

RMSE² = (1/nₜ) Σᵢ [F(νₜ,ᵢ) − F̂(νₜ,ᵢ)]²

Should be as small as possible.

**CIR₀.₉ (Confidence Interval Ratio at 90%):**

CIR₀.₉ = (1/nₜ) Σᵢ 1{ |F(νₜ,ᵢ) − F̂(νₜ,ᵢ)| ≤ q₀.₉₅ · σ̂(νₜ,ᵢ) }

Should be close to 0.9. This checks whether the GP's predicted uncertainty σ̂ is well-calibrated — i.e., do 90% of true values actually fall within the 90% confidence interval?

```python
def compute_metrics(y_true, y_pred, y_std, alpha=0.9):
    errors = y_true - y_pred
    rmse = np.sqrt(np.mean(errors**2))

    q = norm.ppf(0.5 + alpha/2)       # 1.645 for alpha=0.9
    in_interval = np.abs(errors) <= q * y_std
    cir = np.mean(in_interval)
    return rmse, cir
```

---

## Results

Our replication of Table I:

| Model               | RMSE  | CIR₀.₉ |
|---------------------|-------|---------|
| **distribution (W₂)** | **0.169** | **0.87** |
| Legendre order 5    | 0.956 | 0.84 |
| PCA order 5         | 0.986 | 0.77 |

Paper's Table I values:

| Model               | RMSE  | CIR₀.₉ |
|---------------------|-------|---------|
| **distribution (W₂)** | **0.094** | 0.92 |
| Legendre order 5    | 0.49  | 0.92 |
| Legendre order 15   | 0.29  | 0.91 |
| PCA order 5         | 0.63  | 0.82 |

**Why our RMSE is higher than the paper's 0.094:** The paper used 500 samples per distribution for the target function estimate and ran more optimisation restarts. With 200 samples and fewer restarts, variance in the target estimate introduces noise. However, the key qualitative result — W₂-based GP outperforms projection baselines by a factor of 5× in RMSE — is clearly replicated.

MLE converged parameters in our run:
- σ² ≈ 56.49 (output variance scale)
- ℓ ≈ 0.157 (W₂ correlation length)
- H ≈ 0.98 (close to 1 → near-Gaussian kernel)

---

## The Full Pipeline (Conceptual Flow)

```
1. generate_dataset(n=100, seed=0)
        ↓
   [dist_1, dist_2, ..., dist_100], [F(ν_1), ..., F(ν_100)]

2. pairwise_w2_matrix(train_distributions)
        ↓
   D_train : 100×100 matrix of W₂ distances

3. fit_gp(D_train, y_train)
        → MLE over (σ², ℓ, H) using L-BFGS-B with 5 restarts
        ↓
   best_params : {sigma2, ell, H}

4. For each test distribution ν*:
        w2_to_train = [W₂(ν*, νᵢ) for i in 1..100]
        r = kernel_power_exp(w2_to_train, best_params)
        y_pred, y_var = Kriging formula
        ↓
   (y_pred, y_var) for each of 500 test points

5. compute_metrics(y_true, y_pred, sqrt(y_var))
        ↓
   RMSE, CIR₀.₉
```

---

## Key Mathematical Connections (Paper → Code)

| Paper equation | Code location | What it does |
|---|---|---|
| W₂(μ,ν) = √∫(q_μ−q_ν)² dt  (eq. 2, 24) | `wasserstein.py: wasserstein2()` | Core distance metric |
| K_{σ²,ℓ,H}(μ,ν) = σ²·exp(−W₂^{2H}/ℓ)  (eq. 14) | `kernels.py: kernel_power_exp()` | GP covariance kernel |
| L_θ = log det R + yᵀR⁻¹y  (eq. 19) | `gp_regression.py: neg_log_likelihood()` | MLE objective |
| Ŷ(μ*) = rᵀ R⁻¹ y  (eq. 20) | `gp_regression.py: predict()` | Kriging prediction |
| W₂^{2H} is negative definite iff H ∈ (0,1]  (Thm IV.3) | Enforced via sigmoid on H in MLE | Guarantees valid kernel |

---

## Why This Works (Intuition)

Standard GP kernels measure similarity via Euclidean distance between feature vectors. When inputs are distributions, Euclidean distance between density values (L² distance) is a poor proxy for how "similar" two distributions really are — it depends heavily on where the mass is, not just how much there is.

W₂ measures the minimum cost to transport one distribution's mass into the shape of the other. Two narrow Gaussians with slightly different means are very close in W₂ (a small shift moves the mass cheaply), and they also have similar target values F(ν). Two Gaussians with very different widths are far in W₂. L² projections can confuse these cases.

By plugging W₂ into a standard GP kernel framework, we get a model that inherits all of GP's uncertainty quantification and MLE apparatus, while using a geometrically appropriate notion of similarity for distribution inputs.

---

## Dependencies

```
numpy >= 1.23     — numerical arrays, linear algebra
scipy >= 1.9      — trapezoid integration, L-BFGS-B optimiser, norm.ppf
matplotlib >= 3.6 — all plots
```

Run the simulation:
```
python run_simulation.py   # prints Table I to console (~30 sec)
python plot_results.py     # saves 6 .png diagnostic figures
```

---

*Simulation study completed June 2026 | IITPKD Summer Internship | GitHub: itsmilindsahu/Simulation_IITPKD_*
