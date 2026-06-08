# 🎯 How We Taught a Computer to Predict Things About Distributions
### A slide-by-slide walkthrough — no scary math required

---

## Slide 1 — What Problem Are We Solving?

Imagine you have a bag of marbles. Some bags have marbles bunched in the middle. Some are spread out. Some are lopsided.

**Your job:** Look at a bag → Predict a number we care about.

That number is:

> **"The average marble position ÷ (a tiny bit + how spread-out the bag is)"**

If the marbles are all in the middle → small spread → big number.  
If the marbles are all over the place → big spread → small number.

We want a computer to **learn this rule from examples**, without being told the formula.

---

## Slide 2 — What is a "Distribution Input"?

Normal machine learning:
> Input = a list of numbers like `[1.2, 3.4, 0.7]`

Our problem:
> Input = **a whole bag of marbles** (a probability distribution)

We can't just feed a bag into a standard model. We need a smarter way to compare bags.

**Key question:** When are two bags "similar"?

---

## Slide 3 — Two Ways to Compare Bags

**Bad way — compare the shapes of the histograms directly (L² distance)**

Imagine two bags:
- Bag A: marbles bunched around position 0.3
- Bag B: marbles bunched around 0.35

Their histogram *shapes* look almost the same, but shifted a tiny bit. L² distance says they're *very different* because the bar heights at each position differ a lot.

**Good way — Wasserstein-2 (W₂) distance**

> "How much work does it take to *move* Bag A's marbles to match Bag B?"

A tiny shift costs almost nothing. That's a small W₂ distance. ✅

Two bags with very different spreads cost a lot to rearrange. That's a big W₂ distance. ✅

W₂ matches our *intuition* about similarity far better than comparing histogram bars.

---

## Slide 4 — How We Actually Compute W₂

**The magic trick:** Sort the marbles.

The sorted list of marbles **is** the quantile function (fancy name: inverse CDF).

W₂ between two bags = the average squared difference between their **sorted marble lists**.

```python
def wasserstein2(samples_mu, samples_nu, n_grid=200):
    t_grid = np.linspace(0.01, 0.99, n_grid)

    # Sort each bag → that's the quantile function
    q_mu = quantile_function(samples_mu, t_grid)
    q_nu = quantile_function(samples_nu, t_grid)

    # Average squared gap between sorted lists
    w2_squared = np.mean((q_mu - q_nu) ** 2)
    return np.sqrt(w2_squared)
```

> **In plain English:** Line up both bags from smallest to largest. At each rank (1st, 2nd, 3rd marble...), measure the gap. Average all those gaps. That's W₂.

No density estimation. No grids. Just sort and compare. 🎉

---

## Slide 5 — Building the Distance Table

We have 100 training bags. We compare **every pair**.

```python
def pairwise_w2_matrix(distributions, n_grid=200):
    n = len(distributions)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = wasserstein2(distributions[i], distributions[j])
            D[i, j] = d
            D[j, i] = d   # distance is symmetric
    return D
```

Result: a 100×100 table where `D[i][j]` = "how different are Bag i and Bag j?"

> Think of it like a travel-time table between 100 cities. Once you have it, you never need to recompute.

---

## Slide 6 — Gaussian Processes in 30 Seconds

A **Gaussian Process (GP)** is a model that says:

> "Nearby inputs should have nearby outputs. The more similar two inputs are, the more similar I expect their outputs to be."

It's like saying: *"If Bag A and Bag B look almost the same, their predicted numbers should be almost the same too."*

The GP doesn't use a fixed formula. It's a **flexible interpolator** that also tells you **how uncertain it is** about each prediction.

Two things the GP needs:
1. A way to measure "how similar are two inputs?" → **the kernel**
2. Training examples → **our 100 (bag, number) pairs**

---

## Slide 7 — The Kernel: Turning Distance Into Similarity

We have W₂ distances. The kernel converts a distance into a similarity score (between 0 and 1):

```
K(Bag_i, Bag_j) = σ² × exp( − W₂(i, j)^(2H) / ℓ )
```

Three knobs:
- **σ²** — how much the outputs vary overall (output scale)
- **ℓ** — how far apart two bags can be and still be "similar" (length scale)
- **H** — how smooth the relationship is (H ≈ 1 → very smooth, H = 0.5 → rougher)

```python
def kernel_power_exp(D, sigma2, ell, H):
    K = sigma2 * np.exp(-(D ** (2*H)) / ell)
    return K
```

> **In plain English:** Bags with small W₂ distance → kernel score near 1 (very similar).  
> Bags far apart → kernel score near 0 (not similar). This tells the GP who to "listen to."

---

## Slide 8 — Learning the Knobs (MLE)

We don't guess σ², ℓ, H. We **learn** them from data.

We find the knob settings that make the training data most "likely" under the GP. This is called **Maximum Likelihood Estimation (MLE)**.

```python
def neg_log_likelihood(params, D_train, y_train, delta=1e-6):
    sigma2 = np.exp(params[0])               # keep positive
    ell    = np.exp(params[1])               # keep positive
    H      = 1 / (1 + np.exp(-params[2]))   # keep between 0 and 1

    K = kernel_power_exp(D_train, sigma2, ell, H)
    K += delta * np.eye(len(K))              # small safety padding

    # Cholesky: efficient and stable way to invert K
    c, low = cho_factor(K)
    log_det = 2 * np.sum(np.log(np.diag(c)))
    alpha   = cho_solve((c, low), y_train)
    quad    = y_train @ alpha

    return (log_det + quad) / len(y_train)   # smaller = better fit
```

We run an optimiser (`L-BFGS-B`) 5 times from random starting knob values, take the best result.

Our best knobs: **σ² ≈ 56.5, ℓ ≈ 0.157, H ≈ 0.98**

---

## Slide 9 — Making a Prediction (Kriging)

New bag comes in. We've never seen it. What's its number?

**Step 1:** Compute W₂ from the new bag to all 100 training bags.  
**Step 2:** Turn those distances into similarity scores using the kernel → vector `r`.  
**Step 3:** The prediction is a weighted average of training outputs, where weights come from `r`.

```python
def predict(dist_test, dists_train, y_train, params, K_train):
    # Similarity of new bag to each training bag
    r = np.array([
        kernel_single(wasserstein2(dist_test, dists_train[i]), params)
        for i in range(len(dists_train))
    ])

    c, low = cho_factor(K_train)
    alpha  = cho_solve((c, low), y_train)   # learned weights

    y_pred = float(r @ alpha)               # weighted sum → prediction

    v      = cho_solve((c, low), r)
    y_var  = float(max(0.0, params['sigma2'] - r @ v))  # uncertainty estimate
    return y_pred, y_var
```

> **In plain English:** "The new bag looks most like Bag 17 and Bag 42. So my prediction leans heavily on their outputs. And here's how confident I am."

---

## Slide 10 — The Baselines (What We're Beating)

**Legendre projection:** Describe each bag by 5 or 15 numbers (how much it "looks like" each polynomial shape). Feed those numbers to a standard GP.

**PCA projection:** Squish each bag's histogram down to its top 5 principal components. Feed those to a standard GP.

**The problem with both:** They measure L² similarity between *density shapes*, not W₂ similarity. Two bags with the same mean but different spreads look very *different* under L² — but have similar F(ν) values. The projection GPs get confused. Our W₂ GP does not.

---

## Slide 11 — How We Made the Fake Bags (Data Generation)

Each bag is a random bumpy distribution on [0, 1]:

1. Pick a random centre μ ~ Uniform(0.3, 0.7) and width σ ~ Uniform(0.001, 0.2)
2. Start with a Gaussian bump: `f = Normal(μ, σ²)`
3. Add random "waviness" using a GP sample `z`: `g = f × exp(z)`
4. Normalise so it sums to 1 → that's your bag's density

```python
# Step 3: the waviness
C  = matern52_cov(x_grid, x_grid, length_scale=0.2)
C += 1e-8 * np.eye(n_grid)           # numerical safety
L  = np.linalg.cholesky(C)
z_i = L @ rng.standard_normal(n_grid)  # one random wiggly curve

g_i = f_i * np.exp(z_i)             # always positive!
density = g_i / trapezoid(g_i, x_grid)  # normalise to area = 1
```

Why `exp(z)`? Multiplying by exp of anything keeps values positive — a density can never go negative.

The result: bags that are Gaussian-ish but with random asymmetries and bumps. No two are alike.

---

## Slide 12 — The Target Number F(ν)

Once we have a bag's samples, we compute:

```python
def target_function(samples):
    m1 = np.mean(samples)            # average marble position
    m2 = np.mean(samples ** 2)       # average of squared positions
    variance = max(0.0, m2 - m1**2)  # spread² (clamped ≥ 0)
    return m1 / (0.05 + np.sqrt(variance))
```

> Mean divided by (a tiny constant + standard deviation)

If marbles are centred high AND tightly packed → big number.  
If marbles are spread all over → small number.

This is deliberately a nonlinear mix of mean and spread — something W₂ captures naturally.

---

## Slide 13 — Did It Work? The Results

We trained on **100 bags**, predicted on **500 new bags**, measured RMSE (average prediction error).

| Model | RMSE | Notes |
|---|---|---|
| **Our W₂ GP** | **0.169** | ✅ Best by far |
| Legendre (order 5) | 0.956 | ~5.6× worse |
| PCA (order 5) | 0.986 | ~5.8× worse |

**Paper's numbers:** their W₂ GP got 0.094 (we got 0.169 — we used fewer samples per bag, so more noise in the target estimates). But the qualitative story is the same: **W₂-based GP crushes projection baselines**.

We also checked calibration (CIR₀.₉): we got **0.87**, meaning 87% of true values fell inside the 90% confidence interval. Pretty well calibrated.

---

## Slide 14 — The Full Pipeline in One Picture

```
100 random bumpy bags  ──┐
                         ├─► pairwise W₂ distances (100×100 table)
                         │
                         └─► MLE: learn σ², ℓ, H from training data
                                        │
500 new test bags ──────────────────────┴─► Kriging: predict + uncertainty
                                                      │
                                               RMSE = 0.169 ✅
```

Every step maps to one file:

| What | File |
|---|---|
| Make the bags | `data_generation.py` |
| Compute W₂ | `wasserstein.py` |
| Build the kernel | `kernels.py` |
| Learn the knobs + predict | `gp_regression.py` |
| Legendre & PCA baselines | `baselines.py` |
| Run everything + print table | `run_simulation.py` |
| Draw the 6 diagnostic plots | `plot_results.py` |

---

## Slide 15 — The One Sentence Summary

> We replaced "how different do these histograms look?" with "how much work does it take to move one bag of marbles into the other?" — and that one change made our predictions **5× more accurate**.

---

*IITPKD Summer Internship 2026 | Supervisor: Dr. Tanmay Sahoo | github.com/itsmilindsahu/Simulation_IITPKD_*
