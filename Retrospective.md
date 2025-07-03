
# Capstone Project: Bayesian Optimization

This project applied **Bayesian Optimization** to a range of real-world problems, evolving from random sampling to advanced acquisition strategies using custom Gaussian Process models. 

## Initial Codebase

**Notebook:** `function_1.ipynb`
**Source:** Custom-developed based on foundational examples from scikit-learn’s Gaussian Process documentation.

### Why This Starting Point?

* Began with a simple setup for **random sampling**.
* Used `ba_optimizer_v1.py`, a basic custom optimizer wrapper implementing **Expected Improvement (EI)**.
* Chosen for flexibility and ease of integration with multiple kernels and acquisition functions.

### Initial Code Sample:

```python
from ba_optimizer_v1 import BayesianOptimizer

optimizer = BayesianOptimizer(np.array(X, dtype=np.float64), y, bounds=(0, 1))
submission_data = optimizer.optimize_step(
    num_candidates=2000,
    acquisition_func='ei',
    kappa=0.7
)
```

**Rationale:** This initial structure enabled quick experiments while offering a familiar interface to integrate surrogate models and acquisition functions.

---

## Code Modifications (Week-by-Week Evolution)

### **Weeks 1–3: Random Inputs**

* Submitted purely **random samples** as inputs.
* Reason: Focused on building the pipeline and understanding the competition framework.
* Result: Low baseline score; useful for benchmarking.

---

### **Week 4–5: First Attempt at Expected Improvement**

* Implemented **EI acquisition** from scratch (outside optimizer).
* Realized poor performance due to numerical instability and lack of proper kernel configuration.

```python
def compute_expected_improvement(x):
    mu, sigma = gpr.predict([x], return_std=True)
    f_best = np.max(y)
    z = (mu - f_best) / sigma
    ei = (mu - f_best) * norm.cdf(z) + sigma * norm.pdf(z)
    return ei
```

* **Learning:** Need for robust kernel design and data scaling.

---

### **Week 6–10: Upper Confidence Bound (UCB) Implementation**

* Switched to **UCB**:

```python
mean, std = gpr.predict(X_grid, return_std=True)
ucb = mean + 0.76 * std
```

* Based on lecture material covering UCB's exploration-exploitation trade-off.
* **Result:** Slight performance bump, but inconsistency observed in noisy settings.

---

### **Week 11–15: Custom Kernel Tuning**

* Introduced **Gaussian Process kernels** explicitly:

```python
kernel = ConstantKernel(0.1, (1e-3, 1e1)) * RBF(length_scale=0.5) + WhiteKernel(noise_level=1e-4)
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
```

* Motivation: Improve model expressiveness and noise modeling.
* **Outcome:** Score began to improve consistently.

---

### **Week 16–20: Modularize with `ba_optimizer_v2`**

* Refactored into `ba_optimizer_v2.py`, supporting:

  * External kernel injection
  * Custom acquisition function switching (EI, UCB, PI)

```python
optimizer = BayesianOptimizer_v2(X, y, bounds=(0, 1))
new_submission = optimizer.optimize_step(
    num_candidates=100000,
    acquisition_func='ei',
    kappa=1.0,
    kernel=kernel
)
```

* **Learning:** Proper abstractions reduce boilerplate and help tune quickly.

---

### **Week 21–24: Advanced Acquisition Functions**

* Integrated:

  * **Thompson Sampling**
  * **Expected Loss Minimization**
  * **Probability of Improvement (PI)**
* Notebooks like `function_3.ipynb` (drug discovery) and `function_6.ipynb` (cake recipe) explored these options.
* **Result:** Problem-specific tuning (e.g., Matern vs. RBF kernels) led to clear gains.

---

### **Weeks 25–End: Final Optimizations**

* Focused on high-candidate searches (`num_candidates=100000`)
* Multiple kernel experiments for final tuning in `function_8.ipynb`.
* Used dimension-specific length scales in kernels for sensitivity analysis:

```python
kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0]*8)
```

* **Result:** Best scores across all tasks achieved during this window.

---

## Final Result & Reflections

### Final Weeks Summary

* Scores improved steadily after switching to:

  * `ba_optimizer_v2`
  * Custom GPR kernels
  * Robust EI implementations
* Highest-performing configurations used:

  * EI with large candidate sets
  * Tailored kernels per task

---

### What I Would Improve

* Add **Bayesian Optimization visualization tools** earlier (acquisition landscape, convergence plots).
* Integrate **automatic acquisition function selection** based on noise level or kernel performance.
* Introduce **data normalization** and **feature scaling** earlier to improve GPR stability.

---

### Key Learnings

* **Kernel choice** is critical to GPR success.
* Acquisition functions perform differently across domains—no one-size-fits-all.
* Modularizing the optimizer paid off significantly in later stages.

---

### For Next Time

* Start with a **modular optimizer design** from day one.
* Implement acquisition function visualizations early.
* Benchmark against standard optimizers like `skopt` or `GPyOpt`.


