## `function_1.ipynb` – Evolution of Strategy

* **Start:** Used random inputs; early attempt at Expected Improvement (EI) was ineffective.
* **Middle:** Switched to Upper Confidence Bound (UCB), but results were suboptimal.
* **Later:** Adopted Expected Improvement with custom kernels using `ba_optimizer_v2`, improving performance.

**Pivot to EI**: The most significant change, aligning with the README, is the shift to Expected Improvement (EI) as the acquisition function, particularly with the introduction and use of `ba_optimizer_v2` which allows for more advanced kernel handling and a refined EI calculation (`expected_improvement_v2`).

In essence, `function_1.ipynb` visually and programmatically traces the iterative process of refining the Bayesian Optimization strategy, from initial random exploration to more sophisticated acquisition functions and kernel choices, mirroring the learning and adaptation described in the project's README.

---

## `function_2.ipynb` – Core Bayesian Optimization Logic

* Implements key methods:

  * `_expected_improvement` – Calculates EI.
  * `_propose_next_point` – Suggests next sample based on EI.
* Demonstrates iterative sampling and model updates to converge on optimal solutions.

---

## `function_3.ipynb` – Drug Discovery Application

* Goal: Minimize adverse reaction from compound combinations.
* Uses:

  * Gaussian Process Regression with Matern kernel.
  * Acquisition strategies: **Thompson Sampling**, **Expected Loss**, and **Expected Improvement**.


### Iterative Optimization and Visualization:
The notebook demonstrates the iterative nature of Bayesian Optimization by showing how the GPR model is updated with new data and how the acquisition functions are used to suggest the next query point. The 3D scatter plot visualizes the input compounds and their corresponding adverse reactions, providing a visual representation of the search space and the observed outcomes. The output also highlights the point with the 'y closest to 0', indicating the best-found drug combination so far.

In summary, `function_3.ipynb` showcases the practical application of Gaussian Process Regression and various acquisition functions for optimizing a real-world problem, emphasizing the iterative refinement of the model and the strategic selection of new data points.

---

## `function_4.ipynb` – Tuning ML Model Approximation

* Problem: Match a fast model’s output to an expensive one.
* Optimization targets 4 hyperparameters.
* Uses both `ba_optimizer_v1` and `v2` with UCB and EI.
* Focused on minimizing prediction error.

### Iterative Observation and Re-optimization:
The notebook simulates an iterative optimization loop by:
1.  Adding new observations (e.g., `input_may_27`, `output_may_27`, `input_jun_2`, `output_jun_2`) to the dataset. These represent the results of evaluating the model with new hyperparameters.
2.  Re-initializing and re-fitting the `BayesianOptimizer_v2` with the updated `X` and `y` data.
3.  Performing another optimization step using either UCB or EI to recommend the next set of hyperparameters. This mimics the real-world scenario where new data from experiments or simulations is used to refine the optimization process.

### Focus on Maximization:
Unlike some Bayesian optimization problems that aim for minimization, this notebook implicitly focuses on maximization. The output `y` represents the difference between the expensive calculation and the model's output. Therefore, minimizing this difference (making it closer to zero) would be the goal. However, the use of `np.max(y)` and the positive acquisition values (for UCB and EI) suggest that the optimization is framed as maximizing a negative difference or minimizing a positive difference. The goal is to find hyperparameters that lead to the smallest absolute difference, ideally pushing the `y` values towards zero.

---

## `function_5.ipynb` – Maximizing Chemical Yield

* Problem: 4D unimodal black-box function.
* Uses:

  * Matern kernel (`nu=2`) for smooth modeling.
  * EI as the main acquisition function.
* Iteratively adds new observations to refine optimization.

---

## `function_6.ipynb` – Cake Recipe Optimization

* Objective: Minimize negative scores (flavor, calories, etc.) → score close to zero.
* 5D input space with per-dimension RBF kernel.
* Applies **Probability of Improvement** and **Expected Improvement**.
* Refines recipe through iterative updates.

---

## `function_7.ipynb` – ML Hyperparameter Tuning

* Problem: Optimize 6 hyperparameters for a known ML model.
* Uses:

  * Matern kernel (`nu=1.5`)
  * UCB and EI acquisition functions.
* Demonstrates progressive learning and model improvement over iterations.

---

## `function_8.ipynb` – High-Dimensional Optimization

* Problem: 8D black-box optimization.
* Uses EI and a custom kernel to handle complexity.
* Focuses on local optima due to high dimensionality.
* Iterative approach with feedback from each step.



