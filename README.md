# Statistical Learning & Neural Coding Simulations

This repository contains independent, self-contained Python simulations exploring fundamental limits in estimation, selective inference, and generalization bounds. These scripts were developed as theoretical proofs-of-concept for applications in computational neuroscience and machine learning.

## 1. Neural Decoding and the Cramér-Rao Bound
**File:** `poisson_mle.py`
Simulates a population of Poisson neurons with Gaussian tuning curves to test the precision of a Maximum Likelihood Estimation (MLE) decoder.
*   **Result:** The empirical variance of the MLE decoder (0.002134) tightly bounds the theoretical Cramér-Rao lower limit (0.002081) calculated via Fisher Information, yielding a ratio of 1.0252[cite: 1].

## 2. Post-Selection Inference with HSIC-Lasso
**File:** `Post-Selection Inference with HSIC-Lasso.py`
Demonstrates the necessity of valid inference techniques after feature selection. It uses HSIC-Lasso on synthetic pure-noise data to compare naive p-values against a data-splitting approach.
*   **Result:** Naive testing on the same data used for selection severely inflates the Type I error rate to 13.3%, while valid data-splitting correctly controls the False Positive Rate at the target 5% alpha level (measured at 4.6%)[cite: 2].

## 3. Generalization Bounds in SGLD
**File:** `SGLD.py`
Trains a 2-layer multi-layer perceptron on synthetic data using Stochastic Gradient Langevin Dynamics (SGLD) to measure the empirical generalization gap against information-theoretic limits.
*   **Result:** Calculates an expected trajectory mutual-information bound (2.1635) and compares it to the network's empirical generalization gap (3.2890) between train and test loss[cite: 3].