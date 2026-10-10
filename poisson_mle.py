import numpy as np
from scipy.optimize import minimize_scalar

# 1. Population parameters and tuning curves
np.random.seed(42)
N, M, s_true, sigma, r_max = 60, 1000, 0.0, 1.0, 20.0
s_pref = np.linspace(-3.0, 3.0, N)


def tuning(s):
  return r_max * np.exp(-0.5 * ((s - s_pref) / sigma) ** 2)


# 2. Fisher Information and Cramer-Rao Lower Bound (CRLB)
rates = tuning(s_true)
fisher_info = np.sum(rates * ((s_true - s_pref) / (sigma**2)) ** 2)
crlb = 1.0 / fisher_info


# 3. Negative log-likelihood (Poisson noise)
def nll(s, k):
  lam = tuning(s)
  return np.sum(lam - k * np.log(lam))


# 4. Simulation: spike counts and MLE decoding across M trials
estimates = np.zeros(M)
for i in range(M):
  k = np.random.poisson(rates)
  res = minimize_scalar(nll, bounds=(-3.0, 3.0), args=(k,), method="bounded")
  estimates[i] = res.x

# 5. Compare empirical variance to CRLB
emp_var = np.var(estimates)
print(f"Empirical Variance : {emp_var:.6f}")
print(f"Cramer-Rao Bound   : {crlb:.6f}")
print(f"Ratio (Var / CRLB) : {emp_var / crlb:.4f}")