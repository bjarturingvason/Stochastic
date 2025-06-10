import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2

N = 10_000
rng = np.random.default_rng()        # NumPy's MT19937-based generator
u = rng.random(N)                    # Uniform(0,1)

#2. χ² frequency test
k = 10
obs, _ = np.histogram(u, bins=k)
exp = np.full(k, N / k)
chi2_stat = ((obs - exp) ** 2 / exp).sum()
dof = k - 1


p_chi2 = 1 - chi2.cdf(chi2_stat, dof)


#KS
try:
    from scipy.stats import kstest
    ks_stat, p_ks = kstest(u, 'uniform')
except ImportError:
    ks_stat = max(abs(np.arange(1, N + 1) / N - np.sort(u)),
                  abs(np.sort(u) - np.arange(N) / N))
    p_ks = None

# ----------------- 4. Runs test -------------------------------------------------
median = 0.5
signs = np.where(u >= median, 1, 0)
runs = 1 + np.sum(signs[:-1] != signs[1:])
n1, n0 = signs.sum(), N - signs.sum()
expected_runs = 1 + 2 * n1 * n0 / N
var_runs = (2 * n1 * n0 * (2 * n1 * n0 - N)) / (N**2 * (N - 1))
z_runs = (runs - expected_runs) / math.sqrt(var_runs)

# ----------------- 5. Autocorrelations ------------------------------------------
def autocorr(x, h=1):
    x = x - x.mean()
    return np.dot(x[:-h], x[h:]) / np.dot(x, x)

lags = [1, 2, 3, 5, 10]
r = [autocorr(u, h) for h in lags]

# ----------------- 6. Plots ------------------------------------------------------
# Histogram
plt.figure()
plt.hist(u, bins=k, edgecolor='black')
plt.title("Histogram of 10 000 values – system RNG")
plt.xlabel("Value")
plt.ylabel("Frequency")

# Successive-pair scatter plot
plt.figure()
plt.scatter(u[:-1], u[1:], s=4)
plt.title("Successive pairs (u_n , u_{n+1}) – system RNG")
plt.xlabel("u_n")
plt.ylabel("u_{n+1}")

# Autocorrelation bar plot
plt.figure()
plt.bar(lags, r, width=0.8)
plt.axhline(0, linewidth=1)
plt.title("Autocorrelations – system RNG")
plt.xlabel("Lag (h)")
plt.ylabel("r(h)")

plt.show()

# ----------------- 7. Summary table ---------------------------------------------
summary = pd.DataFrame({
    "Test": ["χ² (10 bins)", "Kolmogorov–Smirnov", "Runs (z)"],
    "Statistic": [chi2_stat, ks_stat, z_runs],
    "p‑value": [p_chi2, p_ks, None]
})
print(summary)