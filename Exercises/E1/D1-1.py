import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
import pandas as pd
import math as math
from scipy.stats import kstest, chi2

#THe LCG
def LCG(X0, m, a, c, N):
    X = np.zeros(N)
    X[0] = X0
    for i in range(1,N):
        X[i] = (a*X[i-1]+c) % m
    return X

# 1a)
X0 = int(1)
m = int(2**32)
a = int(1664525)
c = int(1013904223)
N = int(10000)

X = LCG(X0, m, a, c, N)

plt.hist(X,bins=10, rwidth=0.6)
plt.savefig('Figures/Hist-1a.png')
plt.show()

#1b)
fig, ax = plt.subplots()
ax.scatter(X[:-1], X[1:], s= 3)        # (u_n , u_{n+1})
ax.set_title("Successive pairs (uₙ , uₙ₊₁)")
ax.set_xlabel("uₙ")
ax.set_ylabel("uₙ₊₁")
plt.savefig('Figures/Pairscatter-1b')
plt.show()

# ---------- χ² frequency test ----------
k = 10                          # number of bins
observed, _ = np.histogram(X, bins=k)
expected = np.full(k, N / k)
chi2_stat = ((observed - expected) ** 2 / expected).sum()
dof = k - 1

p_chi2 = 1 - chi2.cdf(chi2_stat, dof)
print(f'p-value for the Chisq test:{p_chi2}')

median = 0.5
signs = np.where(X >= median, 1, 0)
runs = 1 + np.sum(signs[:-1] != signs[1:])
n1, n0 = signs.sum(), N - signs.sum()
expected_runs = 1 + 2 * n1 * n0 / N
var_runs = (2 * n1 * n0 * (2 * n1 * n0 - N)) / (N**2 * (N - 1))
z_runs = (runs - expected_runs) / math.sqrt(var_runs)


s = pd.Series(X)
lags = range(0,10)
r = [s.autocorr(lag=lag) for lag in lags]

plt.figure()
plt.bar(lags, r, width=0.2)
plt.axhline(0, linewidth=1)
plt.title("Sample autocorrelation")
plt.xlabel("Lag (h)")
plt.ylabel("r(h)")
plt.savefig('Figures/Autocorrelation.png')
plt.show()


import math
from scipy.stats import kstest, chi2

# --- Modified LCG function to generate batches ---
def test_lcg_batches(X0, m, a, c, N, num_batches=20):
    results = []
    for _ in range(num_batches):
        X0 = np.random.randint(1,m)
        X = LCG(X0, m, a, c, N)
        u = X / m
        # Chi-squared test
        k = 10
        observed, _ = np.histogram(u, bins=k)
        expected = np.full(k, N / k)
        chi2_stat = ((observed - expected) ** 2 / expected).sum()
        p_chi2 = 1 - chi2.cdf(chi2_stat, k-1)
        
        # Kolmogorov-Smirnov test
        ks_stat, p_ks = kstest(u, 'uniform')
        
        
        results.append({
            "Batch": _ + 1,
            "χ² Statistic": chi2_stat,
            "χ² p-value": p_chi2,
            "KS Statistic": ks_stat,
            "KS p-value": p_ks,
        })
    
    return pd.DataFrame(results)

# --- Run tests on 20 batches ---
batch_results = test_lcg_batches(105, m, a, c, N, num_batches=20)

# --- Check consistency of results ---
chi2_passes = np.sum(batch_results["χ² p-value"] > 0.05)  # Count passes (p > 0.05)
ks_passes = np.sum(batch_results["KS p-value"] > 0.05)    # Count passes (p > 0.05)

print(f"χ² test passed {chi2_passes}/20 times ({chi2_passes/20*100:.1f}%)")
print(f"KS test passed {ks_passes}/20 times ({ks_passes/20*100:.1f}%)")

# --- Display full results ---
print("\nDetailed batch results:")
print(batch_results)

