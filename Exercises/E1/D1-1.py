import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
import pandas as pd
import math as math
from scipy import stats
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
m = int(1000)
a = int(3)
c = int(7)
N = int(10000)

X = LCG(X0, m, a, c, N)

plt.hist(X,bins=10, rwidth=0.6)
#plt.savefig('Figures/Hist-1a.png')
plt.show()

#1b)
fig, ax = plt.subplots()
ax.scatter(X[:-1], X[1:], s= 3)        # (u_n , u_{n+1})
ax.set_title("Successive pairs (uₙ , uₙ₊₁)")
ax.set_xlabel("uₙ")
ax.set_ylabel("uₙ₊₁")
#plt.savefig('Figures/Pairscatter-1b')
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
N = len(s)                           # sample size
lags = range(0, 100)
r = [s.autocorr(lag=lag) for lag in lags]

# 95 % confidence limits for white-noise: ±1.96/√N
conf = 1.96 / np.sqrt(N)

plt.figure()
plt.bar(lags, r, width=0.2)
plt.axhline( conf, color='red', linestyle='--', linewidth=1)
plt.axhline(-conf, color='red', linestyle='--', linewidth=1)
plt.axhline(0, color='black', linewidth=1)
plt.title("Sample autocorrelation")
plt.xlabel("Lag (h)")
plt.ylabel("r(h)")
plt.show()


s  = pd.Series(X)          # X = random numbers from your LCG
N  = len(s)
h_to_test = range(1, 100) 
# --- correlation test ---
results = []
for h in h_to_test:
    r = s.autocorr(lag=h)
    t = r * np.sqrt((N - h - 2) / (1 - r**2))
    p = 2 * stats.t.sf(abs(t), df=N - h - 2)
    results.append((h, r, p))

print("lag  r(h)        p-value")
for h, r, p in results:
    print(f"{h:3d}  {r:+.5f}   {p:.4f}")


import math
from scipy.stats import kstest, chi2

def ks_test_uniform(u, tol=1e-10, max_iter=100):
    """
    Two-sided Kolmogorov–Smirnov test against U(0,1) without scipy.kstest.
    
    Parameters
    ----------
    u : 1-D array-like, the sample on (0,1).
    tol : float, stop the p-value series when the next term < tol.
    max_iter : int, hard cap on number of terms (safety).
    
    Returns
    -------
    D  : float, the KS statistic.
    p  : float, asymptotic two-sided p-value.
    """
    u = np.sort(np.asarray(u))
    n = u.size
    i = np.arange(1, n + 1)

    d_plus  = np.max(i / n - u)
    d_minus = np.max(u - (i - 1) / n)
    D = max(d_plus, d_minus)

    # Kolmogorov asymptotic p-value
    #    p = 2 Σ_{k=1}^∞ (−1)^{k−1} exp(−2 k² n D²)
    nd2 = n * D * D * 2
    s = 0.0
    k = 1
    while k <= max_iter:
        term = (-1) ** (k - 1) * np.exp(-nd2 * k * k)
        s += term
        if abs(term) < tol:
            break
        k += 1
    p = 2.0 * s
    p = np.clip(p, 0.0, 1.0)        # numerical safety
    return D, p
import numpy as np
import math                             # for erfc -> normal tail

def runs_test_uniform(u, threshold=0.5, correction=False):
    """
    Wald–Wolfowitz runs test for randomness against U(0,1).

    Parameters
    ----------
    u : 1-D array-like of floats on (0,1)
    threshold : float, cut-off to turn the sample into 0/1 symbols.
                For a true U(0,1) null the theoretical median is 0.5,
                but you can pass np.median(u) if you prefer the sample median.
    correction : bool, apply the usual ±0.5 continuity correction.

    Returns
    -------
    Z : float, standard-normal test statistic (large-sample approx.)
    p : float, two-sided p-value based on Z.
    """
    # turn the sample into a boolean sequence above/below the threshold
    s = np.asarray(u) > threshold
    n = s.size
    if n < 2:
        raise ValueError("Need at least two observations for a runs test.")

    # counts of each symbol
    n1 = s.sum()          # True (above threshold)
    n2 = n - n1           # False (below threshold)

    if n1 == 0 or n2 == 0:    # all the same symbol -> 0 runs variance
        return 0.0, 1.0       # degenerate, accept H0

    # number of runs: 1 + transitions between successive symbols
    R = 1 + np.count_nonzero(s[1:] != s[:-1])

    # mean and variance under H0
    mu_R = 1 + 2 * n1 * n2 / n
    var_R = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)
             ) / (n**2 * (n - 1))

    # continuity correction, if requested
    corr = 0.0
    if correction:
        corr = 0.5 if R > mu_R else -0.5

    Z = (R - mu_R - corr) / math.sqrt(var_R)

    # two-sided p-value via the normal tail: p = 2 * Φ̅(|Z|)
    p = math.erfc(abs(Z) / math.sqrt(2))   # erfc(x) = 2*Φ̅(x*√2)

    return Z, p

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
        ks_stat, p_ks = ks_test_uniform(u)
        
        
              # Runs test (home-built)
        Z_runs, p_runs = runs_test_uniform(u, threshold=0.5, correction=True)

        results.append({
            "Batch":        _ + 1,
            "χ² Statistic": chi2_stat,
            "χ² p-value":   p_chi2,
            "KS Statistic": ks_stat,
            "KS p-value":   p_ks,
            "Runs Z":       Z_runs,
            "Runs p-value": p_runs,
        })
    
    return pd.DataFrame(results)

# --- Run tests on 20 batches ---
batch_results = test_lcg_batches(105, m, a, c, N, num_batches=20)

# --- Check consistency of results ---
chi2_passes = np.sum(batch_results["χ² p-value"] > 0.05)  # Count passes (p > 0.05)
ks_passes = np.sum(batch_results["KS p-value"] > 0.05)    # Count passes (p > 0.05)
runs_passes = np.sum(batch_results["Runs p-value"] > 0.05)
print(f"χ² test passed {chi2_passes}/20 times ({chi2_passes/20*100:.1f}%)")
print(f"KS test passed {ks_passes}/20 times ({ks_passes/20*100:.1f}%)")
print(f"KS test passed {runs_passes}/20 times ({runs_passes/20*100:.1f}%)")


# --- Display full results ---
print("\nDetailed batch results:")
print(batch_results)

