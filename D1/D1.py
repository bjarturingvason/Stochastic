import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
import pandas as pd
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




# -- Use the Series.autocorr *built‑in* method ------------------------------------
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
