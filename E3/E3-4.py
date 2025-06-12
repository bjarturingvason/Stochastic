import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest

from Latex_helper import Pareto_TexTable

def pareto_via_composition(beta, k, size, rng=None):
    """
    Simulate `size` draws from Pareto(beta, k) on [beta, ∞) 
    by mixing an exponential with a Gamma(k, 1/beta) rate.
    """
    if rng is None:
        rng = np.random.default_rng()
    # 1) draw the mixing rates
    lam = rng.gamma(shape=k, scale=1.0/beta, size=size)
    # 2) draw exponentials with those rates
    y   = rng.exponential(scale=1.0/lam)   # scale = 1/rate
    # 3) shift by beta
    return beta + y

# Generate sample
beta = 1.0
k = 2.5
N = 10_000
rng = np.random.default_rng(42)
sample = pareto_via_composition(beta, k, size=N, rng=rng)

# Plot histogram and theoretical PDF
plt.figure()
plt.hist(sample, density=True, bins = 200, edgecolor='black')
x = np.linspace(beta, sample.max(), 400)
pdf = k * beta**k / x**(k+1)
plt.plot(x, pdf)
plt.title(f"Pareto(β={beta}, k={k}) – Composition Method")
plt.xlabel("x")
plt.ylabel("Density")
plt.savefig('E3/Figures/Pareto-composition')
plt.xlim(0,10)
plt.show()

table = None
table = Pareto_TexTable(sample, beta,k,results_df=table)
print(table.to_latex(index=False,
                     float_format="%.3f".__mod__,
                     caption="Pareto normality checks",
                     label="tab:pareto"))