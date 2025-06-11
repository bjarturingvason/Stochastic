from scipy.stats import expon, norm, pareto, kstest, anderson
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
N = 100000

#a)

exp_sample = expon.rvs(size=N)

fig, ax = plt.subplots()
ax.hist(exp_sample, bins=40, density=True, alpha=0.6, edgecolor='black',
        label='Simulated histogram')

x = np.linspace(0, exp_sample.max(), 400)
pdf = np.exp(-x)                             # f(x)=e^{-x} for λ=1

ax.plot(x, pdf, linewidth=2, label='Exp(1) pdf')
ax.set_title("Exponential(λ=1)")
ax.set_xlabel("x"); ax.set_ylabel("density")
ax.legend()
#plt.savefig('Figures/exp-3-1.png')
plt.show()

#tests
ks_stat, ks_p = kstest(exp_sample, 'expon')
ad_stat, _, _ = anderson(exp_sample, 'expon')

#b)
norm_sample =  norm.rvs(size=N)

fig, ax = plt.subplots()
ax.hist(norm_sample, bins=40, density=True, alpha=0.6, edgecolor='black')
x = np.linspace(-4, 4, 400)
pdf = (1/np.sqrt(2*np.pi)) * np.exp(-x**2 / 2)
ax.plot(x, pdf, linewidth=2)
ax.set_title("Normal(0,1) – Box-Muller")
ax.set_xlabel("x"); ax.set_ylabel("density")
#plt.savefig('Figures/Norm-3-1.png')
plt.show()

ks_stat, ks_p = kstest(norm_sample, 'norm')
ad_stat, crit, sig = anderson(norm_sample, 'norm')

#c)
k_vals = [2.05, 2.5, 3, 4]
N = 10_000
beta = 1
rows = []
for k in k_vals:
    u = np.random.default_rng().random(N)
    pareto_sample = (1 - u)**(-1/k) - 1       # inverse-cdf

    # ----- plot --------------------------------------------------
    fig, ax = plt.subplots()
    ax.hist(pareto_sample, bins=80, range=(0, 10), density=True,
            alpha=0.6, edgecolor='black')
    x = np.linspace(0, 10, 400)
    pdf = k * (1+x)**(-k-1)
    ax.plot(x, pdf, linewidth=2)
    ax.set_title(f"Pareto β=1, k={k}")
    ax.set_xlabel("x"); ax.set_ylabel("density")
    filename = f"Figures/Pareto-3-{k:.2f}.png"  # e.g. "pareto_hist_k_2.05.png"
    #fig.savefig(filename)
    plt.show()

    # ----- KS test ----------------------------------------------
    ks_stat, ks_p = kstest(pareto_sample, 'pareto', args=(k,))

    #2)
    sample_mean = pareto_sample.mean()
    sample_var = pareto_sample.var(ddof=1)

    theor_mean = beta * k / (k - 1) if k > 1 else math.inf
    theor_var = (beta**2) * k / ((k - 1) ** 2 * (k - 2)) if k > 2 else math.inf

    rows.append({
        "k": k,
        "Sample mean": sample_mean,
        "Theory mean": theor_mean,
        "Mean % error": 100 * (sample_mean - theor_mean) / theor_mean,
        "Sample var": sample_var,
        "Theory var": theor_var,
        "Var % error": 100 * (sample_var - theor_var) / theor_var
    })
df = pd.DataFrame(rows).round(4)
print(df)


