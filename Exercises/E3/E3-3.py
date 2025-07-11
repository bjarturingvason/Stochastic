import numpy as np
import pandas as pd
from scipy.stats import t, chi2

true_mu = 0.0
true_sigma2 = 1.0
n = 10
R = 100
rng = np.random.default_rng()

records = []
for _ in range(R):
    x = rng.normal(loc=true_mu, scale=np.sqrt(true_sigma2), size=n)
    x_bar = x.mean()
    s2 = x.var(ddof=1)
    s = np.sqrt(s2)
    
    tcrit = t.ppf(0.975, df=n-1)
    mean_low  = x_bar - tcrit * s / np.sqrt(n)
    mean_high = x_bar + tcrit * s / np.sqrt(n)
    mean_contains = mean_low <= true_mu <= mean_high
    
    chi2_low  = chi2.ppf(0.975, df=n-1)
    chi2_high = chi2.ppf(0.025, df=n-1)
    var_low  = (n-1) * s2 / chi2_low
    var_high = (n-1) * s2 / chi2_high
    var_contains = var_low <= true_sigma2 <= var_high
    
    records.append({
        "mean_low": mean_low,
        "mean_high": mean_high,
        "mean_contains": mean_contains,
        "var_low": var_low,
        "var_high": var_high,
        "var_contains": var_contains
    })

df = pd.DataFrame(records)
summary = pd.DataFrame({
    "Parameter": ["Mean", "Variance"],
    "Intervals covering true value": [df["mean_contains"].sum(), df["var_contains"].sum()],
    "Expected (if exactly 95%)": [95, 95]
})

print(summary)

# --- existing code above here ---

import matplotlib.pyplot as plt

# 1) CI for the mean ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 8))

for idx, row in df.iterrows():
    colour = "tab:green" if row["mean_contains"] else "tab:red"
    ax.hlines(
        y=idx + 1,               # vertical position for this replication
        xmin=row["mean_low"],
        xmax=row["mean_high"],
        color=colour,
        linewidth=2,
    )

ax.axvline(true_mu, color="black", linestyle="--", label=f"true μ = {true_mu}")
ax.set_xlabel("μ")
ax.set_ylabel("Replication")
ax.set_title("95 % confidence intervals for the mean")
ax.legend()
plt.tight_layout()
plt.savefig('E3-3-1.png', dpi = 300)
plt.show()

# 2) CI for the variance ------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 8))

for idx, row in df.iterrows():
    colour = "tab:green" if row["var_contains"] else "tab:red"
    ax.hlines(
        y=idx + 1,
        xmin=row["var_low"],
        xmax=row["var_high"],
        color=colour,
        linewidth=2,
    )

ax.axvline(true_sigma2, color="black", linestyle="--", label=f"true σ² = {true_sigma2}")
ax.set_xlabel("σ²")
ax.set_ylabel("Replication")
ax.set_title("95 % confidence intervals for the variance")
ax.legend()
plt.tight_layout()
plt.savefig('E3-3.png', dpi = 300)
plt.show()
