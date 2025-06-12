import numpy as np
import pandas as pd
from scipy.stats import kstest, anderson, norm, chi2

def Normal_TexTable(x, results_df=None, bins='auto'):
    """
    Perform KS, AD, χ² tests for normality and append the outcome
    as a single row to `results_df`.

    Parameters
    ----------
    x : 1-D array-like
        Sample to test.
    results_df : pandas.DataFrame or None (default None)
        Existing results table.  If None, a fresh DataFrame is created.
    bins : 'auto' or int
        Number of bins for the χ² test ('auto' → ⌈√n⌉).

    Returns
    -------
    results_df : pandas.DataFrame
        The updated table with one extra row.
    """

    # ---------- prepare sample ----------
    x   = np.asarray(x)
    n   = len(x)
    mu  = x.mean()
    s   = x.std(ddof=1)
    z   = (x - mu) / s                    # standardised data

    # ---------- KS test ----------
    ks_stat, ks_p = kstest(z, 'norm')

    # ---------- AD test ----------
    ad_res        = anderson(z, 'norm')
    ad_stat       = ad_res.statistic
    ad_crit_5pct  = ad_res.critical_values[2]   # 3rd entry = 5 % level

    # ---------- χ² test ----------
    k = int(np.ceil(np.sqrt(n))) if bins == 'auto' else int(bins)
    hist, edges  = np.histogram(z, bins=k)
    expected     = np.diff(norm.cdf(edges)) * n

    # merge low-expectation bins
    while any(expected < 5) and len(expected) > 1:
        expected[-2] += expected[-1]
        hist[-2]     += hist[-1]
        expected, hist = expected[:-1], hist[:-1]
        k -= 1

    chi2_stat = ((hist - expected)**2 / expected).sum()
    dof       = k - 1 - 2                    # k bins, minus 1, minus 2 fitted
    chi2_p    = 1 - chi2.cdf(chi2_stat, dof)

    # ---------- build one tidy row ----------
    row = {
        "KS_stat"      : ks_stat,
        "KS_p"         : ks_p,
        "AD_stat"      : ad_stat,
        "AD_crit_5pct" : ad_crit_5pct,
        "Chi2_stat"    : chi2_stat,
        "Chi2_p"       : chi2_p,
        "Chi2_dof"     : dof,
        "Chi2_bins"    : k,
        "N"            : n,          # keep sample size for reference
        "mean"         : mu,
        "sd"           : s
    }

    # ---------- append and return ----------
    if results_df is None:
        results_df = pd.DataFrame([row])
    else:
        results_df = pd.concat([results_df, pd.DataFrame([row])],
                               ignore_index=True)

    return results_df.to_latex(index=False,
                         float_format="%.3f".__mod__,
                         caption="Simulated Normal moments",
                         label="tab:pareto")




def Pareto_TexTable(x,
                            beta      = 1.0,        # scale (cut–off)
                            k         = None,        # shape; if None → MLE
                            results_df= None,
                            bins      = 'auto'):
    """
    KS, AD, χ² tests for Pareto(β,k) on [β,∞), append one result row.

    Parameters
    ----------
    x : 1-D array-like
        Sample to test (must all be >= beta).
    beta : float, default 1.0
        Scale / lower cut-off parameter β.
    k : float or None, default None
        Shape parameter.  If None, use MLE  k̂ = n / Σ log(x_i/β).
    results_df : pandas.DataFrame or None
        Existing results table; if None a new one is started.
    bins : 'auto' or int
        Number of χ² bins (equal-probability).  'auto' → ⌈√n⌉.

    Returns
    -------
    results_df : pandas.DataFrame
        Updated table, ready for `.to_latex(...)`.
    """

    # ----- sanity & numpy array -----
    x = np.asarray(x, dtype=float)
    if np.any(x < beta):
        raise ValueError("All observations must be ≥ beta.")

    n = len(x)
    if n < 8:
        raise ValueError("Need at least 8 obs for these tests.")

    # ----- estimate shape if needed (MLE) -----
    if k is None:
        k_hat = n / np.sum(np.log(x / beta))
    else:
        k_hat = float(k)

    # ----- KS test --------------------------------------------------------
    # CDF: F(x) = 1 - (β/x)^k   for x ≥ β ; 0 otherwise.
    def pareto_cdf(t):                     # vectorised
        t = np.asarray(t, dtype=float)
        cdf = np.zeros_like(t)
        mask = t >= beta
        cdf[mask] = 1.0 - (beta / t[mask]) ** k_hat
        return cdf

    ks_stat, ks_p = kstest(x, pareto_cdf)

    # ----- Anderson–Darling ----------------------------------------------
    # SciPy has 'pareto' built in (uses same paramisation);
    # it refits k internally, so AD is unconditional.
    ad_res = anderson(x, 'pareto')
    ad_stat       = ad_res.statistic
    ad_crit_5pct  = ad_res.critical_values[2]   # 5 % level

    # ----- χ² test --------------------------------------------------------
    k_bins = int(np.ceil(np.sqrt(n))) if bins == 'auto' else int(bins)

    # build equal-probability bin edges via inverse-CDF
    # F^{-1}(p) = β / (1-p)^{1/k}
    p = np.linspace(0, 1, k_bins + 1)
    p[-1] = 1 - 1e-12                       # avoid ∞
    edges = beta / (1 - p) ** (1.0 / k_hat)

    hist, _ = np.histogram(x, bins=edges)
    expected = np.full_like(hist, n / k_bins)

    # merge if any expected < 5
    while (expected < 5).any() and len(expected) > 1:
        expected[-2] += expected[-1]
        hist[-2]     += hist[-1]
        expected, hist = expected[:-1], hist[:-1]
        k_bins -= 1

    chi2_stat = ((hist - expected) ** 2 / expected).sum()
    dof       = k_bins - 1 - 1            # estimate 1 param (k̂)
    chi2_p    = 1 - chi2.cdf(chi2_stat, dof)

    # ----- tidy row -------------------------------------------------------
    row = {
        "KS_stat"      : ks_stat,
        "KS_p"         : ks_p,
        "AD_stat"      : ad_stat,
        "AD_crit_5pct" : ad_crit_5pct,
        "Chi2_stat"    : chi2_stat,
        "Chi2_p"       : chi2_p,
        "Chi2_dof"     : dof,
        "Chi2_bins"    : k_bins,
        "n"            : n,
        "beta"         : beta,
        "k_hat"        : k_hat
    }

    # ----- append & return -----------------------------------------------
    new_row = pd.DataFrame([row])
    results_df = (pd.concat([results_df, new_row], ignore_index=True)
                  if results_df is not None else new_row)
    return results_df
