"""Statistics utilities (SPEC §6): sequence-level BCa bootstrap, Wilcoxon + Holm,
rank-biserial effect size. analysis/ reads only results/ artifacts (SPEC §7.5);
this module is pure computation with no I/O.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def bca_ci(units: np.ndarray, stat_fn, n_boot: int = 10000, alpha: float = 0.05,
           seed: int = 0) -> dict:
    """BCa bootstrap CI for stat_fn over exchangeable units.

    units: array of unit indices (0..S-1) is implicit; stat_fn receives an index
    array into the units and returns a scalar. Resampling is at the unit
    (=sequence) level per SPEC §6.
    """
    rng = np.random.default_rng(seed)
    S = len(units)
    theta_hat = stat_fn(np.arange(S))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = stat_fn(rng.integers(0, S, size=S))
    # bias correction
    prop = np.mean(boots < theta_hat)
    prop = min(max(prop, 1.0 / (n_boot + 1)), 1 - 1.0 / (n_boot + 1))
    z0 = sps.norm.ppf(prop)
    # acceleration via jackknife
    jack = np.array([stat_fn(np.delete(np.arange(S), i)) for i in range(S)])
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6.0 * (((jm - jack) ** 2).sum() ** 1.5)
    a = num / den if den > 0 else 0.0
    z = sps.norm.ppf([alpha / 2, 1 - alpha / 2])
    adj = sps.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo, hi = np.quantile(boots, adj)
    return {"stat": float(theta_hat), "ci_lo": float(lo), "ci_hi": float(hi),
            "alpha": alpha, "n_boot": n_boot, "method": "bca"}


def wilcoxon_rank_biserial(x: np.ndarray, y: np.ndarray,
                           alternative: str = "two-sided") -> dict:
    """Paired Wilcoxon signed-rank + rank-biserial effect size r = (W+ - W-)/(W+ + W-)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    nz = d[d != 0]
    if len(nz) == 0:
        return {"p": 1.0, "W": 0.0, "r": 0.0, "n_nonzero": 0}
    res = sps.wilcoxon(x, y, alternative=alternative, zero_method="wilcox")
    ranks = sps.rankdata(np.abs(nz))
    w_pos = ranks[nz > 0].sum()
    w_neg = ranks[nz < 0].sum()
    r = (w_pos - w_neg) / (w_pos + w_neg)
    return {"p": float(res.pvalue), "W": float(res.statistic), "r": float(r),
            "n_nonzero": int(len(nz))}


def holm_correct(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (SPEC §6)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
