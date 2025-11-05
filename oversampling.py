#!/usr/bin/env python3
"""
Oversampling calculator for synthetic gene libraries with non-uniform representation.

Inputs:
  - N (int): diversity (number of intended variants)
  - fidelity (float 0..1): fraction of perfect molecules
  - gini (float 0..1): Gini coefficient describing unevenness of representation
  - coverage/probability targets or sample size t

Model:
  - Build a lognormal-shaped probability distribution over variants whose Gini ≈ target.
  - Use Poissonization for sampling: each variant i has count ~ Poisson(t * f * p_i).
  - Coverage indicator for variant i is Bernoulli(q_i) with q_i = 1 - exp(-t f p_i).
  - Approximate coverage fraction C via CLT across variants to map mean/variance to a
    high-probability guarantee at level 'prob' (e.g., 0.95).

Supports:
  - t_for_coverage(): minimum t to achieve coverage >= cov_target with probability >= prob
  - coverage_for_t(): coverage guaranteed with probability >= prob at sample size t

Notes:
  - Avoids enumerating N probabilities by using K bins (default K=400).
  - The average p_i is 1/N by construction, so sum_i p_i = 1.

Reference: Guido et al., "Determination of a Screening Metric for High Diversity DNA Libraries"
"""

import math
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np

# --------------------------- Utility: Gini for weighted samples ---------------------------

def weighted_gini(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Gini coefficient for non-negative values with weights.
    Returns 0 for perfectly uniform, approaches 1 for extreme inequality.
    """
    assert np.all(values >= 0), "Values must be non-negative"
    w = weights / np.sum(weights)
    # Sort by value
    order = np.argsort(values)
    v = values[order]
    w = w[order]
    # Cumulative sums
    cw = np.cumsum(w)
    cv = np.cumsum(v * w)
    # Relative mean absolute difference via Lorenz curve:
    # G = 1 - 2 * ∫_0^1 L(u) du, approximated from discrete cum sums
    # Trapezoid integral over Lorenz curve points (cw, cv / cv[-1])
    L = cv / cv[-1] if cv[-1] > 0 else np.zeros_like(cv)
    # integral via trapezoids
    integral = np.sum((L[1:] + L[:-1]) * (cw[1:] - cw[:-1]) / 2.0)
    G = 1.0 - 2.0 * integral
    return float(G)

# --------------------- Build a representation with desired Gini ---------------------------

def lognormal_quantiles(K: int, sigma: float, seed: int = 0) -> np.ndarray:
    """
    Return K deterministic quantile points of a lognormal with sigma (mu=adjusted below).
    We don't random-sample to keep it stable; use mid-quantiles.
    """
    # base normal quantiles
    ps = (np.arange(K) + 0.5) / K
    z = np.sqrt(2) * erfinv(2*ps - 1)
    # pick mu so the mean of LN is 1 before normalization; we can simply take mu = -0.5*sigma^2
    mu = -0.5 * sigma**2
    ln_vals = np.exp(mu + sigma * z)
    return ln_vals

# fast, robust erfinv without SciPy
def erfinv(x):
    # Winitzki approximation + refinement (sufficient for quantiles here)
    a = 0.147  # magic constant
    sgn = np.sign(x)
    ln = np.log(1 - x**2)
    term = 2/(np.pi*a) + ln/2
    inner = term**2 - ln/a
    y = sgn * np.sqrt(np.sqrt(inner) - term)
    # one Newton step to refine: f(y) = erf(y) - x
    # erf'(y) = 2/sqrt(pi) * exp(-y^2)
    erf_y = erf(y)
    y = y - (erf_y - x) / (2/np.sqrt(np.pi) * np.exp(-y*y))
    return y

def erf(x):
    # Abramowitz-Stegun approximation
    # good enough for our purposes
    t = 1.0 / (1.0 + 0.5 * np.abs(x))
    tau = t * np.exp(-x*x - 1.26551223 + t * (1.00002368 +
          t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
          t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
          t * (-0.82215223 + t * 0.17087277)))))))))
    return np.where(x >= 0, 1 - tau, tau - 1)

def construct_probs_with_gini(N: int, target_gini: float, K: int = 400, tol: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an aggregated representation of probabilities as K bins with equal variant counts.
    Returns:
      p_bins: length-K representative probabilities p_k for variants in bin k
      w_bins: length-K weights w_k (fraction of variants in each bin, sum to 1)

    We use a lognormal shape and tune sigma by bisection until the weighted Gini matches target_gini.
    """
    # Edge cases
    if target_gini <= 1e-6:
        p_k = np.full(K, 1.0/N)
        w_k = np.full(K, 1.0/K)
        return p_k, w_k

    # Bisection over sigma
    low, high = 1e-6, 5.0
    w_k = np.full(K, 1.0/K)

    def gini_for_sigma(sig: float) -> float:
        raw = lognormal_quantiles(K, sig)      # positive shape values
        # normalize *across variants* so that average p_i = 1/N
        # mean(raw) represents average "weight" per variant; set p = raw / (N * mean(raw))
        p = raw / (N * np.average(raw, weights=w_k))
        return weighted_gini(p, w_k)

    g_low = gini_for_sigma(low)
    g_high = gini_for_sigma(high)
    # If target outside achievable range, clamp
    if target_gini <= g_low + 1e-6:
        p_k = np.full(K, 1.0/N)
        return p_k, w_k
    if target_gini >= g_high - 1e-6:
        raw = lognormal_quantiles(K, high)
        p_k = raw / (N * np.average(raw, weights=w_k))
        return p_k, w_k

    # Bisection
    for _ in range(60):
        mid = 0.5 * (low + high)
        g_mid = gini_for_sigma(mid)
        if g_mid < target_gini:
            low = mid
        else:
            high = mid
        if abs(g_mid - target_gini) < tol:
            break

    # Build final bins
    sigma = 0.5 * (low + high)
    raw = lognormal_quantiles(K, sigma)
    p_k = raw / (N * np.average(raw, weights=w_k))
    return p_k, w_k

# ------------------------ Coverage stats and solvers ------------------------

@dataclass
class LibrarySpec:
    N: int
    fidelity: float   # 0..1
    gini: float       # 0..1
    K: int = 400      # number of aggregation bins

@dataclass
class CoverageStats:
    mean: float
    var: float

def coverage_stats(t: float, spec: LibrarySpec, p_bins: np.ndarray, w_bins: np.ndarray) -> CoverageStats:
    """Compute mean and variance of coverage fraction at sample size t."""
    f = spec.fidelity
    # Poissonization => P(variant seen at least once) = 1 - exp(-lambda), lambda = t * f * p_i
    lam = f * t * p_bins
    # numerical safety for large/small lambda
    q = 1.0 - np.exp(-np.clip(lam, 0, 700))  # exp(-700) ~ 5e-305
    mean = float(np.sum(w_bins * q))
    # Var of average of N Bernoullis with probs q_i: sum q_i(1-q_i)/N^2
    # Grouped by bins: (1/N^2) * sum_k (w_k*N) * q_k(1-q_k) = (1/N) * sum_k w_k q_k(1-q_k)
    var = float((1.0/spec.N) * np.sum(w_bins * q * (1.0 - q)))
    return CoverageStats(mean=mean, var=var)

def z_from_prob(prob: float) -> float:
    """Two-sided? We need one-sided quantile since we want P(C >= c)."""
    # inverse CDF of standard normal; use erfinv
    # For one-sided tail: prob = Phi(z) => z = Phi^{-1}(prob)
    return math.sqrt(2.0) * float(erfinv(2.0*prob - 1.0))

def t_for_coverage(cov_target: float, prob: float, spec: LibrarySpec, p_bins: np.ndarray, w_bins: np.ndarray,
                   t_lo: float = 0.0, t_hi: Optional[float] = None) -> Tuple[float, CoverageStats]:
    """
    Smallest t such that P(C >= cov_target) >= prob.
    Uses CLT: require mean - z*sqrt(var) >= cov_target.
    """
    assert 0.0 < cov_target < 1.0, "coverage target must be in (0,1)"
    assert 0.5 < prob < 1.0, "probability should be >0.5 for a guarantee (e.g., 0.9, 0.95, 0.99)"
    z = z_from_prob(prob)

    # Find an upper bound if not provided
    if t_hi is None:
        t_hi = 10.0 * spec.N / max(spec.fidelity, 1e-9)  # generous initial bound
        # escalate until condition met
        for _ in range(60):
            st = coverage_stats(t_hi, spec, p_bins, w_bins)
            if st.mean - z * math.sqrt(max(st.var, 1e-18)) >= cov_target:
                break
            t_hi *= 2.0

    # Bisection
    for _ in range(80):
        t_mid = 0.5 * (t_lo + t_hi)
        st = coverage_stats(t_mid, spec, p_bins, w_bins)
        lhs = st.mean - z * math.sqrt(max(st.var, 1e-18))
        if lhs >= cov_target:
            t_hi = t_mid
        else:
            t_lo = t_mid
        if t_hi - t_lo <= 1e-6 * max(1.0, t_hi):
            break
    final_stats = coverage_stats(t_hi, spec, p_bins, w_bins)
    return t_hi, final_stats

def coverage_for_t(t: float, prob: float, spec: LibrarySpec, p_bins: np.ndarray, w_bins: np.ndarray) -> Tuple[float, CoverageStats]:
    """
    Coverage guaranteed with probability >= prob at sample size t.
    Returns (guaranteed_coverage, stats), where guaranteed_coverage = mean - z * sqrt(var).
    """
    z = z_from_prob(prob)
    st = coverage_stats(t, spec, p_bins, w_bins)
    guaranteed = st.mean - z * math.sqrt(max(st.var, 1e-18))
    return guaranteed, st

# ------------------------------ CLI demo -----------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Oversampling calculator for synthetic libraries with fidelity and Gini.")
    ap.add_argument("--N", type=int, required=True, help="Diversity (number of intended variants)")
    ap.add_argument("--fidelity", type=float, required=True, help="Fraction perfect (0..1). E.g., 0.8 for 80%% perfect")
    ap.add_argument("--gini", type=float, required=True, help="Gini coefficient of representation (0..1)")
    ap.add_argument("--K", type=int, default=400, help="Aggregation bins (default 400)")

    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("t_for_coverage", help="Solve for samples t given coverage & probability")
    s1.add_argument("--coverage", type=float, required=True, help="Target coverage fraction in (0,1), e.g. 0.95")
    s1.add_argument("--prob", type=float, default=0.95, help="Guarantee probability, e.g. 0.95")

    s2 = sub.add_parser("coverage_for_t", help="Compute guaranteed coverage at sample size t")
    s2.add_argument("--t", type=float, required=True, help="Sample size (number of molecules drawn)")
    s2.add_argument("--prob", type=float, default=0.95, help="Guarantee probability, e.g. 0.95")

    args = ap.parse_args()

    spec = LibrarySpec(N=args.N, fidelity=args.fidelity, gini=args.gini, K=args.K)
    p_bins, w_bins = construct_probs_with_gini(spec.N, spec.gini, K=spec.K)

    if args.cmd == "t_for_coverage":
        t_req, st = t_for_coverage(args.coverage, args.prob, spec, p_bins, w_bins)
        eff = spec.fidelity * t_req
        print(f"[t_for_coverage]")
        print(f"N={spec.N:,}  fidelity={spec.fidelity:.4f}  gini={spec.gini:.4f}  K={spec.K}")
        print(f"Target: coverage >= {args.coverage:.4f} with probability >= {args.prob:.3f}")
        print(f"Required samples t  = {t_req:,.3f}")
        print(f"Effective perfect draws f*t = {eff:,.3f}")
        print(f"At t, E[coverage]={st.mean:.6f}  Var={st.var:.6e}")

    elif args.cmd == "coverage_for_t":
        cov_guaranteed, st = coverage_for_t(args.t, args.prob, spec, p_bins, w_bins)
        print(f"[coverage_for_t]")
        print(f"N={spec.N:,}  fidelity={spec.fidelity:.4f}  gini={spec.gini:.4f}  K={spec.K}")
        print(f"At t={args.t:,.3f}, guaranteed coverage (p>={args.prob:.3f}) = {cov_guaranteed:.6f}")
        print(f"E[coverage]={st.mean:.6f}  Var={st.var:.6e}")

if __name__ == "__main__":
    main()
