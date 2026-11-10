"""
dcos.py - the threshold-free measure of the protocol gap.

Companion code for:
  Wrong Input, Wrong Threshold: Re-evaluating Adversarial Facial Privacy.

Why this file exists
--------------------
Adversarial facial-privacy work reports protection as a success rate, and
compares two evaluation pipelines by the difference of two success rates
(K1). A success rate is bounded at both ends, so that difference saturates:
near 0% there is nothing to lose and near 100% nothing to gain. Over a
perturbation-budget sweep we measured K1 moving by a factor of 8.2 while the
effect underneath it moved by 1.34.

The measure below has no threshold in it at all. Write the margin of a
protected photograph under front end P as

    m_P = tau_P - cos(f_P(x + delta), f_P(x))

and the paired shift between the two front ends as

    Delta_m = (tau_ali - tau_dir) - (cos_ali - cos_dir).

An unprotected photograph has delta = 0, so both cosines are 1 and
Delta_m = tau_ali - tau_dir. That is a property of the two calibrations and
carries nothing about the protection, so subtract it. What remains is

    Delta_cos = cos_dir - cos_ali

in which no threshold appears. `identity_check` below computes both routes
and reports the difference; on 35 stored runs it was exactly 0 in double
precision.

Sign convention (this is the part people get wrong)
---------------------------------------------------
Success means the protected photograph no longer matches its own identity:

    success(sim, tau) = sim < tau,  sim = cos(protected, own clean identity)

so a LOWER cosine is BETTER protection. Therefore

    Delta_cos < 0  ->  protection is weaker under the aligned front end
                       -> the whole-image protocol OVERSTATES protection
    Delta_cos > 0  ->  the protocol understates it

No dependencies beyond numpy.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "success", "dcos", "dcos_via_margin", "identity_check",
    "bootstrap_ci", "asr_gap", "SaturationWarning",
]

SEED = 12345
NBOOT = 5000


# --------------------------------------------------------------------------
# the measure
# --------------------------------------------------------------------------

def success(sim: np.ndarray, tau: float) -> np.ndarray:
    """Protection succeeded: the protected image no longer matches itself."""
    return np.asarray(sim, float) < tau


def dcos(cos_dir: np.ndarray, cos_ali: np.ndarray) -> np.ndarray:
    """Per-image threshold-free protocol gap. Negative = protocol overstates.

    cos_dir and cos_ali must be PAIRED: entry i of each is the same
    protected file scored through the two front ends, against the same
    clean identity.
    """
    a = np.asarray(cos_dir, float)
    b = np.asarray(cos_ali, float)
    if a.shape != b.shape:
        raise ValueError(f"unpaired inputs: {a.shape} vs {b.shape}")
    return a - b


def dcos_via_margin(cos_dir, cos_ali, tau_dir: float, tau_ali: float):
    """The same quantity by the long route, with the thresholds still in it.

    Kept only so that `identity_check` can compare the two. Do not use this
    for reporting: it needs a calibration that `dcos` does not.
    """
    a = np.asarray(cos_dir, float)
    b = np.asarray(cos_ali, float)
    m_dir = tau_dir - a
    m_ali = tau_ali - b
    clean_baseline = tau_ali - tau_dir        # delta = 0 => both cosines 1
    return (m_ali - m_dir) - clean_baseline


def identity_check(cos_dir, cos_ali, tau_dir: float, tau_ali: float,
                   tol: float = 1e-9, verbose: bool = True) -> float:
    """Compute the gap both ways and report the largest disagreement.

    Run this every time you report a number. If a threshold ever leaks into
    a measurement that is supposed to be threshold-free, this catches it
    instead of letting it pass silently.
    """
    short = dcos(cos_dir, cos_ali)
    long_ = dcos_via_margin(cos_dir, cos_ali, tau_dir, tau_ali)
    d = float(np.nanmax(np.abs(short - long_)))
    if verbose:
        status = "ok" if d <= tol else "BROKEN"
        print(f"  identity check: max|short - long| = {d:.3e}  [{status}]")
    if d > tol:
        raise AssertionError(
            f"threshold-free identity violated: max difference {d:.3e} > {tol:.0e}. "
            "Either the two arms are not paired, or a threshold entered the "
            "short route."
        )
    return d


def bootstrap_ci(v, nboot: int = NBOOT, alpha: float = 0.05, seed: int = SEED):
    """Paired bootstrap over images. Returns (mean, lo, hi, n)."""
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    n = v.size
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(nboot, n))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi), int(n)


# --------------------------------------------------------------------------
# the saturating measure, for comparison with the literature only
# --------------------------------------------------------------------------

class SaturationWarning(UserWarning):
    pass


def asr_gap(cos_dir, cos_ali, tau_dir: float, tau_ali: float,
            floor: float = 5.0, ceiling: float = 95.0, verbose: bool = True):
    """K1: the difference of two success rates, in percentage points.

    This is the unit the literature reports. It saturates. Use it only to
    place a result beside published numbers, and always print where the two
    rates sit relative to the floor and the ceiling.
    """
    asr_dir = 100.0 * success(cos_dir, tau_dir).mean()
    asr_ali = 100.0 * success(cos_ali, tau_ali).mean()
    k1 = asr_ali - asr_dir
    near = [n for n, v in (("P_dir", asr_dir), ("P_ali", asr_ali))
            if v < floor or v > ceiling]
    if verbose:
        flag = f"  <- SATURATED: {', '.join(near)}" if near else ""
        print(f"  ASR  P_dir {asr_dir:5.1f}   P_ali {asr_ali:5.1f}   "
              f"K1 {k1:+6.1f} pp{flag}")
    if near:
        import warnings
        warnings.warn(
            f"success rate within {floor}pp of a bound for {', '.join(near)}; "
            "K1 understates the effect here. Report Delta cos instead.",
            SaturationWarning, stacklevel=2,
        )
    return k1, asr_dir, asr_ali


# --------------------------------------------------------------------------
# self-test: synthetic data, no dataset needed
# --------------------------------------------------------------------------

def _self_test() -> None:
    rng = np.random.default_rng(SEED)
    n = 2000
    tau_dir, tau_ali = 0.241045, 0.208      # arbitrary, the point is they cancel
    cos_dir = rng.normal(0.18, 0.09, n)
    cos_ali = cos_dir + rng.normal(0.045, 0.05, n)   # aligned scores higher

    print("self-test on synthetic data")
    identity_check(cos_dir, cos_ali, tau_dir, tau_ali)

    m, lo, hi, k = bootstrap_ci(dcos(cos_dir, cos_ali))
    print(f"  Delta cos = {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]   n = {k}")
    print("  reading:", "protocol OVERSTATES protection" if m < 0
          else "protocol understates protection")

    asr_gap(cos_dir, cos_ali, tau_dir, tau_ali)

    # the identity must hold for ANY pair of thresholds, which is the point
    for td, ta in [(0.0, 0.0), (0.9, -0.3), (0.166788, 0.409131)]:
        d = identity_check(cos_dir, cos_ali, td, ta, verbose=False)
        assert d <= 1e-9
    print("  identity holds for arbitrary thresholds: ok")
    print("self-test passed")


if __name__ == "__main__":
    _self_test()
