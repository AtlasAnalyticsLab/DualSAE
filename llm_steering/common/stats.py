"""Per-cell statistics for the confirmatory families (D-01, D-03; PROTOCOL.md 1.2-1.5).

What lives here, and nothing else:
  * ``welch``        unpaired Welch t over seeds: delta, 95% CI (Welch-Satterthwaite df), raw
                     two-sided p, df, MDE80, n per arm.
  * ``bh``           Benjamini-Hochberg q-values within ONE family (BH never spans two families).
  * ``verdict``      the PROTOCOL 1.4 verdict words, applied in the pre-registered order.
  * ``pooled_sd`` / ``flag_outliers``   the GLOBAL-scale outlier flag (PROTOCOL 1.5): sd_ref = pooled within-arm seed sd over all cells of a
                     metric (and K or layer); a seed is flagged when |x_i - mean(other seeds)| >
                     4 * sd_ref.
  * small table helpers (sample sd with ddof=1, mean +- sd strings, win-count lines).

MDE80 (noncentral t, the value D-03 / PROTOCOL 1.2 pre-register as "about 3.07 x sd at
3+3").  The minimum detectable effect at 80% power of the two-sided alpha=0.05 test is
MDE80 = d * se, where d solves

    P(|T'| > t_{0.975, df}) = nct.sf(t_c, df, d) + nct.cdf(-t_c, df, d) = 0.80,

T' ~ noncentral t(df, d), se = sqrt(s_a^2/n_a + s_b^2/n_b) and df = the Welch-Satterthwaite df
of the observed data.  At 3+3 with equal sds (df=4) this gives d*sqrt(2/3) = 3.071 sd.  The
noncentral t is exact for equal n and equal variances (the Welch statistic is then the pooled
t) and the standard approximation otherwise.
The noncentral-t-free shortcut (t_{0.975,df} + t_{0.80,df}) * se is kept ONLY as a fallback if
the root-finder fails.  It UNDER-states the MDE (3.035 sd at df=4, -1.2%; -2.2% at df=3; -5.1%
at df=2), i.e. it overstates power (true power at the shortcut value is 0.79 at df=4), so it is
never the primary value.

Self-test:  python -m common.stats --selftest
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as _st

# PROTOCOL 1.2: a test is significant iff q < 0.10.  Module constants so that no caller can
# drift from the pre-registered values by passing its own.
ALPHA = 0.05            # two-sided test level (and the 95% CI)
Q_THRESHOLD = 0.10      # BH significance threshold
OUTLIER_K = 4.0         # outlier flag multiplier on sd_ref (PROTOCOL 1.5)
POWER = 0.80            # for MDE80

# PROTOCOL 1.3 SESOIs, keyed by metric name.  The verdict() function takes the SESOI explicitly,
# but callers read it from here so the pre-registered value is the only one in the code base.
# (The shared statistics module also served other downstream applications; their SESOI entries
# are kept so the selftest covers the same cases.)
SESOI: Dict[str, float] = {
    "D": 0.03,                  # topic diversity (other application)
    "C_R": 2.0,                 # coherence rating, 0-100 (other application)
    "C_I": 3.0,                 # intruder accuracy, percentage points (other application)
    "macro_sig": 0.3,           # macro #Sig (other application)
    "harmonic_mean": 0.10,      # AxBench harmonic mean, 0-2
    "p_concept": 0.05,          # bge classifier P(concept), 0-1
    "auroc": 0.01,              # detection AUROC (exploratory: no verdict words)
}

# The words PROTOCOL 1.4 forbids.  verdict() can never emit them; the check is an assertion so a
# future edit that tries to cannot pass the selftest.
FORBIDDEN_WORDS = ("no effect", "parity")


# --------------------------------------------------------------------------- descriptives
def _as_1d(x: Iterable[float], name: str) -> np.ndarray:
    a = np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=np.float64).ravel()
    if not np.all(np.isfinite(a)):
        # A NaN seed value is a missing measurement, not a number: silently dropping it would
        # change n without anyone noticing, so the caller must drop it explicitly (and report it).
        raise ValueError(f"{name} contains non-finite values {a.tolist()}; drop missing seeds "
                         f"explicitly before testing")
    return a


def sample_sd(x: Iterable[float]) -> float:
    """Sample standard deviation, ddof=1 (D-01).  NaN for fewer than 2 values.

    Identical values give EXACTLY 0.0.  Why: np.std computes the mean first, and the mean of
    three copies of 0.9 is 0.9000000000000001, so np.std([0.9]*3) is ~1.3e-16, not 0.  Welch
    would then divide a real delta by ~1e-16 and report p ~ 0 for two constant arms -- a
    "significant" result produced by rounding.  A constant arm must be recognised as constant
    (welch's zero-variance rule)."""
    a = _as_1d(x, "x")
    if a.size < 2:
        return float("nan")
    if np.all(a == a[0]):
        return 0.0
    return float(np.std(a, ddof=1))


def mean_sd(x: Iterable[float]) -> Tuple[float, float, int]:
    """(mean, sample sd with ddof=1, n)."""
    a = _as_1d(x, "x")
    return (float(a.mean()) if a.size else float("nan"), sample_sd(a), int(a.size))


def fmt_mean_sd(x: Iterable[float], digits: int = 3) -> str:
    """'mean +- sd (n=k)' with a fixed number of decimals, for seed-averaged table cells."""
    m, s, n = mean_sd(x)
    s_txt = "nan" if math.isnan(s) else f"{s:.{digits}f}"
    return f"{m:.{digits}f} +- {s_txt} (n={n})"


def fmt_p(p: float) -> str:
    """p or q for tables: 3 significant digits, scientific below 1e-3, 'nan' when undefined."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "nan"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"


def win_count_line(n_wins: int, n_tests: int, what: str = "wins") -> str:
    """PROTOCOL 1.4 / D-03: every win count prints n_tests and 0.05 * n_tests (the number of
    'wins' expected under the global null at alpha 0.05)."""
    return f"{n_wins} {what} of n_tests={n_tests} (0.05 x n_tests = {0.05 * n_tests:.2f})"


# --------------------------------------------------------------------------- Welch
def welch(a: Iterable[float], b: Iterable[float]) -> Dict[str, float]:
    """Unpaired Welch t-test of mean(a) - mean(b), two-sided.

    Convention: ``a`` is the Dual arm and ``b`` the comparator, so delta > 0 favours the Dual
    (every primary metric is higher-is-better, PROTOCOL 1.2).

    Returns a dict with keys
      delta, ci_lo, ci_hi (95%, Welch-Satterthwaite df), p (two-sided), p_bh (the value to feed
      bh(), see below), t, df, se, mde80 (see module docstring), mde80_method, sd_a, sd_b
      (ddof=1), mean_a, mean_b, n_a, n_b, degenerate (bool), zero_variance (bool), note (str).

    Edge cases:
      * n < 2 in either arm: no variance estimate exists.  ``degenerate=True``, p = p_bh = NaN;
        PROTOCOL R5 makes the test "not assessable" and it leaves the family (verdict_from).
      * se == 0 with n >= 2 in both arms (every seed identical in both arms, e.g. a ceiling
        metric): the t statistic is undefined (scipy returns NaN).  This is NOT a pre-registered
        removal (PROTOCOL reserves "not assessable" for the outlier flag and R5), so the test STAYS
        in the family and PROTOCOL 1.4 is applied literally: CI = [delta, delta], p undefined (NaN,
        printed as such), and it enters BH with p_bh = 1.0 -- it counts in m (conservative for
        every other test) and can never be significant.  Its q is then >= 0.10, so the verdict
        is "equivalent" when [delta, delta] lies inside +-SESOI and "inconclusive (MDE80 = 0)"
        otherwise.  ``zero_variance=True`` flags it for the report.  (Declared reading; the
        PROTOCOL change log should record it before the first family is scored.)
    """
    xa, xb = _as_1d(a, "a"), _as_1d(b, "b")
    na, nb = int(xa.size), int(xb.size)
    out: Dict[str, float] = {"n_a": na, "n_b": nb, "degenerate": False, "zero_variance": False,
                             "note": "", "mde80_method": ""}
    out["mean_a"] = float(xa.mean()) if na else float("nan")
    out["mean_b"] = float(xb.mean()) if nb else float("nan")
    out["delta"] = out["mean_a"] - out["mean_b"]
    out["sd_a"], out["sd_b"] = sample_sd(xa), sample_sd(xb)
    nan = float("nan")
    if na < 2 or nb < 2:
        out.update(t=nan, df=nan, se=nan, p=nan, p_bh=nan, ci_lo=nan, ci_hi=nan, mde80=nan,
                   degenerate=True, note=f"n_a={na}, n_b={nb}: fewer than 2 seeds in an arm")
        return out
    va, vb = out["sd_a"] ** 2 / na, out["sd_b"] ** 2 / nb
    se = math.sqrt(va + vb)
    out["se"] = se
    if se == 0.0:
        # Kept in the family (see the docstring): p undefined, p_bh = 1 (counts in m, never
        # significant), zero-width CI, MDE80 = 0 (any nonzero effect would be "detected").
        out.update(t=nan, df=nan, p=nan, p_bh=1.0, ci_lo=out["delta"], ci_hi=out["delta"],
                   mde80=0.0, mde80_method="zero_variance", zero_variance=True,
                   note="zero variance in both arms: p undefined, enters BH as p=1")
        return out
    # Welch-Satterthwaite degrees of freedom.  When one arm has zero variance its term drops
    # out and df = n_other - 1, which is the correct limit.
    df = (va + vb) ** 2 / ((va ** 2 / (na - 1) if va > 0 else 0.0) + (vb ** 2 / (nb - 1) if vb > 0 else 0.0))
    t = out["delta"] / se
    tcrit = float(_st.t.ppf(1.0 - ALPHA / 2.0, df))
    d, method = mde80_ncp(df)
    p = float(2.0 * _st.t.sf(abs(t), df))
    out.update(
        t=float(t), df=float(df), p=p, p_bh=p,
        ci_lo=float(out["delta"] - tcrit * se), ci_hi=float(out["delta"] + tcrit * se),
        mde80=float(d * se), mde80_method=method,
    )
    return out


# --------------------------------------------------------------------------- MDE80
def _power_two_sided(ncp: float, df: float, tcrit: float) -> float:
    """Power of the two-sided t test at noncentrality ``ncp``: P(|T'| > tcrit)."""
    return float(_st.nct.sf(tcrit, df, ncp) + _st.nct.cdf(-tcrit, df, ncp))


def mde80_ncp(df: float, power: float = POWER, alpha: float = ALPHA) -> Tuple[float, str]:
    """The noncentrality d with P(|T'| > t_{1-alpha/2, df}) = ``power`` for T' ~ nct(df, d),
    i.e. MDE80 / se.  Returns (d, method): method 'nct' normally, 'approx' when the root-finder
    fails (then d = t_{1-alpha/2} + t_{power}, which understates the MDE; module docstring).

    Solved with brentq on [0, hi]: power(0) = alpha < 0.80 and power is increasing in d > 0, so
    the root is bracketed once hi reaches power >= 0.80 (hi starts at twice the shortcut value
    and doubles).  df may be non-integer (Welch)."""
    tcrit = float(_st.t.ppf(1.0 - alpha / 2.0, df))
    approx = tcrit + float(_st.t.ppf(power, df))
    try:
        from scipy.optimize import brentq
        hi = 2.0 * approx
        for _ in range(20):
            if _power_two_sided(hi, df, tcrit) >= power:
                break
            hi *= 2.0
        d = float(brentq(lambda x: _power_two_sided(x, df, tcrit) - power, 0.0, hi,
                         xtol=1e-10, rtol=1e-12, maxiter=200))
        if not math.isfinite(d):
            raise ValueError("non-finite root")
        return d, "nct"
    except Exception:                    # pragma: no cover - numerical fallback only
        return approx, "approx"


def mde80_for_sd(sd: float, n_a: int = 3, n_b: int = 3) -> float:
    """Projected MDE80 of a future n_a + n_b Welch test when both arms have seed sd ``sd``
    (PROTOCOL R8 / P3 record: projection from a pilot's pooled seed sd).  Equal sds give the
    Welch-Satterthwaite df = n_a + n_b - 2 exactly when n_a == n_b (and the usual Welch value
    otherwise); se = sd * sqrt(1/n_a + 1/n_b)."""
    if n_a < 2 or n_b < 2:
        raise ValueError("MDE80 needs n >= 2 per arm")
    va, vb = sd ** 2 / n_a, sd ** 2 / n_b
    se = math.sqrt(va + vb)
    if se == 0.0:
        return 0.0
    df = (va + vb) ** 2 / (va ** 2 / (n_a - 1) + vb ** 2 / (n_b - 1))
    return float(mde80_ncp(df)[0] * se)


# --------------------------------------------------------------------------- BH
def bh(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg q-values (step-up, monotone, capped at 1).

    ``pvals`` must be ALL tests of ONE family that remain after the health gates (PROTOCOL 1.2,
    1.5): gated tests are removed by the caller BEFORE this call; labelled tests stay (R4).

    Feed it ``welch(...)['p_bh']``, not ``['p']``: the two differ only for a zero-variance test
    (both arms constant), whose p is undefined but which stays in the family with p_bh = 1.0
    (counted in m, never significant; see welch).

    NaN p-values are returned as NaN q and are NOT counted in m.  They should not occur (an
    n < 2 test is "not assessable" and is removed with the gated ones before this call), and
    excluding them rather than counting them keeps q monotone in p; callers that want them
    counted must decide that explicitly.
    """
    p = np.asarray(pvals, dtype=np.float64).ravel()
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    if np.any((p[ok] < 0) | (p[ok] > 1)):
        raise ValueError(f"p-values outside [0,1]: {p[ok][(p[ok] < 0) | (p[ok] > 1)]}")
    pv = p[ok]
    m = pv.size
    order = np.argsort(pv, kind="mergesort")      # stable, so ties keep input order
    ranked = pv[order] * m / np.arange(1, m + 1)
    # Step-up: q_(i) = min_{j >= i} p_(j) m / j, then cap at 1.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qv = np.empty(m)
    qv[order] = np.minimum(ranked, 1.0)
    q[ok] = qv
    return q


# --------------------------------------------------------------------------- verdict words
def _fmt_mde(x: float) -> str:
    """MDE80 rendering inside the verdict: 3 significant digits (e.g. 'inconclusive (MDE80 =
    0.912)').  Callers needing another format pass ``mde_fmt`` to verdict()."""
    return f"{x:.3g}"


def ci_inside_sesoi(ci_lo: float, ci_hi: float, sesoi: float) -> bool:
    """PROTOCOL 1.4: 'the 95% CI lies inside (-SESOI, +SESOI)' -- the OPEN interval, so a CI
    touching +-SESOI is not inside."""
    if sesoi is None or not (sesoi > 0):
        raise ValueError(f"SESOI must be > 0, got {sesoi!r}")
    return bool(-sesoi < ci_lo and ci_hi < sesoi)


def verdict(delta: float, ci: Tuple[float, float], q: float, sesoi: float, mde80: float, *,
            comparator: Optional[str] = None, gate_label: Optional[str] = None,
            app_label: Optional[str] = None, mde_fmt=_fmt_mde) -> str:
    """The pre-registered verdict word for ONE confirmatory test (PROTOCOL 1.4).

    Steps, first match wins:
      1. ``gate_label`` given (a health gate removed the test, or the outlier flag fired):
         no verdict; the gate's label is returned as-is ("not sparsity-matched", "collapsed",
         "baseline collapsed", "not assessable").
      2. q < 0.10: "Dual better than <comparator>" (delta > 0) / "Dual worse than
         <comparator>" (delta < 0); rule R1 appends " (below SESOI)" when the CI also lies
         inside +-SESOI.
      3. q >= 0.10 and the CI lies inside +-SESOI: "equivalent".
      4. otherwise: "inconclusive (MDE80 = x)".
    ``app_label`` (an application-specific label, e.g. "scoring-dependent",
    "annotation-sensitive") REPLACES a step-2 or step-3 verdict and never a step-4 one
    (PROTOCOL 1.4: "App-specific labels replace a step 2 or step 3 verdict").  The test stays in
    BH's m (R4); that is the caller's business.

    A NaN q (degenerate test) is treated as NOT significant, so it can reach at most
    "equivalent" or "inconclusive"; degenerate tests should have been gated as "not assessable"
    before BH, and the caller should pass gate_label for them.
    """
    if gate_label:
        return gate_label
    lo, hi = float(ci[0]), float(ci[1])
    inside = ci_inside_sesoi(lo, hi, sesoi) if (math.isfinite(lo) and math.isfinite(hi)) else False
    who = f" than {comparator}" if comparator else ""
    word: Optional[str] = None
    sig = q is not None and math.isfinite(q) and q < Q_THRESHOLD
    if sig and delta > 0:
        word = f"Dual better{who}"
    elif sig and delta < 0:
        word = f"Dual worse{who}"
    if word is not None:                         # step 2
        if app_label:
            return app_label
        return word + (" (below SESOI)" if inside else "")
    if inside:                                   # step 3 (q >= 0.10, or q undefined)
        return app_label if app_label else "equivalent"
    # step 4.  (A significant test with delta exactly 0 cannot occur -- p = 1 there -- but if
    # it did it lands here rather than claiming a direction.)
    mtxt = "nan" if (mde80 is None or not math.isfinite(mde80)) else mde_fmt(mde80)
    return f"inconclusive (MDE80 = {mtxt})"


def verdict_from(res: Mapping[str, float], q: float, sesoi: float, **kw) -> str:
    """verdict() fed from a welch() result dict (delta, ci_lo, ci_hi, mde80).

    * n < 2 in an arm (``degenerate``): "not assessable" (PROTOCOL R5) unless the caller passes
      its own gate_label.
    * zero variance in both arms (``zero_variance``): NOT removed -- PROTOCOL 1.4 applies with
      CI = [delta, delta] and the q BH gave its p_bh = 1 (or NaN if the caller never ran BH on
      it; either way not significant), i.e. "equivalent" if |delta| < SESOI, else
      "inconclusive (MDE80 = 0)"."""
    if res.get("degenerate") and not kw.get("gate_label"):
        return "not assessable"
    if res.get("zero_variance") and q is not None and math.isfinite(q) and q < Q_THRESHOLD:
        # Cannot happen when q came from bh(): p_bh = 1 is the family's largest p, so its
        # step-up q is exactly 1 * m / m = 1.  A q < 0.10 here means the caller fed something
        # else; refuse rather than print a direction that no test supports.
        raise ValueError("a zero-variance test (p undefined) cannot be significant; pass the q "
                         "that bh() gave its p_bh")
    return verdict(res["delta"], (res["ci_lo"], res["ci_hi"]), q, sesoi, res["mde80"], **kw)


def exploratory_row(res: Mapping[str, float], digits: int = 3) -> str:
    """Exploratory results print delta, CI and raw p ONLY: no verdict words, no BH
    (PROTOCOL 1.4 last bullet, 1.7)."""
    return (f"delta={res['delta']:.{digits}f} 95% CI [{res['ci_lo']:.{digits}f}, "
            f"{res['ci_hi']:.{digits}f}] p={fmt_p(res['p'])}")


# --------------------------------------------------------------------------- outlier flag
def pooled_sd(groups: Iterable[Sequence[float]]) -> float:
    """Pooled within-group sample sd: sqrt( sum_g (n_g - 1) s_g^2 / sum_g (n_g - 1) ).

    Groups with fewer than 2 values contribute nothing (no within-group variance).  NaN when
    no group has 2+ values.
    """
    num, den = 0.0, 0
    for g in groups:
        a = _as_1d(g, "group")
        if a.size >= 2:
            num += float(np.sum((a - a.mean()) ** 2))
            den += a.size - 1
    return math.sqrt(num / den) if den > 0 else float("nan")


def outlier_seeds(values: Mapping[Hashable, float], sd_ref: float,
                  k: float = OUTLIER_K) -> List[Hashable]:
    """Seeds of ONE (cell, arm) whose value is more than k * sd_ref from the mean of the OTHER
    seeds of that arm.  Returns the flagged seed ids (sorted when sortable).

    With fewer than 2 seeds there is no "other seeds" mean, so nothing can be flagged.
    A zero sd_ref (every arm perfectly constant) flags any seed that differs at all, which is
    the literal rule; a NaN sd_ref flags nothing and should be reported by the caller.
    """
    if not math.isfinite(sd_ref):
        return []
    keys = list(values)
    if len(keys) < 2:
        return []
    x = np.array([float(values[s]) for s in keys])
    if not np.all(np.isfinite(x)):
        raise ValueError(f"non-finite seed values {dict(values)}")
    flagged = []
    for i, s in enumerate(keys):
        others = np.delete(x, i)
        if abs(x[i] - others.mean()) > k * sd_ref:
            flagged.append(s)
    try:
        return sorted(flagged)
    except TypeError:
        return flagged


def flag_outliers(cells: Mapping[Hashable, Mapping[Hashable, float]],
                  k: float = OUTLIER_K) -> Tuple[float, Dict[Hashable, List[Hashable]]]:
    """The PROTOCOL 1.5 outlier flag on the GLOBAL noise scale.

    ``cells`` maps a (cell, arm) key to {seed: value}, and must contain EVERY (cell, arm) of one
    metric at one K or layer -- that population defines sd_ref(metric, K or layer).
    Returns (sd_ref, {(cell, arm): [flagged seeds]}), listing only keys with at least one flag.

    Why global: the former rule estimated the sd from the other 2 seeds of the same arm (1 df)
    and flagged pure noise 74% of the time for a 3-seed arm (Monte Carlo over 200k
    iid N(0,1) draws).  Pooling over all cells of the metric gives sd_ref 2 df per (cell, arm),
    i.e. dozens to hundreds of df per family.
    """
    sd_ref = pooled_sd([list(v.values()) for v in cells.values()])
    flags = {key: fl for key, v in cells.items() if (fl := outlier_seeds(v, sd_ref, k))}
    return sd_ref, flags


# --------------------------------------------------------------------------- selftest
def _selftest() -> int:
    failures: List[str] = []

    def check(cond: bool, msg: str) -> None:
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            failures.append(msg)

    rs = np.random.default_rng(0)
    # 1. welch vs scipy.stats.ttest_ind(equal_var=False): t, p, df and the 95% CI.
    worst = 0.0
    for trial in range(300):
        na, nb = int(rs.integers(2, 8)), int(rs.integers(2, 8))
        a = rs.normal(rs.normal(), rs.uniform(0.1, 3), na)
        b = rs.normal(rs.normal(), rs.uniform(0.1, 3), nb)
        w = welch(a, b)
        ref = _st.ttest_ind(a, b, equal_var=False)
        ci = ref.confidence_interval(0.95)
        dev = max(abs(w["t"] - ref.statistic), abs(w["p"] - ref.pvalue), abs(w["df"] - ref.df),
                  abs(w["ci_lo"] - ci.low), abs(w["ci_hi"] - ci.high))
        worst = max(worst, dev)
    check(worst < 1e-9, f"welch == scipy ttest_ind(equal_var=False) over 300 draws "
                        f"(max |dev| {worst:.2e})")
    w = welch([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
    check(abs(w["delta"] - 1.0) < 1e-12 and abs(w["sd_a"] - 1.0) < 1e-12, "delta and ddof=1 sd")
    # MDE80 at 3+3 equal sd (df=4): the noncentral-t value D-03 quotes as ~3.07 sd.
    check(w["mde80_method"] == "nct" and abs(w["mde80"] - 3.071) < 5e-4,
          f"MDE80 at 3+3 = {w['mde80']:.4f} sd (exact noncentral t; PROTOCOL 'about 3.07')")
    tc4 = float(_st.t.ppf(0.975, 4))
    pw = _power_two_sided(w["mde80"] / w["se"], 4.0, tc4)
    check(abs(pw - 0.80) < 1e-8, f"power at the printed MDE80 = {pw:.6f} (== 0.80)")
    approx = (_st.t.ppf(0.975, 4) + _st.t.ppf(0.8, 4)) * math.sqrt(2 / 3)
    pw_apx = _power_two_sided(approx / math.sqrt(2 / 3), 4.0, tc4)
    check(approx < w["mde80"] and pw_apx < 0.80,
          f"the t-shortcut understates it ({approx:.4f} sd, true power {pw_apx:.3f}) -> fallback only")
    check(abs(mde80_for_sd(1.0) - w["mde80"]) < 1e-9, f"mde80_for_sd(1.0) at 3+3 = "
          f"{mde80_for_sd(1.0):.4f} (P3 projection helper)")
    # Monte Carlo cross-check of the noncentral-t arithmetic: with equal true sds and a true
    # effect of MDE80 (3+3), the POOLED t (exactly nct with df=4) rejects 80% of the time.
    trials = 200_000
    eff = mde80_for_sd(1.0)
    A_ = rs.normal(eff, 1.0, (trials, 3))
    B_ = rs.normal(0.0, 1.0, (trials, 3))
    pw_pool = float(np.mean(_st.ttest_ind(A_, B_, axis=1, equal_var=True).pvalue < ALPHA))
    check(abs(pw_pool - 0.80) < 0.005, f"Monte Carlo pooled-t power at MDE80 = {pw_pool:.4f} (0.80)")
    # For the record (printed, not checked): the Welch test estimates df from the data (often
    # < 4 at 3+3), so its realised power at the same true effect is lower.
    pw_welch = float(np.mean(_st.ttest_ind(A_, B_, axis=1, equal_var=False).pvalue < ALPHA))
    print(f"         (Welch realised power at the same effect: {pw_welch:.3f}; the MDE80 printed "
          f"per test uses that test's own Welch df)")
    check(sample_sd([0.9, 0.9, 0.9]) == 0.0, "identical seeds -> sd exactly 0 (not ~1e-16)")
    d = welch([1.0], [0.0, 1.0])
    check(d["degenerate"] and math.isnan(d["p"]) and math.isnan(d["p_bh"]),
          "n<2 arm -> degenerate, p NaN, p_bh NaN")
    d = welch([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    check((not d["degenerate"]) and d["zero_variance"] and math.isnan(d["p"]) and d["p_bh"] == 1.0
          and d["ci_lo"] == d["ci_hi"] == 0.0 and d["mde80"] == 0.0,
          "zero variance in both arms -> kept in family: p NaN, p_bh 1, CI [0,0], MDE80 0")
    try:
        welch([1.0, float("nan"), 2.0], [0.0, 1.0])
        check(False, "NaN seed value rejected")
    except ValueError:
        check(True, "NaN seed value rejected")

    # 2. BH: a hand example (m=5) and statsmodels.
    p = [0.01, 0.04, 0.03, 0.005, 0.20]
    # sorted: 0.005(1) 0.01(2) 0.03(3) 0.04(4) 0.20(5); raw p*m/i = .025 .025 .05 .05 .20;
    # step-up minima leave them unchanged.
    hand = np.array([0.025, 0.05, 0.05, 0.025, 0.20])
    check(np.allclose(bh(p), hand, atol=1e-12), f"BH hand example {bh(p).round(4).tolist()}")
    # A case where monotonicity matters: raw = [.02*3/1=.06, .03*3/2=.045, .04*3/3=.04] ->
    # step-up minima [.04, .04, .04].
    check(np.allclose(bh([0.02, 0.03, 0.04]), [0.04, 0.04, 0.04]), "BH step-up monotonicity")
    from statsmodels.stats.multitest import multipletests
    pr = rs.uniform(0, 0.2, 57)
    check(np.allclose(bh(pr), multipletests(pr, method="fdr_bh")[1], atol=1e-12),
          "BH == statsmodels fdr_bh on 57 random p")
    qn = bh([0.01, float("nan"), 0.02])
    check(math.isnan(qn[1]) and np.allclose(qn[[0, 2]], [0.02, 0.02]),
          "NaN p -> NaN q, excluded from m")

    # 3. Verdict words, one per PROTOCOL step, plus R1 and the label rules.
    S = 0.3
    check(verdict(1.0, (0.5, 1.5), 0.01, S, 0.9, comparator="baseline") ==
          "Dual better than baseline", "step 2 better")
    check(verdict(-1.0, (-1.5, -0.5), 0.01, S, 0.9, comparator="dual_g0") ==
          "Dual worse than dual_g0", "step 2 worse")
    check(verdict(0.1, (0.05, 0.2), 0.05, S, 0.9, comparator="baseline") ==
          "Dual better than baseline (below SESOI)", "R1 (below SESOI) suffix")
    check(verdict(0.05, (-0.2, 0.25), 0.5, S, 0.9) == "equivalent", "step 3 equivalent")
    check(verdict(0.05, (-0.3, 0.25), 0.5, S, 0.9) == "inconclusive (MDE80 = 0.9)",
          "CI touching -SESOI is not inside (open interval)")
    check(verdict(0.5, (-0.4, 1.4), 0.3, S, 1.234) == "inconclusive (MDE80 = 1.23)", "step 4")
    check(verdict(1.0, (0.5, 1.5), 0.01, S, 0.9, gate_label="collapsed") == "collapsed",
          "step 1 gate label")
    check(verdict(1.0, (0.5, 1.5), 0.01, S, 0.9, app_label="scoring-dependent") ==
          "scoring-dependent", "app label replaces step 2")
    check(verdict(0.05, (-0.2, 0.25), 0.5, S, 0.9, app_label="scoring-dependent") ==
          "scoring-dependent", "app label replaces step 3")
    check(verdict(0.5, (-0.4, 1.4), 0.3, S, 1.0, app_label="scoring-dependent").startswith(
          "inconclusive"), "app label never replaces step 4")
    w3 = welch([1.0, 1.2, 0.9], [0.1, 0.2, 0.0])
    check(verdict_from(w3, 0.01, S, comparator="baseline") == "Dual better than baseline",
          "verdict_from(welch result)")
    check(verdict_from(welch([1.0], [0.0, 1.0]), 0.01, S) == "not assessable",
          "verdict_from on an n<2 test -> not assessable (R5)")
    # Zero variance in both arms: PROTOCOL 1.4 literally (q from BH with p_bh = 1).
    z_in = welch([0.9, 0.9, 0.9], [0.8, 0.8, 0.8])            # delta 0.1 < SESOI 0.3
    z_out = welch([1.4, 1.4, 1.4], [0.8, 0.8, 0.8])           # delta 0.6 > SESOI 0.3
    qz = bh([0.001, z_in["p_bh"], z_out["p_bh"]])
    check(abs(qz[0] - 0.003) < 1e-15 and qz[1] == 1.0 and qz[2] == 1.0,
          f"zero-variance tests count in BH's m (q of a p=0.001 test = {qz[0]:.3f} with m=3)")
    check(verdict_from(z_in, qz[1], S, comparator="baseline") == "equivalent",
          "zero variance, |delta| < SESOI -> equivalent")
    check(verdict_from(z_out, qz[2], S, comparator="baseline") == "inconclusive (MDE80 = 0)",
          "zero variance, |delta| > SESOI -> inconclusive (MDE80 = 0)")
    try:
        verdict_from(z_out, 0.01, S)
        check(False, "a significant q on a zero-variance test is refused")
    except ValueError:
        check(True, "a significant q on a zero-variance test is refused")
    words = [verdict(float(dl), (float(dl) - float(h), float(dl) + float(h)), float(qq), S, 1.0)
             for dl in np.linspace(-1, 1, 9) for h in (0.05, 0.5) for qq in (0.01, 0.2)]
    check(not any(fw in wd for wd in words for fw in FORBIDDEN_WORDS),
          "no forbidden words ('no effect', 'parity') in any verdict")

    # 4. Outlier flag: pooled sd and the 4*sd_ref rule.
    cells = {("c1", "dual"): {42: 1.0, 43: 1.1, 44: 0.9},
             ("c1", "base"): {42: 2.0, 43: 2.2, 44: 1.8},
             ("c2", "dual"): {42: 1.0, 43: 1.05, 44: 5.0}}
    sd_ref, flags = flag_outliers(cells)
    manual = math.sqrt((0.02 + 0.08 + float(np.sum((np.array([1, 1.05, 5.0]) -
                                                     np.mean([1, 1.05, 5.0])) ** 2))) / 6)
    check(abs(sd_ref - manual) < 1e-12, f"pooled sd_ref = {sd_ref:.4f}")
    # |5.0 - 1.025| = 3.975 vs 4*sd_ref (= 4*1.33 = 5.33): NOT flagged; with only 3 cells the
    # outlier itself inflates the pooled sd, which is why the rule needs a large population.
    check(flags == {}, "a seed within 4*sd_ref is not flagged")
    big = {("c%d" % i, "a"): {42: 1.0 + 0.01 * i, 43: 1.02 + 0.01 * i, 44: 0.98 + 0.01 * i}
           for i in range(20)}
    big[("cx", "a")] = {42: 1.0, 43: 1.0, 44: 1.5}
    sd_ref, flags = flag_outliers(big)
    check(flags == {("cx", "a"): [44]}, f"outlier seed flagged on the global scale "
                                        f"(sd_ref {sd_ref:.4f}, flags {flags})")
    # Monte Carlo sanity of the design claim: with 21 cells of 3 iid N(0,1) seeds, a 4*sd_ref
    # flag should fire rarely (vs 74% for the old 1-df rule).
    fires, trials = 0, 2000
    for _ in range(trials):
        c = {i: dict(zip((42, 43, 44), rs.normal(size=3))) for i in range(21)}
        _, fl = flag_outliers(c)
        fires += 0 in fl
    check(fires / trials < 0.02, f"P(flag a 3-seed null arm) = {fires / trials:.4f} (< 0.02)")

    print("STATS SELFTEST", "OK" if not failures else f"FAILED ({len(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(_selftest())
    ap.print_help()
