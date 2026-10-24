"""Summarize experiment JSONL: means, paired bootstrap CIs, rate fits.

Paired differences are per (n, eps, seed): naive minus dpmoment, so a
positive mean difference with CI excluding 0 means dpmoment wins.
Rate fit: log(error) ~ a + b*log(n) at fixed eps; b is the fitted
exponent (theory for fixed d is roughly n^-1/2 up to logs for the
statistical part, worse under privacy; we report b honestly).
"""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "src")
from experiment import paired_bootstrap_ci  # noqa: E402


def summarize(path):
    rows = [json.loads(l) for l in open(path)]
    groups = {}
    for r in rows:
        groups.setdefault((r["n"], r["eps"]), []).append(r)
    out = {"groups": {}}
    for (n, eps), rs in sorted(groups.items()):
        dp_mmd = np.array([r["dpmoment_mmd"] for r in rs])
        nv_mmd = np.array([r["naive_mmd"] for r in rs])
        np_mmd = np.array([r["nonprivate_mmd"] for r in rs])
        dp_sw = np.array([r["dpmoment_sw1"] for r in rs])
        nv_sw = np.array([r["naive_sw1"] for r in rs])
        np_sw = np.array([r["nonprivate_sw1"] for r in rs])
        d_mmd = nv_mmd - dp_mmd
        d_sw = nv_sw - dp_sw
        g_mmd = dp_mmd - np_mmd
        ci_mmd = paired_bootstrap_ci(d_mmd)
        ci_sw = paired_bootstrap_ci(d_sw)
        ci_gap = paired_bootstrap_ci(g_mmd)
        out["groups"][f"n={n},eps={eps}"] = {
            "reps": len(rs),
            "dpmoment_mmd_mean": float(dp_mmd.mean()),
            "naive_mmd_mean": float(nv_mmd.mean()),
            "nonprivate_mmd_mean": float(np_mmd.mean()),
            "mmd_advantage_mean": float(d_mmd.mean()),
            "mmd_advantage_ci95": [float(ci_mmd[0]), float(ci_mmd[1])],
            "dpmoment_sw1_mean": float(dp_sw.mean()),
            "naive_sw1_mean": float(nv_sw.mean()),
            "nonprivate_sw1_mean": float(np_sw.mean()),
            "sw1_advantage_mean": float(d_sw.mean()),
            "sw1_advantage_ci95": [float(ci_sw[0]), float(ci_sw[1])],
            "utility_gap_mmd_mean": float(g_mmd.mean()),
            "utility_gap_mmd_ci95": [float(ci_gap[0]), float(ci_gap[1])],
            "accountant_ok_all": bool(all(r["accountant_ok"] for r in rs)),
            "max_implied_eps": float(max(r["implied_eps"] for r in rs)),
        }
    # rate fits at fixed eps (full mode only; pilot has one n)
    for eps in sorted({r["eps"] for r in rows}):
        pts = [(r["n"], r["dpmoment_mmd"]) for r in rows if r["eps"] == eps]
        ns = sorted({n for n, _ in pts})
        if len(ns) >= 2:
            means = [np.mean([v for n, v in pts if n == nn]) for nn in ns]
            b, a = np.polyfit(np.log(ns), np.log(means), 1)
            out[f"rate_fit_eps={eps}"] = {
                "n_values": ns, "mean_mmd": [float(x) for x in means],
                "fitted_exponent": float(b),
            }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    s = summarize(a.input)
    txt = json.dumps(s, indent=2)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
