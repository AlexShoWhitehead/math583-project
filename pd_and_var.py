#!/usr/bin/env python3
"""
compute_pd_and_credit_var_normalized.py

Compute portfolio and bucket-level credit risk metrics, including
PER-COMPANY normalized EL/VaR/CreditVaR/ES so segments are comparable
even when segment sizes differ.

Outputs:
- pd_table.csv
- credit_var_normalized.json

Usage:
  python compute_pd_and_credit_var_normalized.py \
    --preds_csv preds.csv \
    --id_col company_folder \
    --asof_latest \
    --segment_col predicted_bucket \
    --segment_values healthy,risky,failed \
    --alpha_risky 0.5 \
    --unknown_floor 0.10 \
    --rho 0.15 \
    --n_sims 200000 \
    --conf_level 0.99
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import norm
    from scipy.special import expit, logit
    from scipy.optimize import brentq
except Exception as e:
    raise RuntimeError("This script requires SciPy. Install with: pip install scipy") from e


# ----------------------------
# As-of selection
# ----------------------------

def pick_latest_per_obligor(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    out = df.copy()
    if "period_of_report" in out.columns:
        t = pd.to_datetime(out["period_of_report"], errors="coerce")
        out["_t"] = t
        out = out.sort_values([id_col, "_t"], kind="mergesort")
        return out.groupby(id_col, as_index=False).tail(1).drop(columns=["_t"])

    if ("q_year" in out.columns) and ("q_num" in out.columns):
        y = pd.to_numeric(out["q_year"], errors="coerce")
        q = pd.to_numeric(out["q_num"], errors="coerce")
        out["_t"] = (y * 4 + q)
        out = out.sort_values([id_col, "_t"], kind="mergesort")
        return out.groupby(id_col, as_index=False).tail(1).drop(columns=["_t"])

    return out.groupby(id_col, as_index=False).tail(1)


# ----------------------------
# PD construction
# ----------------------------

def compute_confidence_from_probs(df: pd.DataFrame) -> Optional[pd.Series]:
    pcols = [c for c in df.columns if c.startswith("p_")]
    if not pcols:
        return None
    probs = df[pcols].apply(pd.to_numeric, errors="coerce")
    return probs.max(axis=1)


def build_pd_vector(
    df: pd.DataFrame,
    alpha_risky: float,
    reject_below: Optional[float],
    unknown_floor: Optional[float],
    pd_col: str = "pd",
) -> Tuple[np.ndarray, pd.DataFrame]:
    out = df.copy()

    if "confidence" not in out.columns:
        conf = compute_confidence_from_probs(out)
        if conf is not None:
            out["confidence"] = conf

    if "flag_unknown" not in out.columns:
        out["flag_unknown"] = 0
        if reject_below is not None and "confidence" in out.columns:
            out["flag_unknown"] = (pd.to_numeric(out["confidence"], errors="coerce") < float(reject_below)).astype(int)

    if pd_col in out.columns:
        pd_raw = pd.to_numeric(out[pd_col], errors="coerce")
    else:
        if "p_failed" not in out.columns or "p_risky" not in out.columns:
            raise ValueError("Need either 'pd' OR both 'p_failed' and 'p_risky'.")
        p_failed = pd.to_numeric(out["p_failed"], errors="coerce").fillna(0.0)
        p_risky = pd.to_numeric(out["p_risky"], errors="coerce").fillna(0.0)
        pd_raw = (p_failed + float(alpha_risky) * p_risky).clip(0.0, 1.0)

    out["pd_raw"] = pd_raw

    pd_used = pd_raw.copy()
    if unknown_floor is not None:
        unk = pd.to_numeric(out["flag_unknown"], errors="coerce").fillna(0).astype(int)
        pd_used = pd_used.where(unk == 0, np.maximum(pd_used, float(unknown_floor)))

    eps = 1e-8
    pd_used = pd.to_numeric(pd_used, errors="coerce").fillna(0.0).clip(eps, 1.0 - eps)
    out["pd_used"] = pd_used
    return pd_used.to_numpy(dtype=float), out


def calibrate_pd_to_target_mean(pd: np.ndarray, target_mean: float) -> np.ndarray:
    target_mean = float(target_mean)
    if not (0.0 < target_mean < 1.0):
        raise ValueError("--target_mean_pd must be in (0,1).")

    pd = np.clip(pd, 1e-8, 1.0 - 1e-8)
    x = logit(pd)

    def f(a: float) -> float:
        return float(expit(x + a).mean() - target_mean)

    a_star = brentq(f, -50.0, 50.0)
    return expit(x + a_star)


# ----------------------------
# Simulation
# ----------------------------

def simulate_losses(
    pd: np.ndarray,
    ead: np.ndarray,
    lgd: np.ndarray,
    n_sims: int,
    rho: float,
    seed: int,
    chunk_sims: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(pd)

    pd = np.clip(pd, 1e-8, 1.0 - 1e-8)
    thr = norm.ppf(pd)

    ead = np.asarray(ead, dtype=float)
    lgd = np.asarray(lgd, dtype=float)
    severity = ead * lgd

    rho = float(rho)
    if rho < 0.0 or rho >= 1.0:
        raise ValueError("--rho must be in [0,1).")

    losses = np.empty(n_sims, dtype=float)
    done = 0

    while done < n_sims:
        m = min(chunk_sims, n_sims - done)

        if rho == 0.0:
            u = rng.random((m, n))
            d = (u < pd).astype(np.float32)
        else:
            z = rng.standard_normal(m)
            eps = rng.standard_normal((m, n))
            a = (np.sqrt(rho) * z[:, None]) + (np.sqrt(1.0 - rho) * eps)
            d = (a < thr[None, :]).astype(np.float32)

        losses[done:done + m] = d @ severity
        done += m

    return losses


def metrics_total_and_per_company(losses: np.ndarray, alpha: float, n_names: int) -> Dict[str, float]:
    """
    Returns metrics for:
      - totals
      - per-company normalized (divide losses by n_names first)
    """
    losses = np.asarray(losses, dtype=float)
    if n_names <= 0:
        return {"n": 0}

    # total
    el = float(losses.mean())
    var_a = float(np.quantile(losses, alpha))
    credit_var = var_a - el
    tail = losses[losses >= var_a]
    es = float(tail.mean()) if tail.size else var_a

    # per-company
    per = losses / float(n_names)
    el_p = float(per.mean())
    var_p = float(np.quantile(per, alpha))
    credit_var_p = var_p - el_p
    tail_p = per[per >= var_p]
    es_p = float(tail_p.mean()) if tail_p.size else var_p

    return {
        "n": int(n_names),
        "total_EL": el,
        "total_VaR": var_a,
        "total_CreditVaR": credit_var,
        "total_ES": es,
        "per_company_EL": el_p,
        "per_company_VaR": var_p,
        "per_company_CreditVaR": credit_var_p,
        "per_company_ES": es_p,
    }


def safe_write_json(path: str, payload: dict) -> str:
    out_json = Path(path).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
    return str(out_json)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--preds_csv", type=str, required=True)
    ap.add_argument("--id_col", type=str, default="company_folder")
    ap.add_argument("--asof_latest", action="store_true")

    ap.add_argument("--segment_col", type=str, default="predicted_bucket")
    ap.add_argument("--segment_values", type=str, default="healthy,risky,failed")

    ap.add_argument("--alpha_risky", type=float, default=0.5)
    ap.add_argument("--reject_below", type=float, default=None)
    ap.add_argument("--unknown_floor", type=float, default=None)
    ap.add_argument("--target_mean_pd", type=float, default=None)

    ap.add_argument("--ead_col", type=str, default=None)
    ap.add_argument("--ead_const", type=float, default=1.0)
    ap.add_argument("--lgd_col", type=str, default=None)
    ap.add_argument("--lgd_const", type=float, default=0.45)

    ap.add_argument("--rho", type=float, default=0.15)
    ap.add_argument("--n_sims", type=int, default=200000)
    ap.add_argument("--chunk_sims", type=int, default=2000)
    ap.add_argument("--conf_level", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--out_pd_table_csv", type=str, default="pd_table.csv")
    ap.add_argument("--out_summary_json", type=str, default="credit_var_normalized.json")

    args = ap.parse_args()

    df = pd.read_csv(args.preds_csv)
    if args.id_col not in df.columns:
        raise ValueError(f"id_col '{args.id_col}' not found in preds_csv.")

    if args.asof_latest:
        df = pick_latest_per_obligor(df, args.id_col)

    # PD
    pd_used, df_pd = build_pd_vector(
        df=df,
        alpha_risky=args.alpha_risky,
        reject_below=args.reject_below,
        unknown_floor=args.unknown_floor,
        pd_col="pd",
    )

    if args.target_mean_pd is not None:
        pd_used = calibrate_pd_to_target_mean(pd_used, float(args.target_mean_pd))
        df_pd["pd_used"] = pd_used
        print(f"[info] Calibrated PDs to mean={args.target_mean_pd}", file=sys.stderr)

    # EAD/LGD
    if args.ead_col and args.ead_col in df_pd.columns:
        ead = pd.to_numeric(df_pd[args.ead_col], errors="coerce").fillna(args.ead_const).to_numpy(dtype=float)
    else:
        ead = np.full(len(df_pd), float(args.ead_const), dtype=float)

    if args.lgd_col and args.lgd_col in df_pd.columns:
        lgd = pd.to_numeric(df_pd[args.lgd_col], errors="coerce").fillna(args.lgd_const).to_numpy(dtype=float)
    else:
        lgd = np.full(len(df_pd), float(args.lgd_const), dtype=float)

    df_pd["EAD_used"] = ead
    df_pd["LGD_used"] = lgd

    # write pd table
    out_pd = Path(args.out_pd_table_csv).expanduser().resolve()
    out_pd.parent.mkdir(parents=True, exist_ok=True)
    df_pd.to_csv(out_pd, index=False)
    print(f"[info] Wrote PD table to: {out_pd}", file=sys.stderr)

    alpha = float(args.conf_level)
    if not (0.0 < alpha < 1.0):
        raise ValueError("--conf_level must be in (0,1).")

    summary: Dict[str, dict] = {
        "meta": {
            "n_obligors": int(len(df_pd)),
            "n_sims": int(args.n_sims),
            "rho": float(args.rho),
            "conf_level": alpha,
            "mean_pd_used": float(np.mean(pd_used)),
            "sum_ead": float(np.sum(ead)),
            "mean_lgd": float(np.mean(lgd)),
            "segment_col": args.segment_col if args.segment_col in df_pd.columns else None,
        },
        "portfolio": {},
        "by_segment": {},
    }

    # Portfolio simulation (total portfolio loss)
    losses_all = simulate_losses(pd_used, ead, lgd, int(args.n_sims), float(args.rho), int(args.seed), int(args.chunk_sims))
    summary["portfolio"] = metrics_total_and_per_company(losses_all, alpha=alpha, n_names=len(df_pd))

    print("\n[results] Portfolio:", file=sys.stderr)
    p = summary["portfolio"]
    print(f"  n={p['n']} | total EL={p['total_EL']:.6f} VaR={p['total_VaR']:.6f} CreditVaR={p['total_CreditVaR']:.6f} ES={p['total_ES']:.6f}", file=sys.stderr)
    print(f"         | per-company EL={p['per_company_EL']:.6f} VaR={p['per_company_VaR']:.6f} CreditVaR={p['per_company_CreditVaR']:.6f} ES={p['per_company_ES']:.6f}", file=sys.stderr)

    # Segments
    if args.segment_col in df_pd.columns:
        seg_col = df_pd[args.segment_col].astype(str).str.lower().str.strip()
        seg_values = [s.strip().lower() for s in args.segment_values.split(",") if s.strip()]
        print("\n[results] By segment (TOTAL and PER-COMPANY):", file=sys.stderr)

        for seg in seg_values:
            mask = (seg_col == seg)
            nseg = int(mask.sum())
            if nseg == 0:
                summary["by_segment"][seg] = None
                print(f"  {seg}: empty", file=sys.stderr)
                continue

            pd_seg = pd_used[mask.values]
            ead_seg = ead[mask.values]
            lgd_seg = lgd[mask.values]

            losses_seg = simulate_losses(pd_seg, ead_seg, lgd_seg, int(args.n_sims), float(args.rho), int(args.seed), int(args.chunk_sims))
            m = metrics_total_and_per_company(losses_seg, alpha=alpha, n_names=nseg)
            summary["by_segment"][seg] = m

            print(
                f"  {seg}: n={nseg} | "
                f"total EL={m['total_EL']:.6f} VaR={m['total_VaR']:.6f} CreditVaR={m['total_CreditVaR']:.6f} ES={m['total_ES']:.6f} | "
                f"per-company EL={m['per_company_EL']:.6f} VaR={m['per_company_VaR']:.6f} CreditVaR={m['per_company_CreditVaR']:.6f} ES={m['per_company_ES']:.6f}",
                file=sys.stderr
            )

    out_json = safe_write_json(args.out_summary_json, summary)
    print(f"\n[info] Wrote summary JSON to: {out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
