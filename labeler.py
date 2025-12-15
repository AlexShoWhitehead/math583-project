#!/usr/bin/env python3
"""
generate_labels.py

Create a labels file for classification (healthy / risky / failed) from a
per-company financials CSV.

Usage:
  python generate_labels.py --input per_company_year_financials.csv --output per_company_year_labels.csv
  python generate_labels.py --input per_company_quarter_financials.csv --output per_company_quarter_labels.csv

Behavior:
- If the input already contains `company_bucket`, we treat it as the ground-truth label and output it.
- If `company_bucket` is missing, we attempt a *heuristic* label using the four-equations inputs:
    liquidity_ratio = H_proxy / A_total
    capital_ratio   = E_total / A_total   (if E_total missing, use A_total - L_total)
  and compare to thresholds (lambda_liq, kappa_cap) if present, else defaults.
  This heuristic is only meant for bootstrapping; prefer real labels when possible.
"""

from __future__ import annotations

import argparse
import sys
import pandas as pd
import numpy as np


LABEL_TO_INT = {"healthy": 0, "risky": 1, "failed": 2}


def _coerce_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def derive_equity(df: pd.DataFrame) -> pd.Series:
    """Try to produce an equity series."""
    if "E_total" in df.columns:
        e = _coerce_float(df["E_total"])
        if e.notna().any():
            return e
    if "A_total" in df.columns and "L_total" in df.columns:
        a = _coerce_float(df["A_total"])
        l = _coerce_float(df["L_total"])
        return a - l
    return pd.Series([np.nan] * len(df), index=df.index)


def heuristic_label(df: pd.DataFrame) -> pd.Series:
    """
    Heuristic label from the "four equations" inputs:
      - liquidity_ratio = H_proxy / A_total
      - capital_ratio   = E_total / A_total
    Fail if either ratio is severely below threshold.
    Risky if close to threshold.
    Healthy otherwise.
    """
    a = _coerce_float(df.get("A_total", np.nan))
    h = _coerce_float(df.get("H_proxy", np.nan))
    e = derive_equity(df)

    # thresholds: use per-row if provided; else fall back to typical values used in your dataset examples
    lam = _coerce_float(df["lambda_liq"]) if "lambda_liq" in df.columns else 0.05
    kap = _coerce_float(df["kappa_cap"]) if "kappa_cap" in df.columns else 0.08

    # ratios
    liq = h / a
    cap = e / a

    # define bands
    # - "failed": materially below either covenant/threshold
    # - "risky": near the threshold or with negative equity
    failed = (liq < (lam * 0.6)) | (cap < (kap * 0.6))
    risky = ((liq < lam) | (cap < kap) | (e < 0)) & (~failed)

    out = pd.Series(np.where(failed, "failed", np.where(risky, "risky", "healthy")), index=df.index)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV (yearly or quarterly)")
    ap.add_argument("--output", required=True, help="Output labels CSV path")
    ap.add_argument("--id-cols", default="", help="Comma-separated id columns to include (optional)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    if "company_bucket" in df.columns:
        label = df["company_bucket"].astype(str)
    else:
        label = heuristic_label(df)

    # Choose a sensible default set of id columns if not provided
    if args.id_cols.strip():
        id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]
    else:
        candidates = [
            "company_folder", "company_name", "ticker",
            "cik", "accn", "form_type", "period_dt",
            "q_year", "q_num",
        ]
        id_cols = [c for c in candidates if c in df.columns]

    out = df[id_cols].copy() if id_cols else pd.DataFrame(index=df.index)
    out["label"] = label
    out["label_num"] = out["label"].map(LABEL_TO_INT)

    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} labels to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
