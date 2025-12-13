#!/usr/bin/env python3
"""
Regime-aware SEC filings ML pipeline for "healthy/risky" labeling.

Key idea:
- Treat the dataset as multiple reporting regimes (A/B/C/D).
- Only use high-trust rows for solvency risk prediction.
- Keep "growth/ops" signals separate from "balance sheet risk" signals.
- Use flags as features and as data-quality gates.

Usage:
  python regime_aware_risk_pipeline.py \
      --input data.csv \
      --outdir out \
      --label-col 0

Notes:
- Input is assumed headerless, one record per line, comma-separated.
- We do NOT assume a fixed number of columns after the early metadata.
- We parse from both ends:
    left side: label, company, paths, name/cik/accession/form/date/year...
    right side: filed_date, flags, trailing booleans/floats/numerics
- Everything in the "middle" becomes numeric features if parseable.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib


# -----------------------------
# Parsing helpers
# -----------------------------

BOOL_TRUE = {"true", "t", "1", "yes", "y"}
BOOL_FALSE = {"false", "f", "0", "no", "n"}

def safe_strip(x: str) -> str:
    return x.strip().strip('"').strip()

def parse_bool(x: str) -> Optional[bool]:
    if x is None:
        return None
    s = safe_strip(str(x)).lower()
    if s in BOOL_TRUE:
        return True
    if s in BOOL_FALSE:
        return False
    return None

def parse_float(x: str) -> Optional[float]:
    if x is None:
        return None
    s = safe_strip(str(x))
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None

def parse_int(x: str) -> Optional[int]:
    f = parse_float(x)
    if f is None:
        return None
    if not math.isfinite(f):
        return None
    return int(f)

def parse_date_yyyymmdd(x: str) -> Optional[pd.Timestamp]:
    if x is None:
        return None
    s = safe_strip(str(x))
    if not s:
        return None
    # Accept YYYYMMDD or YYYY-MM-DD
    try:
        if re.fullmatch(r"\d{8}", s):
            return pd.Timestamp(datetime.strptime(s, "%Y%m%d").date())
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return pd.Timestamp(datetime.strptime(s, "%Y-%m-%d").date())
    except Exception:
        return None
    return None

def tokenize_flags(flag_str: str) -> List[str]:
    if flag_str is None:
        return []
    s = safe_strip(str(flag_str))
    if not s:
        return []
    return [t for t in s.split(";") if t]

def is_probably_flags(s: str) -> bool:
    # Heuristic: flags typically contain semicolons and recognizable tokens
    if s is None:
        return False
    st = safe_strip(str(s))
    if ";" not in st:
        return False
    # common tokens in your data
    tokens = ["InlineXBRL", "MissingH", "LabelFallback", "SANITY_FAIL", "SALVAGE", "HProxyIsCashOnly", "SGML_"]
    return any(t in st for t in tokens)

def is_iso_date(s: str) -> bool:
    return parse_date_yyyymmdd(s) is not None

@dataclass
class ParsedRow:
    label_raw: str
    label: int                   # risky=1, healthy=0
    issuer: str                  # second field e.g., Oracle
    form: Optional[str]
    report_date: Optional[pd.Timestamp]   # from YYYYMMDD field (e.g., 20250531)
    fiscal_year: Optional[int]
    filed_date: Optional[pd.Timestamp]    # last field (often YYYY-MM-DD)
    cik: Optional[str]
    accession: Optional[str]
    company_name: Optional[str]
    path: Optional[str]
    flags_raw: Optional[str]
    flags: List[str]
    # trailing parsed signals
    bool_tail: Dict[str, Optional[bool]]
    float_tail: Dict[str, Optional[float]]
    # generic numeric “middle”
    numeric_middle: List[Optional[float]]
    # raw length for diagnostics
    n_fields: int


def map_label(s: str) -> Optional[int]:
    if s is None:
        return None
    st = safe_strip(str(s)).lower()
    if st == "risky":
        return 1
    if st == "healthy":
        return 0
    return None


def parse_row_fields(fields: List[str]) -> ParsedRow:
    """
    Parse a headerless record using robust heuristics.

    Expected-ish prefix (based on your sample):
      0 label (healthy/risky)
      1 issuer short name (Oracle)
      2 path-ish
      3 path-ish duplicate
      4 company name (may be blank)
      5 cik (may be blank)
      6 accession (may be blank)
      7 form (10-K/10-Q/8-K)
      8 report_date yyyymmdd
      9 fiscal_year (YYYY)

    Suffix (based on your sample):
      last: filed_date yyyy-mm-dd
      second last: flags string (semicolon delimited)
      before that: 0-2 bools (True/False)
      before that: some thresholds (0.05, 0.08) etc.
      before that: ratios / derived floats
      Everything else: numeric_middle
    """
    n = len(fields)
    f = [safe_strip(x) for x in fields]
    # Basic prefix
    label_raw = f[0] if n > 0 else ""
    label = map_label(label_raw)
    if label is None:
        # If unknown, treat as missing; downstream can drop
        label = -1

    issuer = f[1] if n > 1 else ""

    # Grab likely suffix: filed_date, flags
    filed_date = parse_date_yyyymmdd(f[-1]) if n >= 1 else None

    flags_raw = None
    flags = []
    if n >= 2 and is_probably_flags(f[-2]):
        flags_raw = f[-2]
        flags = tokenize_flags(flags_raw)
        suffix_start = n - 2
    else:
        # Sometimes flags might be -3 if there's a trailing empty; try a bit
        suffix_start = n
        for k in range(2, min(6, n) + 1):
            if is_probably_flags(f[-k]):
                flags_raw = f[-k]
                flags = tokenize_flags(flags_raw)
                suffix_start = n - k
                break

    # Identify prefix metadata by position if present
    form = f[7] if n > 7 else None
    report_date = parse_date_yyyymmdd(f[8]) if n > 8 else None
    fiscal_year = parse_int(f[9]) if n > 9 else None

    path = f[3] if n > 3 else (f[2] if n > 2 else None)
    company_name = f[4] if n > 4 else None
    cik = f[5] if n > 5 and f[5] else None
    accession = f[6] if n > 6 and f[6] else None

    # Middle region: from after prefix to before suffix_start
    # Prefix_end index: we take 10 as "usual", but do not assume it's present.
    prefix_end = min(10, n)
    middle = f[prefix_end:suffix_start]

    # Now parse "tail" just before flags area when flags exist,
    # or from the end if flags not found.
    tail_fields = []
    if suffix_start < n:
        tail_fields = f[suffix_start:n-1]  # excludes filed_date; includes flags? no flags is at suffix_start
        # If suffix_start points to flags, tail_fields includes flags; remove it.
        if flags_raw is not None and tail_fields and tail_fields[-1] == flags_raw:
            tail_fields = tail_fields[:-1]
    else:
        # no flags found; take last ~6 as tail candidate
        tail_fields = f[max(prefix_end, n-8):n-1]

    # Separate bools/floats from tail, but keep them named generically
    bool_tail: Dict[str, Optional[bool]] = {}
    float_tail: Dict[str, Optional[float]] = {}

    bool_i = 0
    float_i = 0
    for t in tail_fields[-6:]:  # only inspect last 6 tail candidates
        b = parse_bool(t)
        if b is not None:
            bool_tail[f"bool_tail_{bool_i}"] = b
            bool_i += 1
        else:
            val = parse_float(t)
            if val is not None:
                float_tail[f"float_tail_{float_i}"] = val
                float_i += 1

    # Parse numeric middle
    numeric_middle = []
    for t in middle:
        numeric_middle.append(parse_float(t))

    return ParsedRow(
        label_raw=label_raw,
        label=label,
        issuer=issuer,
        form=form,
        report_date=report_date,
        fiscal_year=fiscal_year,
        filed_date=filed_date,
        cik=cik,
        accession=accession,
        company_name=company_name,
        path=path,
        flags_raw=flags_raw,
        flags=flags,
        bool_tail=bool_tail,
        float_tail=float_tail,
        numeric_middle=numeric_middle,
        n_fields=n,
    )


def iter_rows_csv(path: str) -> Iterable[ParsedRow]:
    """
    Stream parse CSV lines.
    This expects no embedded commas inside quotes for the majority of fields.
    Your sample looks safe (rarely quoted company names).
    If you have heavy quoting, you can replace with csv.reader.
    """
    import csv
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        for fields in reader:
            if not fields:
                continue
            yield parse_row_fields(fields)


# -----------------------------
# Regime + trust scoring
# -----------------------------

def has_flag(flags: List[str], prefix: str) -> bool:
    return any(f == prefix or f.startswith(prefix) for f in flags)

def compute_regime(flags: List[str], form: Optional[str], report_date: Optional[pd.Timestamp]) -> str:
    """
    A/B/C/D regimes:
      A: SGML/HTML fallback heavy, MissingH
      B: HTMLTables_Targeted (income statement partial)
      C: InlineXBRL but mismatch/salvage
      D: InlineXBRL clean-ish
    """
    if has_flag(flags, "InlineXBRL"):
        # InlineXBRL exists; decide C vs D based on sanity/salvage
        if has_flag(flags, "SANITY_FAIL") or has_flag(flags, "SALVAGE") or has_flag(flags, "DerivedE") or has_flag(flags, "DerivedL"):
            return "C"
        return "D"

    # Non InlineXBRL
    if has_flag(flags, "HTMLTables_Targeted"):
        return "B"
    if has_flag(flags, "SGML_") or has_flag(flags, "LabelFallback") or has_flag(flags, "MissingH"):
        return "A"
    # If unknown, guess based on date
    if report_date is not None and report_date.year >= 2018:
        return "B"
    return "A"


def compute_trust_score(flags: List[str], regime: str) -> float:
    """
    Produce a [0,1] trust score used as sample_weight.
    """
    score = 1.0

    # Base by regime
    if regime == "D":
        score *= 1.0
    elif regime == "C":
        score *= 0.6
    elif regime == "B":
        score *= 0.35
    else:  # A
        score *= 0.15

    # Penalties for known bad flags
    penalties = [
        ("MissingH", 0.15),
        ("LabelFallback", 0.10),
        ("SANITY_FAIL", 0.25),
        ("SALVAGE", 0.20),
        ("DerivedE", 0.15),
        ("DerivedL", 0.15),
        ("SGML_", 0.10),
    ]
    for flag_prefix, p in penalties:
        if has_flag(flags, flag_prefix):
            score *= (1.0 - p)

    # Floor/ceiling
    score = max(0.02, min(1.0, score))
    return float(score)


def eligibility_gate(flags: List[str], regime: str) -> bool:
    """
    Strict gate for solvency/risk model:
      InlineXBRL True
      MissingH False
      SANITY_FAIL False
      DerivedE/L False
      SALVAGE False
    """
    if regime not in {"C", "D"}:
        return False
    if not has_flag(flags, "InlineXBRL"):
        return False
    if has_flag(flags, "MissingH"):
        return False
    if has_flag(flags, "SANITY_FAIL"):
        return False
    if has_flag(flags, "SALVAGE"):
        return False
    if has_flag(flags, "DerivedE") or has_flag(flags, "DerivedL"):
        return False
    return True


# -----------------------------
# Feature building
# -----------------------------

FIN_KEYWORDS = {
    # If you later add a real schema mapping, you can map numeric_middle indices to these.
    # For now, we keep them generic.
}

def build_dataframe(rows: Iterable[ParsedRow], max_middle: int = 32) -> pd.DataFrame:
    """
    Convert parsed rows into a model-ready dataframe.
    - numeric_middle padded/truncated to max_middle
    - flags expanded into sparse-ish indicator columns (top N flags)
    """
    records = []
    all_flags = []

    rows_list = list(rows)
    for r in rows_list:
        all_flags.extend(r.flags)

    # Keep top flags to avoid exploding dimensionality
    flag_counts = pd.Series(all_flags).value_counts()
    top_flags = list(flag_counts.head(80).index)

    for r in rows_list:
        regime = compute_regime(r.flags, r.form, r.report_date)
        trust = compute_trust_score(r.flags, regime)
        eligible = eligibility_gate(r.flags, regime)

        rec = {
            "label_raw": r.label_raw,
            "y": r.label,
            "issuer": r.issuer,
            "company_name": r.company_name,
            "cik": r.cik,
            "accession": r.accession,
            "form": r.form,
            "report_date": r.report_date,
            "filed_date": r.filed_date,
            "fiscal_year": r.fiscal_year,
            "path": r.path,
            "flags_raw": r.flags_raw,
            "regime": regime,
            "trust_score": trust,
            "eligible_solvency": bool(eligible),
            "n_fields": r.n_fields,
        }

        # Tail bool/float
        for k, v in r.bool_tail.items():
            rec[k] = v
        for k, v in r.float_tail.items():
            rec[k] = v

        # Numeric middle padded
        nm = r.numeric_middle[:max_middle]
        nm = nm + [None] * (max_middle - len(nm))
        for i, v in enumerate(nm):
            rec[f"mid_{i:02d}"] = v

        # Flag indicators (top flags only)
        flag_set = set(r.flags)
        for fl in top_flags:
            rec[f"flag__{fl}"] = int(fl in flag_set)

        records.append(rec)

    df = pd.DataFrame.from_records(records)
    # Drop unknown labels
    df = df[df["y"].isin([0, 1])].reset_index(drop=True)
    return df


def time_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-aware split using filed_date (fallback to report_date).
    Prevents leakage across reporting eras.
    """
    d = df.copy()
    d["split_date"] = d["filed_date"].fillna(d["report_date"])
    # If still null, push to early
    d["split_date"] = d["split_date"].fillna(pd.Timestamp("1900-01-01"))
    d = d.sort_values("split_date").reset_index(drop=True)
    n = len(d)
    cut = int((1.0 - test_size) * n)
    train = d.iloc[:cut].reset_index(drop=True)
    test = d.iloc[cut:].reset_index(drop=True)
    return train, test


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Create preprocessing: numeric impute+scale, categorical onehot, keep flag indicators as numeric.
    """
    # Targets and meta to exclude from features
    exclude = {
        "y", "label_raw", "flags_raw", "path", "company_name", "accession",
        "report_date", "filed_date", "split_date"
    }

    # Categorical
    cat_cols = ["issuer", "form", "regime"]

    # Numeric candidates: everything else not excluded
    feature_cols = [c for c in df.columns if c not in exclude]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),  # sparse-friendly
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return pre, num_cols, cat_cols


# -----------------------------
# Models
# -----------------------------

def train_and_eval_classifier(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    outdir: str,
    sample_weight_col: Optional[str] = None,
    model_type: str = "hgb",
) -> Dict:
    """
    Train a classifier and write:
      - model.joblib
      - metrics.json
      - predictions.csv
    """
    os.makedirs(outdir, exist_ok=True)

    pre, num_cols, cat_cols = build_preprocessor(train_df)

    if model_type == "hgb":
        clf = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=400,
            l2_regularization=1e-3,
        )
    else:
        # Strong baseline: calibrated-ish logistic
        clf = LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
        )

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    X_train = train_df
    y_train = train_df["y"].astype(int).values
    X_test = test_df
    y_test = test_df["y"].astype(int).values

    sw = None
    if sample_weight_col is not None and sample_weight_col in train_df.columns:
        sw = train_df[sample_weight_col].astype(float).values

    pipe.fit(X_train, y_train, clf__sample_weight=sw)

    # Predict
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # Metrics
    roc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else float("nan")
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    cm = confusion_matrix(y_test, pred).tolist()

    # PR AUC
    pr_prec, pr_rec, _ = precision_recall_curve(y_test, proba)
    pr_auc = auc(pr_rec, pr_prec)

    metrics = {
        "model_name": model_name,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "confusion_matrix": cm,
        "feature_note": "Numeric middle fields are generic; flags and regime drive data quality and separations.",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # Save artifacts
    joblib.dump(pipe, os.path.join(outdir, f"{model_name}.joblib"))
    with open(os.path.join(outdir, f"{model_name}_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    pred_out = test_df.copy()
    pred_out[f"{model_name}__proba_risky"] = proba
    pred_out[f"{model_name}__pred_risky"] = pred
    pred_out.to_csv(os.path.join(outdir, f"{model_name}_predictions.csv"), index=False)

    return metrics


def write_markdown_report(outdir: str, sections: Dict[str, Dict]) -> None:
    lines = []
    lines.append("# Regime-aware SEC Risk Pipeline Report\n")
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z\n")

    for name, m in sections.items():
        lines.append(f"## {name}\n")
        lines.append(f"- Train rows: **{m['n_train']}**\n")
        lines.append(f"- Test rows: **{m['n_test']}**\n")
        lines.append(f"- ROC AUC: **{m['roc_auc']:.4f}**\n")
        lines.append(f"- PR AUC: **{m['pr_auc']:.4f}**\n")
        lines.append(f"- Accuracy: **{m['accuracy']:.4f}**\n")
        lines.append(f"- Precision: **{m['precision']:.4f}**\n")
        lines.append(f"- Recall: **{m['recall']:.4f}**\n")
        lines.append(f"- Confusion matrix: `{m['confusion_matrix']}`\n")
        lines.append("\n")

    with open(os.path.join(outdir, "REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to headerless CSV")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--max-middle", type=int, default=32, help="How many numeric middle columns to keep")
    ap.add_argument("--test-size", type=float, default=0.2, help="Time-based test split fraction")
    ap.add_argument("--save-parquet", action="store_true", help="Also save cleaned dataset as parquet")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Parse
    rows = list(iter_rows_csv(args.input))
    df = build_dataframe(rows, max_middle=args.max_middle)

    # Basic diagnostics
    diag = {
        "rows_total_labeled": int(len(df)),
        "regime_counts": df["regime"].value_counts().to_dict(),
        "eligible_solvency_count": int(df["eligible_solvency"].sum()),
        "eligible_solvency_rate": float(df["eligible_solvency"].mean() if len(df) else 0.0),
        "label_balance": df["y"].value_counts().to_dict(),
    }
    with open(os.path.join(args.outdir, "DATA_DIAGNOSTICS.json"), "w", encoding="utf-8") as fh:
        json.dump(diag, fh, indent=2)

    # Save cleaned dataset
    df.to_csv(os.path.join(args.outdir, "cleaned_dataset.csv"), index=False)
    if args.save_parquet:
        df.to_parquet(os.path.join(args.outdir, "cleaned_dataset.parquet"), index=False)

    # 2) Build dataset slices
    # Solvency/risk model: STRICT eligibility gate (mostly D, some C filtered out by gate)
    solv_df = df[df["eligible_solvency"] == True].copy()
    # Growth/ops model: allow B + D (and optionally C) but exclude A
    growth_df = df[df["regime"].isin(["B", "C", "D"])].copy()

    # 3) Time splits
    solv_train, solv_test = time_split(solv_df, test_size=args.test_size) if len(solv_df) >= 50 else (solv_df, solv_df.iloc[0:0])
    growth_train, growth_test = time_split(growth_df, test_size=args.test_size) if len(growth_df) >= 50 else (growth_df, growth_df.iloc[0:0])

    # 4) Train models
    metrics = {}

    if len(solv_test) > 0 and len(solv_train) > 0:
        metrics["SolvencyRiskModel"] = train_and_eval_classifier(
            solv_train,
            solv_test,
            model_name="solvency_risk_model",
            outdir=args.outdir,
            sample_weight_col="trust_score",
            model_type="hgb",
        )
    else:
        metrics["SolvencyRiskModel"] = {
            "note": "Not enough eligible solvency rows to split/train. Increase data or relax gate carefully."
        }

    if len(growth_test) > 0 and len(growth_train) > 0:
        metrics["GrowthOpsModel"] = train_and_eval_classifier(
            growth_train,
            growth_test,
            model_name="growth_ops_model",
            outdir=args.outdir,
            sample_weight_col="trust_score",
            model_type="hgb",
        )
    else:
        metrics["GrowthOpsModel"] = {
            "note": "Not enough growth rows to split/train."
        }

    with open(os.path.join(args.outdir, "ALL_METRICS.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # 5) Write markdown report if both exist
    if "roc_auc" in metrics.get("SolvencyRiskModel", {}) or "roc_auc" in metrics.get("GrowthOpsModel", {}):
        valid_sections = {}
        if "roc_auc" in metrics.get("SolvencyRiskModel", {}):
            valid_sections["SolvencyRiskModel"] = metrics["SolvencyRiskModel"]
        if "roc_auc" in metrics.get("GrowthOpsModel", {}):
            valid_sections["GrowthOpsModel"] = metrics["GrowthOpsModel"]
        if valid_sections:
            write_markdown_report(args.outdir, valid_sections)

    print("Done.")
    print(f"Outputs written to: {args.outdir}")
    print("Key files:")
    print(" - cleaned_dataset.csv")
    print(" - DATA_DIAGNOSTICS.json")
    print(" - solvency_risk_model.joblib (if trained)")
    print(" - growth_ops_model.joblib (if trained)")
    print(" - *_metrics.json and *_predictions.csv")


if __name__ == "__main__":
    main()
