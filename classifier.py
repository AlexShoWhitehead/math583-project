#!/usr/bin/env python3
"""
company_risk_proxy_model.py

Trains calibrated probability models for:
  - Multi-class: healthy / risky / failed
  - Binary: nonhealthy = risky or failed

This is the correct approach when you do NOT have learnable default transitions
(i.e., failed companies have no pre-fail history under the same company_id).

Outputs calibrated probabilities + confidence-based unknown flags.

Train:
  python company_risk_proxy_model.py \
    --train_csv per_company_quarter_financials.csv \
    --label_col company_bucket \
    --company_id_col company_folder \
    --add_lags \
    --auto_reject_coverage 0.85 \
    --model_out risk_proxy_model.joblib

Predict:
  python company_risk_proxy_model.py \
    --model_in risk_proxy_model.joblib \
    --predict_csv healthy_company_quant_testset.csv \
    --pred_out preds.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    _HAS_HGB = True
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier
    _HAS_HGB = False

import joblib


# -----------------------------
# Column detection / time index
# -----------------------------

@dataclass
class CanonicalColumns:
    net_income: Optional[str]
    revenue: Optional[str]
    assets: Optional[str]
    liabilities: Optional[str]
    equity: Optional[str]
    hqla: Optional[str]


def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_columns(df: pd.DataFrame) -> CanonicalColumns:
    net_income = _first_present(df, ["Pi_Net_Income", "Eq1_Income_Pi_USDm", "Eq1_Income_Pi", "net_income", "NetIncome"])
    revenue = _first_present(df, ["R_Total_Revenue", "Eq1_Income_R_USDm", "Eq1_Income_R", "revenue", "Revenue"])
    assets = _first_present(df, ["A_Total_Assets", "Eq2_BS_A_USDm", "Eq2_BS_A", "assets", "TotalAssets"])
    liabilities = _first_present(df, ["L_Total_Liabilities", "Eq2_BS_L_Derived_USDm", "Eq2_BS_L_USDm", "Eq2_BS_L", "liabilities"])
    equity = _first_present(df, ["E_Total_Equity", "Eq2_BS_E_USDm", "Eq4_Capital_E_USDm", "Eq2_BS_E", "equity"])
    hqla = _first_present(df, ["H_HQLA_Proxy", "Eq3_Liquidity_H_USDm", "Eq3_Liquidity_H", "hqla", "HQLA"])
    return CanonicalColumns(net_income, revenue, assets, liabilities, equity, hqla)


def compute_hqla_fallback(df: pd.DataFrame) -> pd.Series:
    cash_col = _first_present(df, ["Cash_CashEquiv", "CashAndCashEquivalents", "cash", "Cash"])
    mkt_col = _first_present(df, ["Marketable_Securities", "MarketableSecurities", "marketable_securities"])
    if cash_col is None and mkt_col is None:
        return pd.Series([np.nan] * len(df), index=df.index)
    cash = pd.to_numeric(df[cash_col], errors="coerce") if cash_col else 0.0
    mkt = pd.to_numeric(df[mkt_col], errors="coerce") if mkt_col else 0.0
    return cash + mkt


def infer_time_index(df: pd.DataFrame) -> Optional[pd.Series]:
    if "q_year" in df.columns and "q_num" in df.columns:
        y = pd.to_numeric(df["q_year"], errors="coerce")
        q = pd.to_numeric(df["q_num"], errors="coerce")
        if y.notna().any() and q.notna().any():
            return (y * 4 + q).astype("Int64")
    if "period_of_report" in df.columns:
        dt = pd.to_datetime(df["period_of_report"], errors="coerce")
        if dt.notna().any():
            return (dt.dt.year * 12 + dt.dt.month).astype("Int64")
    return None


# -----------------------------
# Features (same family, lags optional)
# -----------------------------

def make_features(df_raw: pd.DataFrame, company_id_col: Optional[str], add_lags: bool, clip: bool = True) -> pd.DataFrame:
    df = df_raw.copy()
    cols = detect_columns(df)

    missing = [k for k, v in cols.__dict__.items() if k in ("net_income","revenue","assets","liabilities","equity") and v is None]
    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {missing}")

    assets = pd.to_numeric(df[cols.assets], errors="coerce")
    liabilities = pd.to_numeric(df[cols.liabilities], errors="coerce")
    equity = pd.to_numeric(df[cols.equity], errors="coerce")
    revenue = pd.to_numeric(df[cols.revenue], errors="coerce")
    net_income = pd.to_numeric(df[cols.net_income], errors="coerce")

    if cols.hqla and cols.hqla in df.columns:
        hqla = pd.to_numeric(df[cols.hqla], errors="coerce")
    else:
        hqla = compute_hqla_fallback(df)

    assets_pos = assets.where(assets > 0)
    revenue_nonzero = revenue.where(revenue != 0)

    net_margin = net_income / revenue_nonzero
    bs_imbalance_ratio = (assets - liabilities - equity) / assets_pos
    liquidity_ratio = hqla / assets_pos
    capital_ratio = equity / assets_pos

    leverage_ratio = liabilities / assets_pos
    abs_bs_imbalance_ratio = bs_imbalance_ratio.abs()

    X = pd.DataFrame({
        "net_margin": net_margin,
        "liquidity_ratio": liquidity_ratio,
        "capital_ratio": capital_ratio,
        "bs_imbalance_ratio": bs_imbalance_ratio,
        "abs_bs_imbalance_ratio": abs_bs_imbalance_ratio,
        "leverage_ratio": leverage_ratio,
    }, index=df.index).replace([np.inf, -np.inf], np.nan)

    if clip:
        X["net_margin"] = X["net_margin"].clip(-1.0, 1.0)
        X["bs_imbalance_ratio"] = X["bs_imbalance_ratio"].clip(-2.0, 2.0)
        X["abs_bs_imbalance_ratio"] = X["abs_bs_imbalance_ratio"].clip(0.0, 2.0)
        X["liquidity_ratio"] = X["liquidity_ratio"].clip(-1.0, 3.0)
        X["capital_ratio"] = X["capital_ratio"].clip(-2.0, 2.0)
        X["leverage_ratio"] = X["leverage_ratio"].clip(0.0, 5.0)

    if add_lags and company_id_col and company_id_col in df.columns:
        t = infer_time_index(df)
        if t is not None:
            tmp = X.copy()
            tmp["_company"] = df[company_id_col].astype(str).values
            tmp["_t"] = t.values
            tmp = tmp.sort_values(["_company", "_t"], kind="mergesort")

            base_cols = ["net_margin", "liquidity_ratio", "capital_ratio", "abs_bs_imbalance_ratio", "leverage_ratio"]
            for c in base_cols:
                tmp[f"{c}_lag1"] = tmp.groupby("_company", sort=False)[c].shift(1)
                tmp[f"{c}_delta1"] = tmp[c] - tmp[f"{c}_lag1"]

            tmp = tmp.drop(columns=["_company", "_t"])
            X = tmp.reindex(df.index)
        else:
            print("[warn] add_lags=True but no time index found; skipping lags.", file=sys.stderr)

    return X


# -----------------------------
# Labels
# -----------------------------

def load_label_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    with open(path, "r") as f:
        m = json.load(f)
    if not isinstance(m, dict):
        raise ValueError("--label_map_json must be a JSON object mapping old->new labels.")
    return {str(k): str(v) for k, v in m.items()}


def normalize_labels(y: pd.Series, label_map: Dict[str, str]) -> pd.Series:
    y = y.astype(str)
    if label_map:
        y = y.map(lambda v: label_map.get(v, v))
    y = y.str.strip().str.lower()
    y = y.replace({"almost failed": "failed", "near failed": "failed", "distressed": "failed"})
    return y


# -----------------------------
# Calibration: FrozenGroupKFold
# -----------------------------

class FrozenGroupKFold:
    def __init__(self, groups, n_splits=3):
        self.groups = np.asarray(groups)
        self.n_splits = int(n_splits)
        self._gkf = GroupKFold(n_splits=self.n_splits)

    def split(self, X, y=None, groups=None):
        return self._gkf.split(X, y, groups=self.groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def make_base_model(random_state: int = 42):
    if _HAS_HGB:
        return HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=600,
            random_state=random_state,
        )
    return GradientBoostingClassifier(random_state=random_state)


def fit_calibrated_groupcv(X, y, groups, random_state: int, n_splits: int = 3):
    base = make_base_model(random_state=random_state)
    cv = FrozenGroupKFold(groups=groups, n_splits=n_splits)
    cal = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
    sw = compute_sample_weight(class_weight="balanced", y=y)
    cal.fit(X, y, sample_weight=sw)
    return cal


def split_companies_3way(X: pd.DataFrame, groups: np.ndarray, test_size: float, calib_size: float, random_state: int):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    traincal_idx, test_idx = next(gss1.split(X, groups=groups))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=calib_size, random_state=random_state + 1)
    train_idx2, calib_idx2 = next(gss2.split(X.iloc[traincal_idx], groups=groups[traincal_idx]))

    train_idx = traincal_idx[train_idx2]
    calib_idx = traincal_idx[calib_idx2]
    return train_idx, calib_idx, test_idx


def auto_choose_reject_threshold(confidence: np.ndarray, target_coverage: float) -> float:
    target_coverage = min(max(float(target_coverage), 0.05), 0.99)
    return float(np.quantile(confidence, 1.0 - target_coverage))


# -----------------------------
# Train / Predict
# -----------------------------

def train(train_csv: str, label_col: str, company_id_col: str, label_map_json: Optional[str],
          add_lags: bool, test_size: float, calib_size: float, random_state: int,
          auto_reject_coverage: Optional[float], reject_below: Optional[float],
          model_out: Optional[str]) -> Dict[str, Any]:

    df = pd.read_csv(train_csv)
    if label_col not in df.columns:
        raise ValueError(f"label_col '{label_col}' missing.")
    if company_id_col not in df.columns:
        raise ValueError(f"company_id_col '{company_id_col}' missing.")

    label_map = load_label_map(label_map_json)
    y = normalize_labels(df[label_col], label_map)

    # Multi-class target
    y_mc = y.copy()

    # Binary target: nonhealthy
    y_bin = (y_mc != "healthy").astype(int)

    X = make_features(df, company_id_col=company_id_col, add_lags=add_lags, clip=True)
    groups = df[company_id_col].astype(str).values

    print(f"[info] Rows: {len(X)}", file=sys.stderr)
    print("[info] Feature NaN rates:\n" + X.isna().mean().sort_values(ascending=False).to_string() + "\n", file=sys.stderr)
    print("[info] Label distribution:\n" + y_mc.value_counts().to_string() + "\n", file=sys.stderr)

    tr_idx, cal_idx, te_idx = split_companies_3way(X, groups, test_size, calib_size, random_state)

    # Train+Cal for calibration CV (test untouched)
    traincal_idx = np.concatenate([tr_idx, cal_idx])
    X_traincal = X.iloc[traincal_idx]
    y_mc_traincal = y_mc.iloc[traincal_idx]
    y_bin_traincal = y_bin.iloc[traincal_idx].values
    groups_traincal = groups[traincal_idx]

    X_test = X.iloc[te_idx]
    y_mc_test = y_mc.iloc[te_idx]
    y_bin_test = y_bin.iloc[te_idx].values

    imputer = SimpleImputer(strategy="median", add_indicator=True)
    X_traincal_i = imputer.fit_transform(X_traincal)
    X_test_i = imputer.transform(X_test)

    # Calibrated models
    mc_model = fit_calibrated_groupcv(X_traincal_i, y_mc_traincal.values, groups_traincal, random_state=random_state, n_splits=3)
    bin_model = fit_calibrated_groupcv(X_traincal_i, y_bin_traincal, groups_traincal, random_state=random_state + 7, n_splits=3)

    # Evaluate multi-class
    pred_mc = mc_model.predict(X_test_i)
    print("\n[evaluation] Multi-class on TEST companies:", file=sys.stderr)
    print("[test] accuracy:", round(accuracy_score(y_mc_test, pred_mc), 4), file=sys.stderr)
    print("[test] macro_f1 :", round(f1_score(y_mc_test, pred_mc, average="macro"), 4), file=sys.stderr)
    print("\n[test] classification_report:\n" + classification_report(y_mc_test, pred_mc, digits=4), file=sys.stderr)
    print("[test] confusion_matrix:\n" + str(confusion_matrix(y_mc_test, pred_mc)), file=sys.stderr)

    # Confidence / unknown threshold from multi-class probs
    proba_mc = mc_model.predict_proba(X_test_i)
    conf = proba_mc.max(axis=1)

    chosen_reject = reject_below
    if chosen_reject is None and auto_reject_coverage is not None:
        chosen_reject = auto_choose_reject_threshold(conf, float(auto_reject_coverage))
        print(f"\n[info] auto reject_below set to {chosen_reject:.4f} to keep ~{auto_reject_coverage:.2f} coverage.", file=sys.stderr)

    bundle = {
        "imputer": imputer,
        "mc_model": mc_model,
        "bin_model": bin_model,
        "meta": {
            "label_col": label_col,
            "company_id_col": company_id_col,
            "add_lags": bool(add_lags),
            "reject_below": chosen_reject,
            "classes_mc": list(getattr(mc_model, "classes_", [])),
        }
    }

    if model_out:
        joblib.dump(bundle, model_out)
        print(f"\n[info] Saved model bundle to: {model_out}", file=sys.stderr)

    return bundle


def predict(model_in: str, predict_csv: str, pred_out: Optional[str], reject_below: Optional[float]) -> pd.DataFrame:
    bundle = joblib.load(model_in)
    imputer = bundle["imputer"]
    mc_model = bundle["mc_model"]
    bin_model = bundle["bin_model"]
    meta = bundle.get("meta", {})

    df = pd.read_csv(predict_csv)
    company_id_col = meta.get("company_id_col", None)
    add_lags = bool(meta.get("add_lags", True))

    X = make_features(df, company_id_col=company_id_col if company_id_col in df.columns else None,
                      add_lags=add_lags, clip=True)
    X_i = imputer.transform(X)

    proba_mc = mc_model.predict_proba(X_i)
    classes = mc_model.classes_
    pred = classes[np.argmax(proba_mc, axis=1)]
    confidence = proba_mc.max(axis=1)

    p_nonhealthy = bin_model.predict_proba(X_i)[:, 1]

    use_reject = reject_below if reject_below is not None else meta.get("reject_below", None)
    flag_unknown = (confidence < float(use_reject)).astype(int) if use_reject is not None else np.zeros(len(df), dtype=int)

    out = df.copy()
    out["predicted_bucket"] = pred
    out["confidence"] = confidence
    out["flag_unknown"] = flag_unknown
    out["p_nonhealthy"] = p_nonhealthy

    # also export class probs
    for i, c in enumerate(classes):
        out[f"p_{c}"] = proba_mc[:, i]

    if pred_out:
        out.to_csv(pred_out, index=False)
        print(f"[info] Wrote predictions to: {pred_out}", file=sys.stderr)
    else:
        print(out.head(20).to_string(index=False))

    return out


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, default=None)
    p.add_argument("--predict_csv", type=str, default=None)

    p.add_argument("--label_col", type=str, default="company_bucket")
    p.add_argument("--company_id_col", type=str, default="company_folder")
    p.add_argument("--label_map_json", type=str, default=None)

    p.add_argument("--add_lags", action="store_true")
    p.add_argument("--test_size", type=float, default=0.25)
    p.add_argument("--calib_size", type=float, default=0.25)

    p.add_argument("--auto_reject_coverage", type=float, default=None)
    p.add_argument("--reject_below", type=float, default=None)

    p.add_argument("--model_out", type=str, default=None)
    p.add_argument("--model_in", type=str, default=None)
    p.add_argument("--pred_out", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.train_csv:
        train(
            train_csv=args.train_csv,
            label_col=args.label_col,
            company_id_col=args.company_id_col,
            label_map_json=args.label_map_json,
            add_lags=args.add_lags,
            test_size=args.test_size,
            calib_size=args.calib_size,
            random_state=42,
            auto_reject_coverage=args.auto_reject_coverage,
            reject_below=args.reject_below,
            model_out=args.model_out,
        )

    if args.predict_csv:
        if not args.model_in:
            raise ValueError("Provide --model_in for prediction.")
        predict(
            model_in=args.model_in,
            predict_csv=args.predict_csv,
            pred_out=args.pred_out,
            reject_below=args.reject_below,
        )

    if (not args.train_csv) and (not args.predict_csv):
        raise ValueError("Nothing to do. Provide --train_csv and/or --predict_csv.")


if __name__ == "__main__":
    main()
