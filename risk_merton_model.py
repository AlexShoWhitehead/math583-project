"""
run_merton_pipeline.py
Runs an end-to-end Merton-model calibration for your cleaned_dataset.csv using yfinance.

Outputs:
 - merton_results.csv  (one row per firm-year with computed A, sigma_A, DD, PD, classification)
 - merton_debug.csv    (intermediate merged market/accounting fields for QA)

Requirements:
 - Python 3.9+
 - pip install pandas numpy scipy yfinance tqdm
 - Input file: ./cleaned_dataset.csv (the file you uploaded)
"""

import math
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import least_squares
import yfinance as yf
from tqdm import tqdm

# ---------------- USER PARAMETERS ----------------
INPUT_CSV = "cleaned_dataset.csv"  # your uploaded dataset filename
OUTPUT_CSV = "merton_results.csv"
DEBUG_CSV = "merton_debug.csv"

# Time horizon for Merton (years)
T = 1.0

# Annual trading days (for converting daily vol -> annual)
TRADING_DAYS = 252

# How many past days of price history to download (yfinance 'period' string)
PRICE_LOOKBACK_DAYS = "1y"   # change to "2y" or "730d" for longer windows

# Classification thresholds (Distance-to-Default)
THRESHOLDS = {
    "healthy": {"dd_min": 2.0},
    "moderate": {"dd_min": 1.0, "dd_max": 2.0},
    "risky": {"dd_max": 1.0}
}

# --- TICKER_MAP (edit as needed) ---
TICKER_MAP = {
    "Cisco": "CSCO",
    "Micron Technology": "MU",
    "Apple": "AAPL",
    "QualComm": "QCOM",
    "IBM": "IBM",
    "Adobe": "ADBE",
    "AMD": "AMD",
    "Roku": "ROKU",
    "Twilio": "TWLO",
    "Uber": "UBER",
    "Zillow": "Z",
    "Snapchat": "SNAP",
    "Microsoft": "MSFT",
    "Block": "SQ",
    "Texas Instruments": "TXN",
    "Amazon": "AMZN",
    "Meta": "META",
    "Pinterest": "PINS",
    "PayPal": "PYPL",
    "Nvidia": "NVDA",
    "DocuSign": "DOCU",
    "MongoDB": "MDB",
    "Salesforce": "CRM",
    "Oracle": "ORCL",
    "Peloton": "PTON",
    "KLA Corp": "KLAC",
    "Cloudflare": "NET",
    "Datadog": "DDOG",
    "Zoom": "ZM",
    "Etsy": "ETSY"
}
# Lowercase map for robust issuer name matching
TICKER_MAP_LOWER = {k.strip().lower(): v for k, v in TICKER_MAP.items()}

def get_ticker_for_issuer(issuer_name):
    if issuer_name is None:
        return None
    return TICKER_MAP_LOWER.get(str(issuer_name).strip().lower())

# Column mapping for accounting fields in your CSV.
# Replace these names if your cleaned_dataset uses different names.
ACCOUNTING_MAP = {
    "total_assets": "mid_00",
    "total_liabilities": "mid_01",
    "total_equity": "mid_02",
    "cash_and_equiv": "mid_03",
    "short_term_debt": "mid_04",
    "long_term_debt": "mid_05",
}

# Name of column in cleaned_dataset.csv that contains the issuer company name to map to tickers
ISSUER_COL = "issuer"

# Column to identify firm-year (used to merge when multiple rows for same issuer)
FISCAL_YEAR_COL = "fiscal_year"
REPORT_DATE_COL = "report_date"

# Risk-free rate: use an external source ideally. For demonstration use a single value (update if you like)
# If you want a per-row risk-free you can add a column 'risk_free_rate' to your CSV.
DEFAULT_RISK_FREE = 0.05  # 5% as example; replace with current 1-year Treasury yield if available
# -------------------------------------------------

def safe_get(row, key):
    return row.get(key) if key in row.index else np.nan

def download_market_data(tickers):
    """
    Bulk-download price histories using yf.download for speed. Fallback to per-Ticker for missing tickers.
    Returns (price_dfs, info_map) where price_dfs[ticker] is a DataFrame and info_map[ticker] a dict.
    """
    price_dfs = {}
    info_map = {}

    if len(tickers) == 0:
        return price_dfs, info_map

    # Attempt bulk download first (faster). Handles multi-ticker output shape.
    try:
        bulk = yf.download(tickers, period=PRICE_LOOKBACK_DAYS, auto_adjust=True, threads=True, group_by='ticker', progress=False)
    except Exception as e:
        print("Bulk price download failed, falling back to per-ticker downloads:", e, file=sys.stderr)
        bulk = pd.DataFrame()

    for t in tqdm(tickers, desc="Downloading market data"):
        price_dfs[t] = pd.DataFrame()
        info_map[t] = {"marketCap": np.nan, "sharesOutstanding": np.nan}

        # If bulk returned multi-index columns, try to extract ticker slice
        try:
            if isinstance(bulk, pd.DataFrame) and hasattr(bulk.columns, "nlevels") and bulk.columns.nlevels > 1:
                if t in bulk.columns.get_level_values(0):
                    # select the subset of columns for this ticker
                    try:
                        price_dfs[t] = bulk[t].copy()
                    except Exception:
                        price_dfs[t] = pd.DataFrame()
            elif isinstance(bulk, pd.DataFrame) and t in bulk.columns:
                # single ticker case
                price_dfs[t] = bulk[[t]].copy()
        except Exception:
            # ignore bulk extraction errors; fallback to per-ticker
            price_dfs[t] = pd.DataFrame()

        # If no bulk data, fallback to per-ticker download and info
        if price_dfs[t] is None or price_dfs[t].empty:
            try:
                tk = yf.Ticker(t)
                hist = tk.history(period=PRICE_LOOKBACK_DAYS, auto_adjust=True)
                price_dfs[t] = hist
                # info access can be slow or incomplete; guard against errors
                try:
                    info = tk.info or {}
                except Exception:
                    info = {}
                info_map[t] = {
                    "marketCap": info.get("marketCap", np.nan),
                    "sharesOutstanding": info.get("sharesOutstanding", np.nan),
                }
            except Exception as e:
                print(f"Error downloading {t}: {e}", file=sys.stderr)
                price_dfs[t] = pd.DataFrame()
                info_map[t] = {"marketCap": np.nan, "sharesOutstanding": np.nan}
        else:
            # If we did get bulk prices but not info, try lightweight info fetch (best-effort)
            try:
                tk = yf.Ticker(t)
                try:
                    info = tk.info or {}
                except Exception:
                    info = {}
                info_map[t] = {
                    "marketCap": info.get("marketCap", np.nan),
                    "sharesOutstanding": info.get("sharesOutstanding", np.nan),
                }
            except Exception:
                info_map[t] = {"marketCap": np.nan, "sharesOutstanding": np.nan}

    return price_dfs, info_map

def compute_annualized_vol(price_df):
    """
    Given a price DataFrame with 'Close' or 'Adj Close', compute daily returns and annualized vol.
    Uses simple percentage returns (pct_change). Returns annualized standard deviation.
    """
    if price_df is None or price_df.empty:
        return np.nan
    # prefer 'Close' or 'Adj Close'
    col = None
    for c in ["Close", "Adj Close", "Adj_Close", "adjclose", "AdjClose"]:
        if c in price_df.columns:
            col = c
            break
    if col is None:
        numeric_cols = price_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return np.nan
        col = numeric_cols[0]
    prices = price_df[col].dropna()
    if prices.shape[0] < 10:
        return np.nan
    # use simple returns; documentable choice
    rets = prices.pct_change().dropna()
    daily_std = rets.std(ddof=1)
    annualized_vol = daily_std * math.sqrt(TRADING_DAYS)
    return annualized_vol

def merton_solve(E, sigma_E, D, r, T=1.0):
    """
    Solve for asset value A and asset volatility sigma_A given E, sigma_E, D (default point),
    and r, using the Merton equations:
      E = A*N(d1) - D*exp(-rT)*N(d2)
      sigma_E = (A/E) * N(d1) * sigma_A
    Returns (A, sigma_A) or (np.nan, np.nan) on failure.
    """
    eps = 1e-12
    if E <= eps or sigma_E <= eps or D <= eps:
        return (np.nan, np.nan)

    def residuals(x):
        A, sigma_A = x
        if A <= 0 or sigma_A <= 0:
            # penalize invalid domain
            return [1e6, 1e6]
        sqrtT = math.sqrt(T)
        d1 = (math.log(A / D) + (r + 0.5 * sigma_A**2) * T) / (sigma_A * sqrtT)
        d2 = d1 - sigma_A * sqrtT
        Nd1 = norm.cdf(d1)
        Nd2 = norm.cdf(d2)
        # Model-implied equity
        E_model = A * Nd1 - D * math.exp(-r * T) * Nd2
        # Model-implied sigma_E
        sigmaE_model = (A / E) * Nd1 * sigma_A
        return [E_model - E, sigmaE_model - sigma_E]

    # initial guess: A ≈ E + D ; sigma_A ≈ sigma_E * (E/A) (approx)
    A0 = max(E + D, 1.0)
    sigmaA0 = max((sigma_E * E) / max(A0 * 0.5, 1e-6), 1e-4)
    x0 = np.array([A0, sigmaA0])

    try:
        result = least_squares(
            residuals,
            x0,
            bounds=([1e-8, 1e-8], [np.inf, np.inf]),
            xtol=1e-10,
            ftol=1e-10,
            verbose=0,
            max_nfev=2000
        )
        if not result.success:
            # solver didn't converge -> return NaNs for downstream handling
            return (np.nan, np.nan)
        A_hat, sigmaA_hat = result.x
        if not np.isfinite(A_hat) or not np.isfinite(sigmaA_hat):
            return (np.nan, np.nan)
        return (A_hat, sigmaA_hat)
    except Exception:
        return (np.nan, np.nan)

def compute_dd_pd(A, sigma_A, DP, r, T=1.0):
    """
    Compute Distance-to-Default and PD = N(-DD) where DP is the default point.
    """
    if A <= 0 or sigma_A <= 0 or DP <= 0:
        return (np.nan, np.nan)
    sqrtT = math.sqrt(T)
    numerator = math.log(A / DP) + (r - 0.5 * sigma_A**2) * T
    DD = numerator / (sigma_A * sqrtT)
    PD = norm.cdf(-DD)
    return (DD, PD)

def classify_by_dd(dd):
    if not np.isfinite(dd):
        return "unknown"
    if dd > 2.0:
        return "healthy"
    elif dd > 1.0:
        return "moderate"
    else:
        return "risky"

def _ensure_accounting_cols(merged):
    """
    Populate canonical accounting columns from ACCOUNTING_MAP and coerce to numeric.
    """
    for key, colname in ACCOUNTING_MAP.items():
        if colname in merged.columns:
            merged[key] = pd.to_numeric(merged[colname], errors="coerce")
        else:
            merged[key] = np.nan

def main():
    # reproducibility / environment info
    print(f"pandas {pd.__version__}, numpy {np.__version__}, scipy {sys.modules['scipy'].__version__ if 'scipy' in sys.modules else 'unknown'}")

    # 1) load accounting dataset
    df = pd.read_csv(INPUT_CSV)
    df = df.copy()

    # Map tickers robustly (case-insensitive)
    if ISSUER_COL not in df.columns:
        print(f"ERROR: issuer column '{ISSUER_COL}' not found in {INPUT_CSV}.", file=sys.stderr)
        return

    df["ticker"] = df[ISSUER_COL].apply(get_ticker_for_issuer)
    missing_tickers = df[df["ticker"].isna()][ISSUER_COL].unique()
    if len(missing_tickers) > 0:
        print("Warning: missing ticker mapping for these issuers (add to TICKER_MAP if needed):")
        print(missing_tickers)

    tickers = sorted(df["ticker"].dropna().unique().tolist())
    price_dfs, info_map = download_market_data(tickers)

    # Compute market variables per ticker
    market_rows = []
    for t in tickers:
        hist = price_dfs.get(t, pd.DataFrame())
        annual_vol = compute_annualized_vol(hist)
        info = info_map.get(t, {})
        marketCap_info = info.get("marketCap", np.nan)
        shares_out = info.get("sharesOutstanding", np.nan)
        last_price = np.nan
        if hist is not None and not hist.empty:
            # prefer 'Close' or the first numeric column
            col = "Close" if "Close" in hist.columns else None
            if col is None:
                numeric_cols = hist.select_dtypes(include=[np.number]).columns
                col = numeric_cols[0] if len(numeric_cols) > 0 else None
            if col is not None:
                try:
                    last_price = float(hist[col].dropna().iloc[-1])
                except Exception:
                    last_price = np.nan
        computed_marketcap = np.nan
        if not pd.isna(last_price) and not pd.isna(shares_out):
            try:
                computed_marketcap = float(last_price) * float(shares_out)
            except Exception:
                computed_marketcap = np.nan
        market_rows.append({
            "ticker": t,
            "annualized_equity_vol": annual_vol,
            "marketCap_info": marketCap_info,
            "shares_outstanding_info": shares_out,
            "last_price": last_price,
            "computed_marketcap": computed_marketcap
        })
    market_df = pd.DataFrame(market_rows)

    # Merge market info into accounting df by ticker
    merged = df.merge(market_df, on="ticker", how="left")

    # If user included explicit risk-free rates per row, use them; otherwise use DEFAULT_RISK_FREE
    if "risk_free_rate" not in merged.columns:
        merged["risk_free_rate"] = DEFAULT_RISK_FREE
    else:
        merged["risk_free_rate"] = pd.to_numeric(merged["risk_free_rate"], errors="coerce").fillna(DEFAULT_RISK_FREE)

    # Populate canonical accounting fields and coerce to numeric
    _ensure_accounting_cols(merged)

    # Ensure canonical debt columns exist as floats (no _f suffix)
    merged["short_term_debt"] = pd.to_numeric(merged.get("short_term_debt", np.nan), errors="coerce")
    merged["long_term_debt"] = pd.to_numeric(merged.get("long_term_debt", np.nan), errors="coerce")
    merged["total_liabilities"] = pd.to_numeric(merged.get("total_liabilities", np.nan), errors="coerce")

    # Compute default point (DP): ST debt + 0.5 * LT debt when available; otherwise fallback to 0.5 * total_liabilities
    merged["default_point"] = np.nan
    mask_both = merged["short_term_debt"].notna() & merged["long_term_debt"].notna()
    merged.loc[mask_both, "default_point"] = merged.loc[mask_both, "short_term_debt"] + 0.5 * merged.loc[mask_both, "long_term_debt"]
    mask_fallback = merged["default_point"].isna() & merged["total_liabilities"].notna()
    merged.loc[mask_fallback, "default_point"] = 0.5 * merged.loc[mask_fallback, "total_liabilities"]

    # Compute market equity E: prefer marketCap_info, then computed_marketcap, else NaN
    merged["market_equity"] = merged["marketCap_info"]
    mask_missing_marketcap = merged["market_equity"].isna() & merged["computed_marketcap"].notna()
    merged.loc[mask_missing_marketcap, "market_equity"] = merged.loc[mask_missing_marketcap, "computed_marketcap"]

    # Equity volatility from market data
    merged["equity_volatility"] = merged["annualized_equity_vol"]

    # Prepare results list
    results = []
    for idx, row in merged.iterrows():
        E = row.get("market_equity", np.nan)
        sigma_E = row.get("equity_volatility", np.nan)
        DP = row.get("default_point", np.nan)
        r = row.get("risk_free_rate", DEFAULT_RISK_FREE) if not pd.isna(row.get("risk_free_rate", np.nan)) else DEFAULT_RISK_FREE

        # If missing necessary inputs, skip (record NaNs)
        if pd.isna(E) or pd.isna(sigma_E) or pd.isna(DP):
            A_hat, sigmaA_hat, DD, PD = (np.nan, np.nan, np.nan, np.nan)
        else:
            A_hat, sigmaA_hat = merton_solve(E=float(E), sigma_E=float(sigma_E), D=float(DP), r=float(r), T=T)
            if pd.isna(A_hat):
                # solver failed -> log for QA
                print(f"Solver failed for row {idx} issuer={row.get(ISSUER_COL)} ticker={row.get('ticker')} E={E} sigma_E={sigma_E} D={DP}", file=sys.stderr)
                DD, PD = (np.nan, np.nan)
            else:
                DD, PD = compute_dd_pd(A_hat, sigmaA_hat, DP=float(DP), r=float(r), T=T)

        classification = classify_by_dd(DD if not pd.isna(DD) else np.nan)

        res = {
            "row_index": idx,
            "issuer": row.get(ISSUER_COL),
            "ticker": row.get("ticker"),
            "fiscal_year": row.get(FISCAL_YEAR_COL),
            "report_date": row.get(REPORT_DATE_COL),
            "market_equity": E,
            "equity_volatility": sigma_E,
            "short_term_debt": row.get("short_term_debt"),
            "long_term_debt": row.get("long_term_debt"),
            "default_point": DP,
            "asset_value_A": A_hat,
            "asset_vol_sigmaA": sigmaA_hat,
            "DD": DD,
            "PD": PD,
            "classification": classification
        }
        results.append(res)

    results_df = pd.DataFrame(results).set_index("row_index")
    final = merged.join(results_df, how="left", rsuffix="_res")

    # Save outputs
    final.to_csv(DEBUG_CSV, index=False)
    cols_to_write = ["issuer", "ticker", "fiscal_year", "report_date", "market_equity", "equity_volatility",
                     "short_term_debt", "long_term_debt", "default_point", "asset_value_A", "asset_vol_sigmaA",
                     "DD", "PD", "classification"]
    # ensure columns exist before writing; fill missing with NaN
    for c in cols_to_write:
        if c not in final.columns:
            final[c] = np.nan
    final[cols_to_write].to_csv(OUTPUT_CSV, index=False)
    print("Done. Wrote:", OUTPUT_CSV, "and", DEBUG_CSV)

if __name__ == "__main__":
    main()
