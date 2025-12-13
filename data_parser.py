#!/usr/bin/env python3
"""
Robust EDGAR parser for mixed filing formats (v5.2).

What it does
------------
- Walks root_dir with subdirs: healthy / risky / failed
- Parses local EDGAR filings (10-K, 10-Q, 8-K) in .txt/.htm/.html/.xml
- Extracts core financial variables needed for your equation system:
    A = Total Assets
    L = Total Liabilities
    E = Total Equity
    R = Revenue
    Pi = Net Income
    Operating Income (optional)
    H = liquidity proxy = Cash + Marketable Securities (or Cash-only fallback)

- Produces:
    per_filing_financials.csv        (one row per filing file)
    per_company_year_financials.csv  (one row per company-year, prefer 10-K)

Major robustness features
-------------------------
- Supports BOTH:
    * Inline XBRL facts: <ix:nonFraction name="us-gaap:Assets" ...>...</ix:nonFraction>
    * Plain XBRL facts:  <us-gaap:Assets contextRef="...">...</us-gaap:Assets>

- Consolidated context preference:
    penalizes contextRef strings containing Axis/Member/Segment/etc.

- FIX (critical):
    us-gaap:LiabilitiesAndStockholdersEquity is NOT a liabilities concept.
    v5.2 removes it from L candidates to prevent BS mismatch blowups.

- Derivation logic:
    * If L missing but A and E exist: derive L = A - E
    * If E missing but A and L exist: derive E = A - L
    * If total liabilities missing but current and noncurrent exist: derive L = Lc + Lnc

- Balance sheet sanity check:
    Computes residual = A - L - E
    If residual too large, tries derivations before blanking.

Dependencies
------------
pip install pandas lxml beautifulsoup4

Usage
-----
python parse_edgar_equations_robust_v52.py --root_dir data

"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from typing import Dict, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup


# -----------------------------
# Configuration
# -----------------------------

COMPANY_BUCKETS = ["healthy", "risky", "failed"]
MAIN_FORMS = {"10-K", "10-Q", "8-K"}

CORE_FIELDS_INSTANT = [
    "A_Total_Assets",
    "L_Total_Liabilities",
    "E_Total_Equity",
    "Cash_CashEquiv",
    "Marketable_Securities",
]
CORE_FIELDS_DURATION = [
    "R_Total_Revenue",
    "Pi_Net_Income",
]

# IMPORTANT: v5.2 removes LiabilitiesAndStockholdersEquity from L candidates.
TAG_CANDIDATES = {
    "A_Total_Assets": ["us-gaap:Assets"],

    "L_Total_Liabilities": [
        "us-gaap:Liabilities",
        "us-gaap:LiabilitiesCurrent",
        "us-gaap:LiabilitiesNoncurrent",
    ],

    "E_Total_Equity": [
        "us-gaap:StockholdersEquity",
        "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "us-gaap:StockholdersEquityAttributableToParent",
        "us-gaap:PartnersCapital",  # sometimes used by partnerships
        "us-gaap:MembersEquity",    # sometimes used by LLC-type filers
    ],

    "R_Total_Revenue": [
        "us-gaap:Revenues",
        "us-gaap:SalesRevenueNet",
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    ],

    "Pi_Net_Income": [
        "us-gaap:NetIncomeLoss",
        "us-gaap:ProfitLoss",
        "us-gaap:NetIncomeLossAvailableToCommonStockholdersBasic",
    ],

    "Cash_CashEquiv": [
        "us-gaap:CashAndCashEquivalentsAtCarryingValue",
        "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],

    "Marketable_Securities": [
        "us-gaap:MarketableSecuritiesCurrent",
        "us-gaap:ShortTermInvestments",
        "us-gaap:AvailableForSaleSecuritiesCurrent",
    ],

    "Operating_Income": [
        "us-gaap:OperatingIncomeLoss",
        "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],

    # Diagnostic only (NOT used as liabilities!)
    "BS_Check_LplusE": [
        "us-gaap:LiabilitiesAndStockholdersEquity",
        "us-gaap:LiabilitiesAndPartnersCapital",
    ],
}

LABEL_SYNONYMS = {
    "A_Total_Assets": ["total assets", "assets, total"],
    "L_Total_Liabilities": ["total liabilities", "liabilities, total"],
    "E_Total_Equity": [
        "total stockholders' equity",
        "total shareholders' equity",
        "total equity",
        "stockholders' equity",
        "shareholders' equity",
        "members' equity",
        "partners' capital",
    ],
    "R_Total_Revenue": ["total revenue", "net sales", "revenues"],
    "Pi_Net_Income": ["net income", "net earnings", "net loss", "net (loss)"],
    "Operating_Income": ["operating income", "income from operations"],
    "Cash_CashEquiv": ["cash and cash equivalents", "cash & cash equivalents"],
    "Marketable_Securities": ["marketable securities", "short-term investments", "short term investments"],
}

BALANCE_SHEET_ANCHORS = [
    "consolidated balance sheets",
    "consolidated statements of financial position",
    "balance sheets",
    "statements of financial position",
]
INCOME_STMT_ANCHORS = [
    "consolidated statements of operations",
    "consolidated statements of income",
    "statements of operations",
    "statements of income",
    "income statements",
]

RE_EDGAR_FORM = re.compile(r"\bFORM\s+(10-K|10-Q|8-K)\b", re.IGNORECASE)
RE_HEADER_FORM = re.compile(r"CONFORMED\s+SUBMISSION\s+TYPE:\s*(.+)", re.IGNORECASE)
RE_HEADER_PERIOD = re.compile(r"CONFORMED\s+PERIOD\s+OF\s+REPORT:\s*(\d{8})", re.IGNORECASE)
RE_HEADER_COMPANY = re.compile(r"COMPANY\s+CONFORMED\s+NAME:\s*(.+)", re.IGNORECASE)
RE_HEADER_CIK = re.compile(r"CENTRAL\s+INDEX\s+KEY:\s*(\d+)", re.IGNORECASE)
RE_HEADER_ACCESSION = re.compile(r"ACCESSION\s+NUMBER:\s*([0-9\-]+)", re.IGNORECASE)

RE_SGML_DOC = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.IGNORECASE | re.DOTALL)
RE_SGML_TYPE = re.compile(r"<TYPE>\s*([A-Z0-9\-\.\_]+)\s*", re.IGNORECASE)
RE_SGML_SEQ = re.compile(r"<SEQUENCE>\s*([0-9]+)\s*", re.IGNORECASE)
RE_SGML_FILENAME = re.compile(r"<FILENAME>\s*([^\s<]+)\s*", re.IGNORECASE)
RE_SGML_TEXT = re.compile(r"<TEXT>(.*?)</TEXT>", re.IGNORECASE | re.DOTALL)

RE_XBRL_CONTEXT = re.compile(
    r"<xbrli:context[^>]*\bid=['\"]([^'\"]+)['\"][^>]*>(.*?)</xbrli:context>",
    re.IGNORECASE | re.DOTALL,
)
RE_XBRL_INSTANT = re.compile(r"<xbrli:instant>\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s*</xbrli:instant>", re.IGNORECASE)
RE_XBRL_START = re.compile(r"<xbrli:startDate>\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s*</xbrli:startDate>", re.IGNORECASE)
RE_XBRL_END = re.compile(r"<xbrli:endDate>\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s*</xbrli:endDate>", re.IGNORECASE)

RE_ATTR_CONTEXTREF = re.compile(r"\bcontextRef\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
RE_ATTR_SCALE = re.compile(r"\bscale\s*=\s*['\"](-?\d+)['\"]", re.IGNORECASE)
RE_NUMBER_TOKEN = re.compile(r"[\(\-]?\$?\d[\d,]*(?:\.\d+)?\)?")
RE_YEARISH = re.compile(r"^(19\d{2}|20\d{2}|2100)$")

# -----------------------------
# Context scoring
# -----------------------------

BAD_CTX_TOKENS = [
    "axis", "member", "segment", "segments",
    "product", "geography", "class", "classes",
    "statementclassofstock", "longtermdebt", "typeaxis",
    "_us-gaap_", "_srt_", "_dei_", "_ifrs_",
]

def context_penalty(context_ref: str) -> int:
    """Lower is better; penalize dimensional contexts."""
    if not context_ref:
        return 10_000
    s = context_ref.lower()
    pen = 0
    for tok in BAD_CTX_TOKENS:
        if tok in s:
            pen += 100
    # shorter IDs often correspond to consolidated contexts
    pen += max(0, len(s) - 14) // 6
    return pen


# -----------------------------
# Utilities
# -----------------------------

def safe_read_text(path: str, max_bytes: int = 30_000_000) -> Optional[str]:
    try:
        if os.path.getsize(path) > max_bytes:
            return None
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return None

def clean_numeric(text: str) -> Optional[float]:
    if text is None:
        return None
    s = str(text).strip().replace("\xa0", " ")
    if not s:
        return None
    m = RE_NUMBER_TOKEN.search(s.replace("&nbsp;", " "))
    if not m:
        return None
    tok = m.group(0).strip().replace("$", "")
    neg = tok.startswith("(") and tok.endswith(")")
    if neg:
        tok = tok[1:-1].strip()
    tok = tok.replace(",", "")
    try:
        v = float(tok)
        return -v if neg else v
    except Exception:
        return None

def iso_from_yyyymmdd(yyyymmdd: str) -> Optional[str]:
    if not yyyymmdd or not re.fullmatch(r"\d{8}", yyyymmdd):
        return None
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"

def parse_period_from_text_fallback(text: str) -> Optional[str]:
    # very light fallback, not meant to be perfect
    m = re.search(r"period ended\s+([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", text, re.IGNORECASE)
    if not m:
        return None
    month, day, year = m.group(1), int(m.group(2)), int(m.group(3))
    try:
        dt = datetime.strptime(f"{month} {day} {year}", "%B %d %Y")
        return dt.strftime("%Y%m%d")
    except Exception:
        return None

def detect_scale_factor(text: str) -> float:
    head = (text or "").lower()
    if "in thousands" in head:
        return 1_000.0
    if "in millions" in head:
        return 1_000_000.0
    return 1.0

def xbrl_scale_multiplier(attrs: str) -> float:
    m = RE_ATTR_SCALE.search(attrs or "")
    if not m:
        return 1.0
    try:
        sc = int(m.group(1))
        return float(10 ** sc)
    except Exception:
        return 1.0

def is_yearish_number(v: float) -> bool:
    if v is None:
        return False
    return RE_YEARISH.match(str(int(abs(v)))) is not None

def accept_money_value(field: str, v: float) -> bool:
    # prevent obvious garbage like "2012" becoming a money number
    if v is None:
        return False
    if is_yearish_number(v):
        return False
    # base magnitude screens (loose, but avoids tons of noise)
    if field in {"A_Total_Assets", "L_Total_Liabilities", "E_Total_Equity", "R_Total_Revenue"}:
        if abs(v) < 10_000:
            return False
    return True


# -----------------------------
# Identify filings & SGML documents
# -----------------------------

def extract_header_fields(content: str) -> Dict[str, str]:
    def grab(rx: re.Pattern) -> str:
        m = rx.search(content)
        return m.group(1).strip() if m else ""
    return {
        "company_name": grab(RE_HEADER_COMPANY),
        "cik": grab(RE_HEADER_CIK),
        "accession_number": grab(RE_HEADER_ACCESSION),
        "form_type_header": grab(RE_HEADER_FORM).upper(),
        "period_of_report": grab(RE_HEADER_PERIOD),
    }

def extract_sgml_documents(content: str) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    for m in RE_SGML_DOC.finditer(content):
        block = m.group(1)
        mt = RE_SGML_TYPE.search(block)
        ms = RE_SGML_SEQ.search(block)
        mf = RE_SGML_FILENAME.search(block)
        mx = RE_SGML_TEXT.search(block)
        docs.append({
            "type": (mt.group(1).upper().strip() if mt else ""),
            "sequence": (ms.group(1).strip() if ms else ""),
            "filename": (mf.group(1).strip() if mf else ""),
            "text": (mx.group(1) if mx else ""),
        })
    return docs

def detect_form_type(content: str) -> str:
    hdr = extract_header_fields(content)
    ft = (hdr.get("form_type_header") or "").strip().upper()
    if ft in MAIN_FORMS:
        return ft
    for d in extract_sgml_documents(content):
        if d["type"] in MAIN_FORMS:
            return d["type"]
    m = RE_EDGAR_FORM.search(content)
    return m.group(1).upper() if m else ""

def pick_primary_document(content: str, form_type: str) -> Tuple[str, str]:
    docs = extract_sgml_documents(content)
    if not docs:
        return content, "SingleDoc"
    candidates = [d for d in docs if d["type"] == form_type and d.get("text")]
    if not candidates:
        # as a fallback, prefer HTML documents if present
        html_docs = [d for d in docs if d.get("text") and ("<html" in (d["text"].lower()))]
        if html_docs:
            best = max(html_docs, key=lambda d: len(d["text"] or ""))
            return best["text"], "SGML_Primary_HTMLFallback"
        return content, "SGML_NoMatchingType"
    seq1 = [d for d in candidates if d.get("sequence") == "1"]
    if seq1:
        return seq1[0]["text"], "SGML_Primary_SEQ1"
    best = max(candidates, key=lambda d: len(d["text"] or ""))
    return best["text"], "SGML_Primary_LargestText"


# -----------------------------
# Inline + plain XBRL extraction
# -----------------------------

@dataclass
class XContext:
    kind: str  # "instant" or "duration"
    instant: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None

@dataclass
class XVal:
    value: float
    context_ref: str

def has_xbrl(content: str) -> bool:
    lo = content.lower()
    return ("<xbrli:context" in lo) and ("contextref" in lo)

def has_inline_xbrl(content: str) -> bool:
    lo = content.lower()
    return ("<ix:nonfraction" in lo) and ("<xbrli:context" in lo)

def parse_xbrl_contexts(x: str) -> Dict[str, XContext]:
    ctx: Dict[str, XContext] = {}
    for m in RE_XBRL_CONTEXT.finditer(x):
        cid = m.group(1).strip()
        body = m.group(2)

        inst = RE_XBRL_INSTANT.search(body)
        if inst:
            ctx[cid] = XContext(kind="instant", instant=inst.group(1))
            continue

        sd = RE_XBRL_START.search(body)
        ed = RE_XBRL_END.search(body)
        if sd and ed:
            ctx[cid] = XContext(kind="duration", start=sd.group(1), end=ed.group(1))
    return ctx

def re_ix_nonfraction(tag: str) -> re.Pattern:
    return re.compile(
        rf"<ix:nonFraction\b([^>]*)\bname\s*=\s*['\"]{re.escape(tag)}['\"]([^>]*)>(.*?)</ix:nonFraction>",
        re.IGNORECASE | re.DOTALL,
    )

def extract_ix_values(x: str, tag: str) -> List[XVal]:
    out: List[XVal] = []
    rx = re_ix_nonfraction(tag)
    for m in rx.finditer(x):
        attrs = ((m.group(1) or "") + " " + (m.group(2) or "")).strip()
        inner = re.sub(r"<[^>]+>", "", m.group(3) or "")
        v = clean_numeric(inner)
        if v is None:
            continue
        cr_m = RE_ATTR_CONTEXTREF.search(attrs)
        cr = cr_m.group(1).strip() if cr_m else ""
        mul = xbrl_scale_multiplier(attrs)
        out.append(XVal(float(v) * mul, cr))
    return out

RE_PLAIN_XBRL_TAG = re.compile(
    r"<(?P<tag>[a-zA-Z0-9\-\:]+)\b(?P<attrs>[^>]*)>(?P<val>[^<]+)</(?P=tag)>",
    re.IGNORECASE | re.DOTALL,
)

def extract_plain_xbrl_values(x: str, tag: str) -> List[XVal]:
    out: List[XVal] = []
    want = tag.lower()
    for m in RE_PLAIN_XBRL_TAG.finditer(x):
        t = (m.group("tag") or "").lower()
        if t != want:
            continue
        attrs = m.group("attrs") or ""
        cr_m = RE_ATTR_CONTEXTREF.search(attrs)
        if not cr_m:
            continue
        cr = cr_m.group(1).strip()
        v = clean_numeric(m.group("val"))
        if v is None:
            continue
        mul = xbrl_scale_multiplier(attrs)
        out.append(XVal(float(v) * mul, cr))
    return out

def duration_days(start: str, end: str) -> Optional[int]:
    try:
        sd = datetime.strptime(start, "%Y-%m-%d")
        ed = datetime.strptime(end, "%Y-%m-%d")
        return (ed - sd).days
    except Exception:
        return None

def pick_by_context(
    values: List[XVal],
    contexts: Dict[str, XContext],
    *,
    want: str,  # "instant" or "duration"
    form_type: str,
    period_end_iso: Optional[str],
) -> Optional[float]:
    if not values:
        return None

    # no period -> best consolidated-ish
    if not period_end_iso:
        best = min(values, key=lambda v: (context_penalty(v.context_ref), -abs(v.value)))
        return best.value

    have_ctx = [(v, contexts.get(v.context_ref)) for v in values]
    have_ctx = [(v, c) for (v, c) in have_ctx if c is not None]
    if not have_ctx:
        best = min(values, key=lambda v: (context_penalty(v.context_ref), -abs(v.value)))
        return best.value

    if want == "instant":
        exact = [(v, c) for (v, c) in have_ctx if c.kind == "instant" and c.instant == period_end_iso]
        if exact:
            best = min(exact, key=lambda t: (context_penalty(t[0].context_ref), -t[0].value))
            return best[0].value

        insts = [(v, c) for (v, c) in have_ctx if c.kind == "instant" and c.instant]
        if insts:
            insts.sort(key=lambda t: (t[1].instant, context_penalty(t[0].context_ref)))
            return insts[-1][0].value

        best = min(values, key=lambda v: (context_penalty(v.context_ref), -abs(v.value)))
        return best.value

    # want == duration
    end_hits = [(v, c) for (v, c) in have_ctx if c.kind == "duration" and c.end == period_end_iso]
    if not end_hits:
        best = min(values, key=lambda v: (context_penalty(v.context_ref), -abs(v.value)))
        return best.value

    scored: List[Tuple[XVal, int, int]] = []
    for v, c in end_hits:
        dd = duration_days(c.start, c.end) if (c.start and c.end) else None
        dd = dd if dd is not None else 10**9
        scored.append((v, dd, context_penalty(v.context_ref)))

    if form_type == "10-Q":
        quarter_like = [x for x in scored if 70 <= x[1] <= 130]
        if quarter_like:
            best = min(quarter_like, key=lambda t: (t[2], t[1]))
            return best[0].value
        best = min(scored, key=lambda t: (t[2], t[1]))
        return best[0].value

    if form_type == "10-K":
        year_like = [x for x in scored if 330 <= x[1] <= 390]
        if year_like:
            best = min(year_like, key=lambda t: (t[2], -t[1]))
            return best[0].value
        best = min(scored, key=lambda t: (t[2], -t[1]))
        return best[0].value

    best = min(scored, key=lambda t: (t[2], -abs(t[0].value)))
    return best[0].value

def infer_xbrl_money_multiplier(doc_text: str, extracted_money_values: Dict[str, Optional[float]]) -> Tuple[float, Optional[str]]:
    scale = detect_scale_factor(doc_text)
    if scale == 1.0:
        return 1.0, None

    vals = [v for v in extracted_money_values.values() if isinstance(v, (int, float)) and v is not None]
    if not vals:
        return 1.0, None

    vmax = max(abs(v) for v in vals)
    # If already large, do not apply heuristic scaling.
    if vmax >= 50_000_000:
        return 1.0, None

    if scale == 1_000.0:
        return 1_000.0, "XBRL_Heuristic_InThousands"
    if scale == 1_000_000.0:
        return 1_000_000.0, "XBRL_Heuristic_InMillions"
    return 1.0, None

def extract_from_xbrl_any(x: str, form_type: str, period_yyyymmdd: str) -> Tuple[Dict[str, Optional[float]], List[str]]:
    flags: List[str] = []
    flags.append("InlineXBRL" if has_inline_xbrl(x) else "PlainXBRL")

    out: Dict[str, Optional[float]] = {}
    contexts = parse_xbrl_contexts(x)
    period_end_iso = iso_from_yyyymmdd(period_yyyymmdd)

    # Diagnostic (not used for L)
    diag_vals: List[XVal] = []
    for t in TAG_CANDIDATES["BS_Check_LplusE"]:
        diag_vals.extend(extract_ix_values(x, t))
        diag_vals.extend(extract_plain_xbrl_values(x, t))
    out["BS_Check_LplusE"] = pick_by_context(diag_vals, contexts, want="instant", form_type=form_type, period_end_iso=period_end_iso)

    # INSTANT fields
    for field in CORE_FIELDS_INSTANT:
        vals: List[XVal] = []
        for tag in TAG_CANDIDATES[field]:
            vals.extend(extract_ix_values(x, tag))
            vals.extend(extract_plain_xbrl_values(x, tag))

        if field == "L_Total_Liabilities":
            # First try direct total liabilities
            L = None
            direct_vals: List[XVal] = []
            for tag in ["us-gaap:Liabilities"]:
                direct_vals.extend(extract_ix_values(x, tag))
                direct_vals.extend(extract_plain_xbrl_values(x, tag))
            L = pick_by_context(direct_vals, contexts, want="instant", form_type=form_type, period_end_iso=period_end_iso)

            if L is None:
                # try sum current + noncurrent
                cur_vals: List[XVal] = []
                non_vals: List[XVal] = []
                for t in ["us-gaap:LiabilitiesCurrent"]:
                    cur_vals.extend(extract_ix_values(x, t))
                    cur_vals.extend(extract_plain_xbrl_values(x, t))
                for t in ["us-gaap:LiabilitiesNoncurrent"]:
                    non_vals.extend(extract_ix_values(x, t))
                    non_vals.extend(extract_plain_xbrl_values(x, t))

                cur = pick_by_context(cur_vals, contexts, want="instant", form_type=form_type, period_end_iso=period_end_iso)
                non = pick_by_context(non_vals, contexts, want="instant", form_type=form_type, period_end_iso=period_end_iso)
                if cur is not None and non is not None:
                    L = float(cur) + float(non)
                    flags.append("DerivedL_SumCurrentNoncurrent")

            out[field] = L
        else:
            out[field] = pick_by_context(vals, contexts, want="instant", form_type=form_type, period_end_iso=period_end_iso)

    # DURATION fields
    for field in CORE_FIELDS_DURATION:
        vals: List[XVal] = []
        for tag in TAG_CANDIDATES[field]:
            vals.extend(extract_ix_values(x, tag))
            vals.extend(extract_plain_xbrl_values(x, tag))
        out[field] = pick_by_context(vals, contexts, want="duration", form_type=form_type, period_end_iso=period_end_iso)

    # Optional operating income (duration)
    op_vals: List[XVal] = []
    for tag in TAG_CANDIDATES["Operating_Income"]:
        op_vals.extend(extract_ix_values(x, tag))
        op_vals.extend(extract_plain_xbrl_values(x, tag))
    out["Operating_Income"] = pick_by_context(op_vals, contexts, want="duration", form_type=form_type, period_end_iso=period_end_iso)

    # Liquidity proxy H
    cash = out.get("Cash_CashEquiv")
    ms = out.get("Marketable_Securities")
    if cash is not None and ms is not None:
        out["H_HQLA_Proxy"] = float(cash) + float(ms)
    elif cash is not None:
        out["H_HQLA_Proxy"] = float(cash)
        flags.append("HProxyIsCashOnly")
    else:
        out["H_HQLA_Proxy"] = None
        flags.append("MissingH")

    # Apply conservative "in thousands/millions" only when values are suspiciously small
    money_fields = {
        "A_Total_Assets": out.get("A_Total_Assets"),
        "L_Total_Liabilities": out.get("L_Total_Liabilities"),
        "E_Total_Equity": out.get("E_Total_Equity"),
        "R_Total_Revenue": out.get("R_Total_Revenue"),
        "Pi_Net_Income": out.get("Pi_Net_Income"),
        "Cash_CashEquiv": out.get("Cash_CashEquiv"),
        "Marketable_Securities": out.get("Marketable_Securities"),
    }
    mul, mul_flag = infer_xbrl_money_multiplier(x, money_fields)
    if mul_flag:
        for k in list(money_fields.keys()):
            if out.get(k) is not None:
                out[k] = float(out[k]) * mul
        if out.get("H_HQLA_Proxy") is not None:
            out["H_HQLA_Proxy"] = float(out["H_HQLA_Proxy"]) * mul
        if out.get("Operating_Income") is not None:
            out["Operating_Income"] = float(out["Operating_Income"]) * mul
        if out.get("BS_Check_LplusE") is not None:
            out["BS_Check_LplusE"] = float(out["BS_Check_LplusE"]) * mul
        flags.append(mul_flag)

    # plausibility filter for key money fields
    for field in ["A_Total_Assets", "L_Total_Liabilities", "E_Total_Equity", "R_Total_Revenue"]:
        v = out.get(field)
        if v is not None and not accept_money_value(field, float(v)):
            out[field] = None
            flags.append(f"XBRL_Filtered:{field}")

    return out, flags


# -----------------------------
# Targeted HTML table extraction
# -----------------------------

def html_statement_windows(html: str, anchors: List[str], window_chars: int = 250_000) -> List[str]:
    lo = html.lower()
    windows: List[str] = []
    for a in anchors:
        idx = lo.find(a)
        if idx != -1:
            start = max(0, idx - window_chars // 3)
            end = min(len(html), idx + window_chars)
            windows.append(html[start:end])
    return windows

def flatten_cols(df: pd.DataFrame) -> List[str]:
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        return [" ".join([str(x) for x in t if str(x) != "nan"]).strip() for t in cols.values]
    return [str(c) for c in cols]

def best_numeric_column(df: pd.DataFrame, period_yyyymmdd: str) -> Optional[str]:
    year = period_yyyymmdd[:4] if period_yyyymmdd else ""
    candidates = []
    for c in df.columns[1:]:
        s = str(c).lower()
        score = 0
        if year and year in s:
            score += 5
        if "twelve months" in s or "year ended" in s:
            score += 2
        if "three months" in s:
            score += 1
        if any(ch.isdigit() for ch in s):
            score += 1
        candidates.append((score, c))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]

def match_row_label(label_text: str, synonyms: List[str]) -> bool:
    t = (label_text or "").lower()
    return any(s in t for s in synonyms)

def extract_from_tables_in_chunks(chunks: List[str], period_yyyymmdd: str, want_fields: List[str]) -> Tuple[Dict[str, Optional[float]], List[str]]:
    out: Dict[str, Optional[float]] = {f: None for f in want_fields}
    flags: List[str] = []

    for chunk in chunks:
        scale = detect_scale_factor(chunk)
        try:
            tables = pd.read_html(StringIO(chunk), flavor="lxml")
        except Exception:
            continue

        for tdf in tables[:40]:
            if tdf is None or tdf.empty:
                continue
            df = tdf.copy()
            df.columns = flatten_cols(df)
            if df.shape[1] < 2:
                continue

            label_col = df.columns[0]
            col = best_numeric_column(df, period_yyyymmdd) or df.columns[-1]

            for field in want_fields:
                if out[field] is not None:
                    continue
                syns = LABEL_SYNONYMS.get(field, [])
                if not syns:
                    continue

                for i in range(len(df)):
                    label = str(df.iloc[i][label_col])
                    if match_row_label(label, syns):
                        raw = df.iloc[i].get(col)
                        v = clean_numeric(raw)
                        if v is None:
                            continue
                        v = float(v) * scale
                        if accept_money_value(field, v):
                            out[field] = v
                            flags.append(f"HTML:{field}")
                        break

    if any(v is not None for v in out.values()):
        flags.insert(0, "HTMLTables_Targeted")
    return out, flags


# -----------------------------
# Tight label fallback (last resort)
# -----------------------------

def extract_by_tight_label(text: str, field: str) -> Optional[float]:
    syns = LABEL_SYNONYMS.get(field, [])
    lo = (text or "").lower()
    for s in syns:
        idx = lo.find(s)
        if idx == -1:
            continue
        snippet = text[idx: idx + 250]
        m = RE_NUMBER_TOKEN.search(snippet)
        if not m:
            continue
        v = clean_numeric(m.group(0))
        if v is None:
            continue
        if accept_money_value(field, v):
            return float(v) * detect_scale_factor(snippet)
    return None


# -----------------------------
# Sanity checks & derivations
# -----------------------------

def derive_missing_L_E(core: Dict[str, Optional[float]], flags: List[str]) -> None:
    """Try to fill missing L/E using A and the other field."""
    A = core.get("A_Total_Assets")
    L = core.get("L_Total_Liabilities")
    E = core.get("E_Total_Equity")

    if A is None:
        return

    if L is None and E is not None:
        core["L_Total_Liabilities"] = float(A) - float(E)
        flags.append("DerivedL_AminusE")

    if E is None and core.get("L_Total_Liabilities") is not None:
        core["E_Total_Equity"] = float(A) - float(core["L_Total_Liabilities"])
        flags.append("DerivedE_AminusL")

def balance_sheet_residual(A: float, L: float, E: float) -> float:
    return float(A) - float(L) - float(E)

def sanity_and_derive(core: Dict[str, Optional[float]], lambda_assumed: float, kappa_assumed: float) -> Tuple[Dict[str, Optional[float]], List[str]]:
    flags: List[str] = []

    # try derivations first
    derive_missing_L_E(core, flags)

    A = core.get("A_Total_Assets")
    L = core.get("L_Total_Liabilities")
    E = core.get("E_Total_Equity")
    R = core.get("R_Total_Revenue")
    Pi = core.get("Pi_Net_Income")
    H = core.get("H_HQLA_Proxy")
    OI = core.get("Operating_Income")

    if A is not None and A <= 0:
        flags.append("SANITY_FAIL_ASSETS_NONPOS")
        core["A_Total_Assets"] = None
        A = None

    bs_residual = None
    if A is not None and L is not None and E is not None and A != 0:
        bs_residual = balance_sheet_residual(A, L, E)
        # 10% tolerance
        if abs(bs_residual) / max(1.0, abs(A)) > 0.10:
            # attempt one more salvage: if one of L/E seems wrong, prefer derivation from the other
            flags.append("SANITY_FAIL_BS_MISMATCH")

            # Option 1: trust L, derive E = A - L
            E2 = float(A) - float(L)
            res2 = balance_sheet_residual(A, L, E2)

            # Option 2: trust E, derive L = A - E
            L2 = float(A) - float(E)
            res3 = balance_sheet_residual(A, L2, E)

            # pick best residual
            best = min(
                [("Keep", abs(bs_residual)), ("DeriveE", abs(res2)), ("DeriveL", abs(res3))],
                key=lambda t: t[1],
            )[0]

            if best == "DeriveE":
                core["E_Total_Equity"] = E2
                E = E2
                bs_residual = res2
                flags.append("SALVAGE_DerivedE_AminusL")
            elif best == "DeriveL":
                core["L_Total_Liabilities"] = L2
                L = L2
                bs_residual = res3
                flags.append("SALVAGE_DerivedL_AminusE")
            else:
                # last resort: blank L/E to avoid poisoning the dataset
                core["L_Total_Liabilities"] = None
                core["E_Total_Equity"] = None
                L = None
                E = None
                bs_residual = None
                flags.append("BLANKED_L_E_AFTER_BS_FAIL")

    # derived cost proxy
    C = None
    if R is not None and Pi is not None:
        C = float(R) - float(Pi)

    # operating income plausibility guard (loose)
    if OI is not None and R is not None:
        if abs(OI) > 3.0 * max(1.0, abs(R)):
            core["Operating_Income"] = None
            flags.append("SANITY_DROP_OperatingIncome_OutOfRange")

    liq_ratio = (float(H) / float(A)) if (H is not None and A not in (None, 0)) else None
    cap_ratio = (float(E) / float(A)) if (E is not None and A not in (None, 0)) else None
    liq_pass = (float(H) >= lambda_assumed * float(A)) if (H is not None and A is not None) else None
    cap_pass = (float(E) >= kappa_assumed * float(A)) if (E is not None and A is not None) else None

    derived = {
        "C_Derived_Cost": C,
        "L_Derived_IfMissing": None,  # kept for compatibility; derivation now happens in-place with flags
        "BS_Residual_A_minus_L_minus_E": bs_residual,
        "Liquidity_Ratio_H_over_A": liq_ratio,
        "Capital_Ratio_E_over_A": cap_ratio,
        "Lambda_Assumed": lambda_assumed,
        "Kappa_Assumed": kappa_assumed,
        "Liquidity_Pass_H_ge_lambdaA": liq_pass,
        "Capital_Pass_E_ge_kappaA": cap_pass,
    }
    return derived, flags


# -----------------------------
# Extract financials from a document
# -----------------------------

def extract_financials_any(content: str, form_type: str, period_yyyymmdd: str) -> Tuple[Dict[str, Optional[float]], List[str]]:
    # 1) XBRL
    if has_xbrl(content):
        return extract_from_xbrl_any(content, form_type, period_yyyymmdd)

    flags: List[str] = []

    # 2) HTML tables
    if "<html" in content.lower() or "<table" in content.lower():
        bs_chunks = html_statement_windows(content, BALANCE_SHEET_ANCHORS)
        is_chunks = html_statement_windows(content, INCOME_STMT_ANCHORS)

        want_bs = ["A_Total_Assets", "L_Total_Liabilities", "E_Total_Equity", "Cash_CashEquiv", "Marketable_Securities"]
        want_is = ["R_Total_Revenue", "Pi_Net_Income", "Operating_Income"]

        out: Dict[str, Optional[float]] = {}
        bs_vals, bs_flags = extract_from_tables_in_chunks(bs_chunks, period_yyyymmdd, want_bs)
        is_vals, is_flags = extract_from_tables_in_chunks(is_chunks, period_yyyymmdd, want_is)

        out.update(bs_vals)
        out.update(is_vals)
        flags.extend(bs_flags)
        flags.extend(is_flags)

        cash = out.get("Cash_CashEquiv")
        ms = out.get("Marketable_Securities")
        if cash is not None and ms is not None:
            out["H_HQLA_Proxy"] = float(cash) + float(ms)
        elif cash is not None:
            out["H_HQLA_Proxy"] = float(cash)
            flags.append("HProxyIsCashOnly")
        else:
            out["H_HQLA_Proxy"] = None
            flags.append("MissingH")

        if any(out.get(k) is not None for k in ["A_Total_Assets", "E_Total_Equity", "R_Total_Revenue", "Pi_Net_Income"]):
            return out, list(dict.fromkeys(flags))

    # 3) tight label fallback
    out: Dict[str, Optional[float]] = {}
    flags.append("LabelFallback_Tight")
    for field in [
        "A_Total_Assets", "L_Total_Liabilities", "E_Total_Equity",
        "R_Total_Revenue", "Pi_Net_Income", "Operating_Income",
        "Cash_CashEquiv", "Marketable_Securities"
    ]:
        out[field] = extract_by_tight_label(content, field)

    cash = out.get("Cash_CashEquiv")
    ms = out.get("Marketable_Securities")
    if cash is not None and ms is not None:
        out["H_HQLA_Proxy"] = float(cash) + float(ms)
    elif cash is not None:
        out["H_HQLA_Proxy"] = float(cash)
        flags.append("HProxyIsCashOnly")
    else:
        out["H_HQLA_Proxy"] = None
        flags.append("MissingH")

    return out, list(dict.fromkeys(flags))


# -----------------------------
# Records & rollup
# -----------------------------

@dataclass
class FilingRecord:
    company_bucket: str
    company_folder: str
    file_path: str
    rel_path: str

    company_name: str
    cik: str
    accession_number: str
    form_type: str
    period_of_report: str
    fiscal_year: Optional[int]

    A_Total_Assets: Optional[float]
    L_Total_Liabilities: Optional[float]
    E_Total_Equity: Optional[float]
    R_Total_Revenue: Optional[float]
    Pi_Net_Income: Optional[float]
    Operating_Income: Optional[float]
    Cash_CashEquiv: Optional[float]
    Marketable_Securities: Optional[float]
    H_HQLA_Proxy: Optional[float]
    BS_Check_LplusE: Optional[float]

    C_Derived_Cost: Optional[float]
    BS_Residual_A_minus_L_minus_E: Optional[float]
    Liquidity_Ratio_H_over_A: Optional[float]
    Capital_Ratio_E_over_A: Optional[float]
    Lambda_Assumed: float
    Kappa_Assumed: float
    Liquidity_Pass_H_ge_lambdaA: Optional[bool]
    Capital_Pass_E_ge_kappaA: Optional[bool]

    flags: str

def rollup_company_year(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    dfx = df.dropna(subset=["fiscal_year"]).copy()
    dfx["period_dt"] = pd.to_datetime(dfx["period_of_report"], format="%Y%m%d", errors="coerce")
    dfx = dfx.dropna(subset=["period_dt"])

    def pick_group(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("period_dt")
        g10k = g[g["form_type"] == "10-K"]
        if not g10k.empty:
            return g10k.iloc[-1]
        return g.iloc[-1]

    return (
        dfx.groupby(["company_bucket", "company_folder", "fiscal_year"], as_index=False)
           .apply(lambda g: pick_group(g))
           .reset_index(drop=True)
    )


# -----------------------------
# Collection
# -----------------------------

def collect_records(root_dir: str, lambda_assumed: float, kappa_assumed: float) -> List[FilingRecord]:
    records: List[FilingRecord] = []

    for bucket in COMPANY_BUCKETS:
        bucket_dir = os.path.join(root_dir, bucket)
        if not os.path.isdir(bucket_dir):
            continue

        for company_folder in sorted(os.listdir(bucket_dir)):
            comp_dir = os.path.join(bucket_dir, company_folder)
            if not os.path.isdir(comp_dir):
                continue

            for dirpath, _, filenames in os.walk(comp_dir):
                for fn in filenames:
                    if not fn.lower().endswith((".txt", ".htm", ".html", ".xml")):
                        continue

                    path = os.path.join(dirpath, fn)
                    content = safe_read_text(path)
                    if content is None:
                        continue

                    form_type = detect_form_type(content)
                    if form_type not in MAIN_FORMS:
                        continue

                    hdr = extract_header_fields(content)
                    period = (hdr.get("period_of_report") or "").strip()

                    if not period:
                        soup = BeautifulSoup(content, "lxml")
                        txt = soup.get_text(" ", strip=True)
                        period = parse_period_from_text_fallback(txt) or ""

                    if not period or not re.fullmatch(r"\d{8}", period):
                        continue

                    fiscal_year = int(period[:4])
                    primary, doc_flag = pick_primary_document(content, form_type)

                    core, method_flags = extract_financials_any(primary, form_type, period)

                    # apply sanity + derivations
                    derived, sanity_flags = sanity_and_derive(core, lambda_assumed, kappa_assumed)

                    flags = [doc_flag] + method_flags + sanity_flags
                    flags = [f for f in dict.fromkeys(flags) if f]

                    rec = FilingRecord(
                        company_bucket=bucket,
                        company_folder=company_folder,
                        file_path=path,
                        rel_path=os.path.relpath(path, root_dir).replace("\\", "/"),

                        company_name=(hdr.get("company_name") or company_folder).strip(),
                        cik=(hdr.get("cik") or "").strip(),
                        accession_number=(hdr.get("accession_number") or "").strip(),
                        form_type=form_type,
                        period_of_report=period,
                        fiscal_year=fiscal_year,

                        A_Total_Assets=core.get("A_Total_Assets"),
                        L_Total_Liabilities=core.get("L_Total_Liabilities"),
                        E_Total_Equity=core.get("E_Total_Equity"),
                        R_Total_Revenue=core.get("R_Total_Revenue"),
                        Pi_Net_Income=core.get("Pi_Net_Income"),
                        Operating_Income=core.get("Operating_Income"),
                        Cash_CashEquiv=core.get("Cash_CashEquiv"),
                        Marketable_Securities=core.get("Marketable_Securities"),
                        H_HQLA_Proxy=core.get("H_HQLA_Proxy"),
                        BS_Check_LplusE=core.get("BS_Check_LplusE"),

                        C_Derived_Cost=derived.get("C_Derived_Cost"),
                        BS_Residual_A_minus_L_minus_E=derived.get("BS_Residual_A_minus_L_minus_E"),
                        Liquidity_Ratio_H_over_A=derived.get("Liquidity_Ratio_H_over_A"),
                        Capital_Ratio_E_over_A=derived.get("Capital_Ratio_E_over_A"),
                        Lambda_Assumed=derived.get("Lambda_Assumed"),
                        Kappa_Assumed=derived.get("Kappa_Assumed"),
                        Liquidity_Pass_H_ge_lambdaA=derived.get("Liquidity_Pass_H_ge_lambdaA"),
                        Capital_Pass_E_ge_kappaA=derived.get("Capital_Pass_E_ge_kappaA"),

                        flags=";".join(flags),
                    )
                    records.append(rec)

    return records


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, default="data")
    ap.add_argument("--out_filing_csv", type=str, default="per_filing_financials.csv")
    ap.add_argument("--out_year_csv", type=str, default="per_company_year_financials.csv")
    ap.add_argument("--lambda_assumed", type=float, default=0.05)
    ap.add_argument("--kappa_assumed", type=float, default=0.08)
    args = ap.parse_args()

    records = collect_records(args.root_dir, args.lambda_assumed, args.kappa_assumed)
    df = pd.DataFrame([r.__dict__ for r in records])

    if df.empty:
        print("❌ No main filings extracted (10-K/10-Q/8-K).")
        return

    df["period_dt"] = pd.to_datetime(df["period_of_report"], format="%Y%m%d", errors="coerce")
    df.to_csv(args.out_filing_csv, index=False)
    print(f"✅ Wrote {args.out_filing_csv} ({len(df)} rows)")

    df_year = rollup_company_year(df)
    df_year.to_csv(args.out_year_csv, index=False)
    print(f"✅ Wrote {args.out_year_csv} ({len(df_year)} rows)")

    print("\nDiagnostics:")
    print(df_year.groupby("company_bucket")["company_folder"].nunique().rename("unique_companies"))
    print("\nTop flags (sample):")
    print(df["flags"].value_counts().head(15))


if __name__ == "__main__":
    main()
