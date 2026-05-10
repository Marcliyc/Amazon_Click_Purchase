from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import week_freq


def _existing(cols: list[str], df: pd.DataFrame) -> list[str]:
    return [c for c in cols if c and c in df.columns]


def add_calendar_week(df: pd.DataFrame, date_col: str, week_start: str = "SUN") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out["calendar_week"] = out[date_col].dt.to_period(week_freq(week_start)).dt.start_time
    return out


def clean_raw_amazon(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    dcfg = config["data"]
    customer_col = dcfg["customer_id_col"]
    date_col = dcfg["date_col"]
    txn_col = dcfg.get("transaction_flag_col")
    pay_col = dcfg.get("payment_col")
    out = df.copy()
    diagnostics: dict[str, Any] = {"raw_rows": int(len(out))}
    out = add_calendar_week(out, date_col, config.get("split", {}).get("week_start", "SUN"))
    out = out.dropna(subset=[customer_col, date_col, "calendar_week"])
    diagnostics["rows_after_required_drop"] = int(len(out))

    if txn_col and txn_col in out.columns:
        out[txn_col] = pd.to_numeric(out[txn_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
    else:
        out["_transaction_flag"] = 0
        txn_col = "_transaction_flag"

    if pay_col and pay_col in out.columns:
        out[pay_col] = pd.to_numeric(out[pay_col], errors="coerce")
        diagnostics["missing_payment_on_txn_rows"] = int(((out[txn_col] == 1) & out[pay_col].isna()).sum())
        diagnostics["negative_payment_rows"] = int((out[pay_col] < 0).fillna(False).sum())
        out.loc[out[pay_col] < 0, pay_col] = np.nan
        out.loc[(out[txn_col] == 1) & out[pay_col].isna(), pay_col] = 0.0
        out.loc[out[txn_col] == 0, pay_col] = 0.0
        out[pay_col] = out[pay_col].fillna(0.0)
    else:
        pay_col = "_payment"
        out[pay_col] = 0.0
    return out, diagnostics


def aggregate_to_sessions(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    dcfg = config["data"]
    customer_col = dcfg["customer_id_col"]
    session_col = dcfg.get("session_id_col") or "_synthetic_session_id"
    if session_col not in df.columns:
        df = df.copy()
        df[session_col] = np.arange(len(df))
    txn_col = dcfg.get("transaction_flag_col") if dcfg.get("transaction_flag_col") in df.columns else "_transaction_flag"
    pay_col = dcfg.get("payment_col") if dcfg.get("payment_col") in df.columns else "_payment"
    pages_col = dcfg.get("pages_col")
    duration_col = dcfg.get("duration_col")
    qty_col = dcfg.get("quantity_col")
    cov_cols = _existing(dcfg.get("covariate_cols", []), df)

    group_cols = [customer_col, session_col, "calendar_week"]
    agg: dict[str, Any] = {txn_col: "max", pay_col: "sum"}
    if pages_col in df.columns:
        agg[pages_col] = dcfg.get("page_agg", "max")
    if duration_col in df.columns:
        agg[duration_col] = dcfg.get("duration_agg", "max")
    if qty_col in df.columns:
        df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)
        agg[qty_col] = "sum"
    for c in cov_cols:
        agg[c] = "first"

    sessions = df.groupby(group_cols, dropna=False).agg(agg).reset_index()
    sessions = sessions.rename(columns={txn_col: "transaction", pay_col: "payment"})
    sessions["visit"] = 1.0
    if pages_col in sessions.columns:
        sessions[pages_col] = pd.to_numeric(sessions[pages_col], errors="coerce").fillna(0)
    if duration_col in sessions.columns:
        sessions[duration_col] = pd.to_numeric(sessions[duration_col], errors="coerce").fillna(0)
    sessions["transaction"] = sessions["transaction"].fillna(0).astype(int).clip(0, 1)
    sessions.loc[sessions["transaction"] == 0, "payment"] = 0.0
    sessions["payment"] = sessions["payment"].fillna(0.0).clip(lower=0.0)
    return sessions


def load_and_preprocess(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_path = Path(config["data"]["raw_path"])
    df = pd.read_csv(raw_path)
    cleaned, diag = clean_raw_amazon(df, config)
    sessions = aggregate_to_sessions(cleaned, config)
    diag["session_rows"] = int(len(sessions))
    return sessions, diag
