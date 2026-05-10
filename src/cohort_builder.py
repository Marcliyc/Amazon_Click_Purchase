from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_preprocess import load_and_preprocess
from .utils import ensure_dir, write_json


def assign_cohorts(sessions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    customer_col = config["data"]["customer_id_col"]
    mode = config["data"].get("cohort_definition", "first_visit")
    base = sessions if mode == "first_visit" else sessions[sessions["transaction"] == 1]
    cohorts = base.groupby(customer_col, as_index=False)["calendar_week"].min().rename(columns={"calendar_week": "cohort_week"})
    return cohorts


def build_cohort_week_panel(sessions: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    customer_col = config["data"]["customer_id_col"]
    cohorts = assign_cohorts(sessions, config)
    sess = sessions.merge(cohorts, on=customer_col, how="inner")
    sess = sess[sess["calendar_week"] >= sess["cohort_week"]].copy()
    cohort_sizes = cohorts.groupby("cohort_week", as_index=False)[customer_col].nunique().rename(columns={customer_col: "cohort_size"})
    weekly = sess.groupby(["cohort_week", "calendar_week"], as_index=False).agg(
        visits=("visit", "sum"), transactions=("transaction", "sum"), revenue=("payment", "sum")
    )
    if config["data"].get("quantity_col") and config["data"].get("quantity_col") in sess.columns:
        qty_col = config["data"]["quantity_col"]
        qty = sess.groupby(["cohort_week", "calendar_week"], as_index=False)[qty_col].sum().rename(columns={qty_col: "total_units"})
        weekly = weekly.merge(qty, on=["cohort_week", "calendar_week"], how="left")

    all_weeks = pd.date_range(cohorts["cohort_week"].min(), sess["calendar_week"].max(), freq="7D")
    rows = []
    for cw in sorted(cohort_sizes["cohort_week"].unique()):
        for w in all_weeks:
            if w >= cw:
                rows.append((pd.Timestamp(cw), pd.Timestamp(w)))
    panel = pd.DataFrame(rows, columns=["cohort_week", "calendar_week"])
    panel = panel.merge(cohort_sizes, on="cohort_week", how="left").merge(weekly, on=["cohort_week", "calendar_week"], how="left")
    for c in ["visits", "transactions", "revenue", "total_units"]:
        if c in panel.columns:
            panel[c] = panel[c].fillna(0.0)
    panel["tenure_week"] = ((panel["calendar_week"] - panel["cohort_week"]).dt.days // 7).astype(int)
    panel["visits_per_customer"] = panel["visits"] / panel["cohort_size"].replace(0, np.nan)
    panel["transactions_per_customer"] = panel["transactions"] / panel["cohort_size"].replace(0, np.nan)
    panel["avg_payment"] = panel["revenue"] / panel["transactions"].where(panel["transactions"] > 0, 1)
    panel["revenue_per_customer"] = panel["revenue"] / panel["cohort_size"].replace(0, np.nan)
    panel = panel.fillna(0.0).sort_values(["cohort_week", "calendar_week"]).reset_index(drop=True)

    aggregate = sess.groupby("calendar_week", as_index=False).agg(
        total_visits=("visit", "sum"), total_transactions=("transaction", "sum"), total_revenue=("payment", "sum"), active_customers=(customer_col, "nunique")
    )
    aggregate["conversion_rate"] = aggregate["total_transactions"] / aggregate["total_visits"].where(aggregate["total_visits"] > 0, np.nan)
    aggregate["avg_payment"] = aggregate["total_revenue"] / aggregate["total_transactions"].where(aggregate["total_transactions"] > 0, np.nan)
    aggregate = aggregate.fillna(0.0)
    return panel, aggregate, cohorts


def build_and_save(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outdir = ensure_dir(config["data"]["output_dir"])
    sessions, diag = load_and_preprocess(config)
    panel, aggregate, cohorts = build_cohort_week_panel(sessions, config)
    sessions.to_csv(outdir / "cleaned_sessions.csv", index=False)
    cohorts.to_csv(outdir / "customer_cohorts.csv", index=False)
    panel.to_csv(outdir / "cohort_week_panel.csv", index=False)
    aggregate.to_csv(outdir / "aggregate_week_panel.csv", index=False)
    write_json(diag | {"panel_rows": int(len(panel)), "aggregate_rows": int(len(aggregate))}, outdir / "preprocess_diagnostics.json")
    return panel, aggregate, cohorts
