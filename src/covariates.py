from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def add_time_covariates(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    w = pd.to_datetime(out["calendar_week"])
    c = pd.to_datetime(out["cohort_week"])
    out["weekofyear"] = w.dt.isocalendar().week.astype(int)
    out["month"] = w.dt.month.astype(int)
    out["quarter"] = w.dt.quarter.astype(int)
    out["year"] = w.dt.year.astype(int)
    base = w.min()
    trend = ((w - base).dt.days // 7).astype(float)
    out["time_trend"] = trend
    out["time_trend_sq"] = trend**2
    out["tenure_week_sq"] = out["tenure_week"].astype(float) ** 2
    out["log1p_tenure_week"] = np.log1p(out["tenure_week"].astype(float))
    if "cohort_size" in out.columns:
        out["log1p_cohort_size"] = np.log1p(out["cohort_size"].astype(float).clip(lower=0))
        out["sqrt_cohort_size"] = np.sqrt(out["cohort_size"].astype(float).clip(lower=0))
    out["acquisition_month"] = c.dt.month.astype(int)
    out["acquisition_weekofyear"] = c.dt.isocalendar().week.astype(int)
    out["is_thanksgiving_week"] = ((out["month"] == 11) & (w.dt.day.between(22, 30))).astype(int)
    out["is_black_friday_week"] = out["is_thanksgiving_week"]
    out["is_cyber_monday_week"] = ((out["month"] == 11) & (w.dt.day >= 24) | ((out["month"] == 12) & (w.dt.day <= 3))).astype(int)
    out["is_christmas_week"] = ((out["month"] == 12) & (w.dt.day.between(19, 31))).astype(int)
    return out


def build_cohort_covariates(sessions: pd.DataFrame, cohorts: pd.DataFrame, config: dict[str, Any], max_cardinality: int = 12) -> pd.DataFrame:
    customer_col = config["data"]["customer_id_col"]
    cov_cols = [c for c in config["data"].get("covariate_cols", []) if c in sessions.columns]
    if not cov_cols:
        return cohorts[[customer_col, "cohort_week"]].drop_duplicates().groupby("cohort_week", as_index=False).size().drop(columns="size")
    cust = sessions[[customer_col] + cov_cols].groupby(customer_col, as_index=False).first().merge(cohorts, on=customer_col, how="inner")
    frames = []
    for col in cov_cols:
        if pd.api.types.is_numeric_dtype(cust[col]):
            g = cust.groupby("cohort_week")[col]
            frames.append(g.agg(["mean", "median", "std"]).add_prefix(f"{col}_"))
            frames.append(g.apply(lambda s: s.isna().mean()).to_frame(f"{col}_missing_rate"))
        else:
            vals = cust[col].astype("string").fillna("__MISSING__")
            tmp = cust.assign(_val=vals)
            mode = tmp.groupby("cohort_week")["_val"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "__MISSING__").to_frame(f"{col}_mode")
            frames.append(mode)
            top = vals.value_counts().head(max_cardinality).index.tolist()
            for cat in top:
                frames.append(tmp.assign(_is=(tmp["_val"] == cat).astype(float)).groupby("cohort_week")["_is"].mean().to_frame(f"{col}_prop_{cat}"))
    features = pd.concat(frames, axis=1).reset_index().fillna(0.0)
    # Turn mode strings into stable integer codes without target leakage.
    for col in features.select_dtypes(include=["object", "string"]).columns:
        features[col] = pd.Categorical(features[col]).codes.astype(float)
    return features


def attach_covariates(panel: pd.DataFrame, cohort_features: pd.DataFrame | None = None) -> pd.DataFrame:
    out = add_time_covariates(panel)
    if cohort_features is not None and "cohort_week" in cohort_features.columns:
        out = out.merge(cohort_features, on="cohort_week", how="left")
    return out.fillna(0.0)


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    # Keep raw semantic columns unscaled because they are used as targets and in
    # visit/transaction/revenue consistency math during training and forecasting.
    # Scaled feature proxies such as log1p_cohort_size and log1p_tenure_week are
    # included instead.
    exclude = {
        "cohort_week",
        "calendar_week",
        "cohort_size",
        "tenure_week",
        "visits",
        "transactions",
        "revenue",
        "visits_per_customer",
        "transactions_per_customer",
        "avg_payment",
        "revenue_per_customer",
        "total_visits",
        "total_transactions",
        "total_revenue",
        "total_units",
    }
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


class CovariateScaler:
    def __init__(self) -> None:
        self.columns: list[str] = []
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame, columns: list[str]) -> "CovariateScaler":
        self.columns = columns
        if columns:
            self.scaler.fit(df[columns].astype(float))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.columns:
            out[self.columns] = self.scaler.transform(out[self.columns].astype(float))
        return out
