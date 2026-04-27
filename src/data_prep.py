from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

DEMOGRAPHIC_COLUMNS = [
    "census_region",
    "household_size",
    "household_income",
    "racial_background",
    "country_of_origin",
]


@dataclass
class SplitInfo:
    global_start: str
    calibration_end: str
    holdout_end: str
    n_machines_calibration: int
    n_machines_holdout_known: int
    n_machines_holdout_new: int


def load_raw_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    df["event_datetime"] = pd.to_datetime(
        df["event_date"].astype(str) + " " + df["event_time"].astype(str),
        errors="coerce",
        cache=True,
    )
    if df["event_datetime"].isna().any():
        bad = int(df["event_datetime"].isna().sum())
        raise ValueError(f"Unable to parse {bad} timestamps from event_date/event_time")

    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
    df = df.sort_values(["machine_id", "event_datetime", "site_session_id"], kind="mergesort").reset_index(drop=True)

    # user-requested rule: purchase is defined strictly by tran_flg only
    df["purchase_row"] = df["tran_flg"].fillna(0).astype(float) > 0
    return df


def _first_non_null(series: pd.Series):
    nn = series.dropna()
    return nn.iloc[0] if len(nn) else np.nan


def make_session_visits(df: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "event_datetime": "min",
        "purchase_row": "max",
        "pages_viewed": "max",
        "duration": "max",
        #"basket_tot": "max",
        #"prod_totprice": "sum",
    }
    for col in DEMOGRAPHIC_COLUMNS:
        if col in df.columns:
            agg[col] = _first_non_null

    session = (
        df.groupby(["machine_id", "site_session_id"], as_index=False)
        .agg(agg)
        .rename(columns={"event_datetime": "visit_datetime", "purchase_row": "purchase"})
    )
    session["visit_date"] = pd.to_datetime(session["visit_datetime"]).dt.date
    session["purchase"] = session["purchase"].astype(int)
    session["purchase_session_count"] = session["purchase"]
    session = session.sort_values(["machine_id", "visit_datetime", "site_session_id"]).reset_index(drop=True)
    return session


def make_daily_visits(session_df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        session_df.groupby(["machine_id", "visit_date"], as_index=False)
        .agg(
            visit_datetime=("visit_datetime", "min"),
            purchase=("purchase", "max"),
            purchase_session_count=("purchase_session_count", "sum"),
            pages_viewed=("pages_viewed", "sum"),
            duration=("duration", "sum"),
            #basket_tot=("basket_tot", "sum"),
            #prod_totprice=("prod_totprice", "sum"),
        )
        .sort_values(["machine_id", "visit_datetime"])
        .reset_index(drop=True)
    )
    global_start = pd.to_datetime(daily["visit_date"]).min()
    daily["t"] = (pd.to_datetime(daily["visit_date"]) - global_start).dt.days.astype(float)
    return daily


def make_session_time_visits(session_df: pd.DataFrame) -> pd.DataFrame:
    out = session_df.copy()
    global_start = out["visit_datetime"].min()
    out["t"] = (out["visit_datetime"] - global_start).dt.total_seconds() / 86400.0
    out = out.sort_values(["machine_id", "t"]).reset_index(drop=True)
    for mid, idx in out.groupby("machine_id").groups.items():
        vals = out.loc[idx, "t"].to_numpy().copy()
        for i in range(1, len(vals)):
            if vals[i] <= vals[i - 1]:
                vals[i] = vals[i - 1] + 1e-6
        out.loc[idx, "t"] = vals
    return out


def split_calibration_holdout(
    visits: pd.DataFrame,
    cutoff: str | None = None,
    calibration_fraction: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, SplitInfo]:
    dates = pd.to_datetime(visits.get("visit_date", pd.to_datetime(visits["visit_datetime"]).dt.date))
    global_start = dates.min()
    holdout_end = dates.max()

    if cutoff:
        cal_end = pd.to_datetime(cutoff)
    else:
        span_days = max((holdout_end - global_start).days, 1)
        cal_end = global_start + pd.Timedelta(days=int(np.floor(span_days * calibration_fraction)))

    cal = visits[dates <= cal_end].copy()
    hold = visits[dates > cal_end].copy()

    cal_m = set(cal["machine_id"].unique())
    hold_m = set(hold["machine_id"].unique())
    known = hold_m & cal_m
    new = hold_m - cal_m
    info = SplitInfo(
        global_start=global_start.date().isoformat(),
        calibration_end=cal_end.date().isoformat(),
        holdout_end=holdout_end.date().isoformat(),
        n_machines_calibration=len(cal_m),
        n_machines_holdout_known=len(known),
        n_machines_holdout_new=len(new),
    )
    return cal, hold, info


def prepare_visits(path: str | Path, visit_unit: Literal["daily", "session"] = "daily") -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_raw_data(path)
    session = make_session_visits(raw)
    if visit_unit == "daily":
        visits = make_daily_visits(session)
    else:
        visits = make_session_time_visits(session)
    return raw, visits
