from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .cm_model import fit_cm_model
from .ev_model import fit_ev_model
from .forecasting import simulate_evcm_forecast
from .utils import ensure_dir, write_json


def sessions_to_evcm_visits(sessions: pd.DataFrame, customer_col: str = "machine_id") -> pd.DataFrame:
    if sessions.empty:
        return pd.DataFrame(columns=[customer_col, "calendar_week", "t", "purchase", "payment", "visit"])
    out = sessions.copy()
    out["calendar_week"] = pd.to_datetime(out["calendar_week"])
    global_start = out["calendar_week"].min()
    out["t"] = (out["calendar_week"] - global_start).dt.days.astype(float)
    out["purchase"] = (out["transaction"] > 0).astype(int)
    out["payment"] = out["payment"].fillna(0.0).clip(lower=0.0)
    return out.sort_values([customer_col, "t"])


def _split_by_weeks(visits: pd.DataFrame, train_weeks: list[pd.Timestamp], holdout_weeks: list[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    train_set = set(pd.to_datetime(train_weeks))
    holdout_set = set(pd.to_datetime(holdout_weeks))
    cal = visits[visits["calendar_week"].isin(train_set)].copy()
    hold = visits[visits["calendar_week"].isin(holdout_set)].copy()
    if len(cal):
        t_cal_end = float((pd.to_datetime(train_weeks).max() - visits["calendar_week"].min()).days + 7)
    else:
        t_cal_end = 0.0
    if len(holdout_weeks):
        t_holdout_end = float((pd.to_datetime(holdout_weeks).max() - visits["calendar_week"].min()).days + 7)
    else:
        t_holdout_end = t_cal_end
    return cal, hold, t_cal_end, t_holdout_end


def train_evcm_early_purchase(config: dict[str, Any], splits: Any) -> dict[str, Any]:
    outdir = Path(config["data"]["output_dir"])
    model_dir = ensure_dir(outdir / "models")
    sessions_path = outdir / "cleaned_sessions.csv"
    segments_path = outdir / "customer_segments.csv"
    if not sessions_path.exists() or not segments_path.exists() or not config.get("evcm", {}).get("enabled", True):
        return {"enabled": False, "n_customers": 0}
    sessions = pd.read_csv(sessions_path, parse_dates=["calendar_week"])
    segments = pd.read_csv(segments_path)
    customer_col = config["data"]["customer_id_col"]
    early = set(segments.loc[segments["cbmt_segment"] == "early_purchase_evcm", customer_col])
    seg_sessions = sessions[sessions[customer_col].isin(early)].copy()
    visits = sessions_to_evcm_visits(seg_sessions, customer_col)
    cal, _, t_cal_end, _ = _split_by_weeks(visits, splits.train_weeks + splits.val_weeks, splits.holdout_weeks)
    if cal.empty or cal[customer_col].nunique() == 0:
        return {"enabled": False, "n_customers": int(len(early)), "reason": "empty calibration segment"}
    ev_params, ev_info = fit_ev_model(cal.rename(columns={customer_col: "machine_id"}), t_cal_end, n_starts=int(config.get("evcm", {}).get("ev_starts", 5)), seed=int(config["model"].get("seed", 123)))
    cm_params, cm_info = fit_cm_model(cal.rename(columns={customer_col: "machine_id"}), n_starts=int(config.get("evcm", {}).get("cm_starts", 5)), seed=int(config["model"].get("seed", 123)))
    avg_payment = float(cal.loc[cal["purchase"] > 0, "payment"].sum() / max(cal.loc[cal["purchase"] > 0, "purchase"].sum(), 1))
    artifact = {
        "ev_params": ev_params,
        "cm_params": cm_params,
        "avg_payment": avg_payment,
        "global_start": str(visits["calendar_week"].min().date()),
        "customer_col": customer_col,
    }
    joblib.dump(artifact, model_dir / "evcm_early_purchase.pkl")
    diag = {"enabled": True, "n_customers": int(len(early)), "avg_payment": avg_payment, "ev_info": ev_info, "cm_info": cm_info}
    write_json(diag, model_dir / "evcm_early_purchase_metrics.json")
    return diag


def forecast_evcm_early_purchase(config: dict[str, Any], splits: Any) -> pd.DataFrame:
    outdir = Path(config["data"]["output_dir"])
    artifact_path = outdir / "models" / "evcm_early_purchase.pkl"
    holdout_weeks = [pd.Timestamp(w) for w in splits.holdout_weeks]
    base = pd.DataFrame({"calendar_week": [str(w.date()) for w in holdout_weeks]})
    if not artifact_path.exists() or not len(holdout_weeks):
        return base.assign(pred_total_visits_evcm=0.0, pred_total_transactions_evcm=0.0, pred_total_revenue_evcm=0.0, actual_total_visits_evcm=0.0, actual_total_transactions_evcm=0.0, actual_total_revenue_evcm=0.0)
    artifact = joblib.load(artifact_path)
    sessions = pd.read_csv(outdir / "cleaned_sessions.csv", parse_dates=["calendar_week"])
    segments = pd.read_csv(outdir / "customer_segments.csv")
    customer_col = artifact["customer_col"]
    early = set(segments.loc[segments["cbmt_segment"] == "early_purchase_evcm", customer_col])
    visits = sessions_to_evcm_visits(sessions[sessions[customer_col].isin(early)].copy(), customer_col)
    cal, hold, t_cal_end, t_holdout_end = _split_by_weeks(visits, splits.train_weeks + splits.val_weeks, holdout_weeks)
    if cal.empty:
        return base.assign(pred_total_visits_evcm=0.0, pred_total_transactions_evcm=0.0, pred_total_revenue_evcm=0.0, actual_total_visits_evcm=0.0, actual_total_transactions_evcm=0.0, actual_total_revenue_evcm=0.0)
    sim = simulate_evcm_forecast(
        cal.rename(columns={customer_col: "machine_id"}),
        t_cal_end,
        t_holdout_end,
        artifact["ev_params"],
        artifact["cm_params"],
        n_sims=int(config.get("evcm", {}).get("n_sims", 300)),
        freq="W",
        seed=int(config["model"].get("seed", 123)),
    )
    pred = base.copy()
    pred["pred_total_transactions_evcm"] = sim["forecast_mean_purchases"].to_numpy()[: len(pred)] if len(sim) else 0.0
    total_visits = float(sim.attrs.get("forecast_holdout_visits_mean", 0.0))
    shares = pred["pred_total_transactions_evcm"] / max(pred["pred_total_transactions_evcm"].sum(), 1e-9)
    pred["pred_total_visits_evcm"] = total_visits * shares if pred["pred_total_transactions_evcm"].sum() > 0 else total_visits / max(len(pred), 1)
    pred["pred_total_revenue_evcm"] = pred["pred_total_transactions_evcm"] * float(artifact["avg_payment"])
    actual = hold.groupby("calendar_week", as_index=False).agg(actual_total_visits_evcm=("visit", "sum"), actual_total_transactions_evcm=("purchase", "sum"), actual_total_revenue_evcm=("payment", "sum"))
    actual["calendar_week"] = pd.to_datetime(actual["calendar_week"]).dt.date.astype(str)
    return pred.merge(actual, on="calendar_week", how="left").fillna(0.0)
