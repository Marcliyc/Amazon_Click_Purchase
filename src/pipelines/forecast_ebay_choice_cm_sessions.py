from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import pandas as pd

from src.ev_beta_choice_cm import EVBetaChoiceCMConfig, predicted_amazon_visits, predicted_ebay_visits, predicted_monthly_mean
from src.pipelines.fit_ebay_choice_cm_jax import _filter_domain, _resolve_session_key, load_config


def _load_raw_params(params_path: str | Path) -> dict[str, jnp.ndarray]:
    with open(params_path, encoding="utf-8") as f:
        payload = json.load(f)
    if "raw" not in payload:
        raise ValueError("params file must contain a 'raw' object, as produced by fit_ebay_choice_cm_jax.py")
    return {k: jnp.asarray(float(v), dtype=jnp.float32) for k, v in payload["raw"].items()}


def _month_grid(start: str, end: str) -> pd.DataFrame:
    months = pd.period_range(pd.to_datetime(start).to_period("M"), pd.to_datetime(end).to_period("M"), freq="M").to_timestamp("M")
    return pd.DataFrame({"month": months}).assign(month_index=lambda d: range(len(d)))


def build_session_visit_transaction_summary(
    df: pd.DataFrame,
    date_col: str,
    purchase_col: str,
    session_col: str | None = "user_session_id",
) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out[purchase_col] = out[purchase_col].fillna(0).astype(int)
    key = _resolve_session_key(out, session_col)
    if key == "fallback_session_key":
        out[key] = out["machine_id"].astype(str) + "_" + out[date_col].dt.strftime("%Y-%m-%d") + "_" + out.get("event_time", "00:00:00").astype(str)

    grouped = out.groupby(key, as_index=False).agg(
        machine_id=("machine_id", "first"),
        visit_start=(date_col, "min"),
        visit_end=(date_col, "max"),
        actual_transactions=(purchase_col, "max"),
        transaction_row_count=(purchase_col, "sum"),
        raw_row_count=(purchase_col, "size"),
    )
    grouped = grouped.rename(columns={key: "user_session_id"})
    grouped["month"] = grouped["visit_start"].dt.to_period("M").dt.to_timestamp("M")
    grouped["actual_visits"] = 1
    return grouped[
        [
            "user_session_id",
            "machine_id",
            "visit_start",
            "visit_end",
            "month",
            "actual_visits",
            "actual_transactions",
            "transaction_row_count",
            "raw_row_count",
        ]
    ].sort_values(["month", "user_session_id"])


def score_sessions_with_monthly_forecast(
    sessions: pd.DataFrame,
    raw_params: dict[str, jnp.ndarray],
    model_cfg: EVBetaChoiceCMConfig,
    total_customers: float,
    start: str,
    end: str,
    platform: str = "ebay",
) -> pd.DataFrame:
    grid = _month_grid(start, end)
    x = jnp.asarray(grid["month_index"].to_numpy(), dtype=jnp.float32)

    if platform == "amazon":
        monthly_visits = predicted_amazon_visits(raw_params, x, total_customers, model_cfg)
        monthly_transactions = jnp.full_like(monthly_visits, jnp.nan)
    elif platform == "ebay":
        monthly_visits = predicted_ebay_visits(raw_params, x, total_customers, model_cfg)
        monthly_transactions = predicted_monthly_mean(raw_params, x, total_customers, model_cfg)
    else:
        raise ValueError("platform must be 'ebay' or 'amazon'")

    forecast = grid.copy()
    forecast["pred_monthly_visits"] = [float(v) for v in monthly_visits]
    forecast["pred_monthly_transactions"] = [float(v) for v in monthly_transactions]

    scored = sessions.merge(forecast, on="month", how="left")
    observed_sessions = scored.groupby("month")["user_session_id"].transform("count").clip(lower=1)
    scored["month_observed_sessions"] = observed_sessions.astype(int)
    scored["model_expected_visits"] = scored["pred_monthly_visits"] / observed_sessions
    scored["model_expected_transactions"] = scored["pred_monthly_transactions"] / observed_sessions
    scored["model_expected_transaction_probability"] = scored["model_expected_transactions"].clip(lower=0.0, upper=1.0)
    return scored


def main(args=None):
    parser = argparse.ArgumentParser(description="Score observed sessions with fitted EV-Beta-Choice-CM monthly forecasts.")
    parser.add_argument("--config", required=True, help="Path to the Python config used for fitting.")
    parser.add_argument("--params", required=True, help="Path to params_fitted.json from fit_ebay_choice_cm_jax.py.")
    parser.add_argument("--output", required=True, help="Output CSV path for per-user_session_id visits and transactions.")
    parser.add_argument("--platform", choices=["ebay", "amazon"], default="ebay", help="Which platform data to score.")
    parser.add_argument("--input", default=None, help="Optional session CSV override. Defaults to platform path in config.")
    parser.add_argument("--start", default=None, help="Optional inclusive start date override.")
    parser.add_argument("--end", default=None, help="Optional inclusive end date override.")
    parsed = parser.parse_args(args=args)

    cfg = load_config(parsed.config)
    data_cfg = cfg["data"]
    date_range = cfg["date_range"]
    model_cfg = EVBetaChoiceCMConfig(
        amazon_fixed=cfg["amazon_fixed"],
        ebay_init=cfg["ebay_init"],
        choice=cfg["choice"],
        priors=cfg["priors"],
        fit=cfg["fit"],
    )

    path = parsed.input or data_cfg[f"{parsed.platform}_path"]
    start = parsed.start or date_range["start"]
    end = parsed.end or date_range["holdout_end"]

    raw = pd.read_csv(path)
    raw[data_cfg["date_col"]] = pd.to_datetime(raw[data_cfg["date_col"]])
    raw = raw[(raw[data_cfg["date_col"]] >= start) & (raw[data_cfg["date_col"]] <= end)].copy()
    raw = _filter_domain(raw, data_cfg.get("domain_col"), data_cfg.get(f"{parsed.platform}_domain_value"))

    raw_params = _load_raw_params(parsed.params)
    sessions = build_session_visit_transaction_summary(
        raw,
        date_col=data_cfg["date_col"],
        purchase_col=data_cfg["purchase_col"],
        session_col=data_cfg.get("session_id_col"),
    )
    total_customers = float(raw["machine_id"].nunique())
    scored = score_sessions_with_monthly_forecast(
        sessions=sessions,
        raw_params=raw_params,
        model_cfg=model_cfg,
        total_customers=total_customers,
        start=start,
        end=end,
        platform=parsed.platform,
    )

    output = Path(parsed.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output, index=False)
    print(f"Wrote {len(scored)} session rows to {output}")


if __name__ == "__main__":
    main()
