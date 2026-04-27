from __future__ import annotations

import numpy as np
import pandas as pd


def information_criteria(ll: float, k: int, n: int):
    n = max(n, 1)
    return {
        "AIC": float(-2 * ll + 2 * k),
        "BIC": float(-2 * ll + k * np.log(n)),
        "CAIC": float(-2 * ll + k * (np.log(n) + 1)),
    }


def forecast_error_metrics(df: pd.DataFrame, eps: float = 1.0):
    out = df.copy()
    out["ape_cumulative"] = (out["forecast_mean_cum_purchases"] - out["actual_cum_purchases"]).abs() / np.maximum(out["actual_cum_purchases"], eps)
    cum_err = out["forecast_mean_cum_purchases"] - out["actual_cum_purchases"]
    inc_err = out["forecast_mean_purchases"] - out["actual_purchases"]
    return {
        "MAPE_cumulative": float(out["ape_cumulative"].mean()),
        "MAE_cumulative": float(cum_err.abs().mean()),
        "RMSE_cumulative": float(np.sqrt((cum_err**2).mean())),
        "final_cumulative_error_pct": float(cum_err.iloc[-1] / max(float(out["actual_cum_purchases"].iloc[-1]), eps)),
        "MAPE_incremental": float((inc_err.abs() / np.maximum(out["actual_purchases"], eps)).mean()),
    }, out
