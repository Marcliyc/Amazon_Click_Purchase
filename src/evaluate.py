from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def smape(a, p, eps: float = 1e-8):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return float(np.mean(2 * np.abs(p - a) / (np.abs(a) + np.abs(p) + eps)))


def mape(a, p, eps: float = 1e-8):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return float(np.mean(np.abs((a - p) / np.maximum(np.abs(a), eps))))


def mae(a, p):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(p, float))))


def rmse(a, p):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(p, float)) ** 2)))


def wape(a, p, eps: float = 1e-8):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return float(np.sum(np.abs(a - p)) / (np.sum(np.abs(a)) + eps))


def metrics_for_frame(df: pd.DataFrame, pairs: dict[str, tuple[str, str]]) -> dict[str, dict[str, float]]:
    out = {}
    for name, (actual, pred) in pairs.items():
        out[name] = {"sMAPE": smape(df[actual], df[pred]), "MAPE": mape(df[actual], df[pred]), "MAE": mae(df[actual], df[pred]), "RMSE": rmse(df[actual], df[pred]), "WAPE": wape(df[actual], df[pred])}
    return out


def rolling_mean_baseline(history: pd.Series, horizon: int, window: int = 4) -> list[float]:
    vals = list(history.astype(float))
    preds = []
    for _ in range(horizon):
        pred = float(np.mean(vals[-window:])) if vals else 0.0
        preds.append(pred)
        vals.append(pred)
    return preds


def evaluate_outputs(output_dir: str | Path) -> dict:
    outdir = Path(output_dir)
    weekly = pd.read_csv(outdir / "predictions" / "holdout_weekly_predictions.csv")
    pairs = {
        "total_visits": ("actual_total_visits", "pred_total_visits"),
        "total_transactions": ("actual_total_transactions", "pred_total_transactions"),
        "total_revenue": ("actual_total_revenue", "pred_total_revenue"),
    }
    metrics = metrics_for_frame(weekly, pairs)
    (outdir / "predictions").mkdir(parents=True, exist_ok=True)
    with open(outdir / "predictions" / "holdout_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics
