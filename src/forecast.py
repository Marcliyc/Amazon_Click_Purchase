from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import ensure_dir


def _load_model(model_dir: Path, device: Any):
    import torch

    from .model_cbmt import CBMTTransformer

    ckpt = torch.load(model_dir / "cbmt_best.pt", map_location=device)
    cfg = ckpt["config"]
    model = CBMTTransformer(input_dim=ckpt["input_dim"], **{k: cfg["model"][k] for k in ["d_model", "n_heads", "n_encoder_layers", "dropout", "head_hidden_dim"] if k in cfg["model"]})
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


def add_predicted_cold_cohorts(panel: pd.DataFrame, holdout_weeks: list[pd.Timestamp]) -> pd.DataFrame:
    """Add non-oracle future cohorts for every remaining holdout week.

    Each holdout week introduces one predicted new cohort.  A cohort born in the
    first holdout week must also have rows for the second, third, etc. holdout
    weeks so rolling forecasts can feed its predicted tenure-0 behavior into
    tenure-1/2 histories and include it in later aggregate totals.
    """
    if not holdout_weeks:
        return panel

    holdout_weeks = sorted(pd.to_datetime(pd.Series(holdout_weeks)).drop_duplicates())
    existing_cols = panel.columns
    existing_pairs = set(zip(pd.to_datetime(panel["cohort_week"]), pd.to_datetime(panel["calendar_week"])))
    pre = panel[panel["calendar_week"] < holdout_weeks[0]]
    new_sizes = pre.loc[pre["cohort_week"] == pre["calendar_week"], "cohort_size"].tail(8)
    pred_size = float(max(new_sizes.mean() if len(new_sizes) else 1.0, 1.0))

    rows = []
    for birth_week in holdout_weeks:
        for calendar_week in holdout_weeks:
            if calendar_week < birth_week or (birth_week, calendar_week) in existing_pairs:
                continue
            r = {c: 0.0 for c in existing_cols}
            tenure_week = int((calendar_week - birth_week).days // 7)
            r.update({
                "cohort_week": birth_week,
                "calendar_week": calendar_week,
                "cohort_size": pred_size,
                "tenure_week": tenure_week,
            })
            rows.append(r)
    return pd.concat([panel, pd.DataFrame(rows)], ignore_index=True) if rows else panel


def forecast_holdout(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    import torch

    from .dataset import WindowedCohortDataset
    from .losses import implied_cohort_counts, implied_cohort_revenue
    from .evcm_segment import forecast_evcm_early_purchase
    from .train import prepare_model_tables

    outdir = Path(config["data"]["output_dir"])
    pred_dir = ensure_dir(outdir / "predictions")
    panel, feature_cols, splits, _ = prepare_model_tables(config)
    oracle = bool(config["split"].get("oracle_holdout_cohorts", False))
    first_holdout = splits.holdout_weeks[0] if splits.holdout_weeks else None
    if not oracle and first_holdout is not None:
        panel = panel[panel["cohort_week"] < first_holdout].copy()
        panel = add_predicted_cold_cohorts(panel, splits.holdout_weeks)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = _load_model(outdir / "models", device)
    lookback = int(config["model"].get("lookback_weeks", 20))
    cohort_preds = []
    work = panel.copy()
    for week in splits.holdout_weeks:
        target = work[work["calendar_week"] == week]
        if target.empty:
            continue
        ds = WindowedCohortDataset(work, feature_cols, [week], lookback)
        for item in ds:
            x = item["x_seq"].unsqueeze(0).to(device)
            with torch.no_grad():
                p = {k: float(v.cpu().item()) for k, v in model(x).items()}
            cw = pd.Timestamp(item["cohort_week"]); tw = pd.Timestamp(item["target_week"])
            mask = (work["cohort_week"] == cw) & (work["calendar_week"] == tw)
            actual = work.loc[mask].iloc[0]
            pred_visits, pred_txns = implied_cohort_counts(float(actual["cohort_size"]), p["visits_pc"], p["txns_pc"])
            pred_rev = implied_cohort_revenue(float(actual["cohort_size"]), p["txns_pc"], p["avg_payment"])
            cohort_preds.append({
                "cohort_week": str(cw.date()), "calendar_week": str(tw.date()), "tenure_week": int(actual["tenure_week"]), "cohort_size": float(actual["cohort_size"]),
                "actual_visits": float(actual.get("visits", 0)), "actual_transactions": float(actual.get("transactions", 0)), "actual_revenue": float(actual.get("revenue", 0)),
                "actual_visits_per_customer": float(actual.get("visits_per_customer", 0)), "actual_transactions_per_customer": float(actual.get("transactions_per_customer", 0)), "actual_avg_payment": float(actual.get("avg_payment", 0)),
                "pred_visits_per_customer": p["visits_pc"], "pred_transactions_per_customer": p["txns_pc"], "pred_avg_payment": p["avg_payment"],
                "pred_visits": float(pred_visits), "pred_transactions": float(pred_txns), "pred_revenue": float(pred_rev), "pred_revenue_per_customer": float(pred_rev / max(float(actual["cohort_size"]), 1.0)),
            })
            # Feed predictions into later holdout histories rather than leaking actual target values.
            work.loc[mask, "visits_per_customer"] = p["visits_pc"]
            work.loc[mask, "transactions_per_customer"] = p["txns_pc"]
            work.loc[mask, "avg_payment"] = p["avg_payment"]
            work.loc[mask, "visits"] = float(pred_visits); work.loc[mask, "transactions"] = float(pred_txns); work.loc[mask, "revenue"] = float(pred_rev)
        week_preds = pd.DataFrame([r for r in cohort_preds if r["calendar_week"] == str(pd.Timestamp(week).date())])
        if not week_preds.empty:
            for c, source in [("total_visits", "pred_visits"), ("total_transactions", "pred_transactions"), ("total_revenue", "pred_revenue")]:
                work.loc[work["calendar_week"] == week, c] = week_preds[source].sum()
    cohort_df = pd.DataFrame(cohort_preds)
    transformer_actual_path = outdir / "aggregate_week_panel_transformer.csv"
    if not transformer_actual_path.exists():
        transformer_actual_path = outdir / "aggregate_week_panel.csv"
    actual_transformer = pd.read_csv(transformer_actual_path, parse_dates=["calendar_week"])
    weekly_pred = (
        cohort_df.groupby("calendar_week", as_index=False).agg(
            pred_total_visits_transformer=("pred_visits", "sum"),
            pred_total_transactions_transformer=("pred_transactions", "sum"),
            pred_total_revenue_transformer=("pred_revenue", "sum"),
        )
        if not cohort_df.empty
        else pd.DataFrame(columns=["calendar_week", "pred_total_visits_transformer", "pred_total_transactions_transformer", "pred_total_revenue_transformer"])
    )
    weekly = pd.DataFrame({"calendar_week": [str(pd.Timestamp(w).date()) for w in splits.holdout_weeks]}).merge(weekly_pred, on="calendar_week", how="left").fillna(0.0)
    actual_transformer["calendar_week"] = actual_transformer["calendar_week"].dt.date.astype(str)
    weekly = weekly.merge(
        actual_transformer.rename(
            columns={
                "total_visits": "actual_total_visits_transformer",
                "total_transactions": "actual_total_transactions_transformer",
                "total_revenue": "actual_total_revenue_transformer",
            }
        )[["calendar_week", "actual_total_visits_transformer", "actual_total_transactions_transformer", "actual_total_revenue_transformer"]],
        on="calendar_week",
        how="left",
    ).fillna(0.0)

    evcm_weekly = forecast_evcm_early_purchase(config, splits)
    evcm_weekly.to_csv(pred_dir / "holdout_weekly_predictions_evcm_early_purchase.csv", index=False)
    weekly = weekly.merge(evcm_weekly, on="calendar_week", how="left").fillna(0.0)
    for metric in ["visits", "transactions", "revenue"]:
        weekly[f"pred_total_{metric}"] = weekly[f"pred_total_{metric}_transformer"] + weekly[f"pred_total_{metric}_evcm"]
        weekly[f"actual_total_{metric}"] = weekly[f"actual_total_{metric}_transformer"] + weekly[f"actual_total_{metric}_evcm"]

    cohort_df.to_csv(pred_dir / "holdout_cohort_week_predictions.csv", index=False)
    weekly.to_csv(pred_dir / "holdout_weekly_predictions.csv", index=False)
    return weekly, cohort_df
