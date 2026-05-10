from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _line_plot(df, actual, pred, title, path):
    plt.figure(figsize=(10, 5))
    x = pd.to_datetime(df["calendar_week"])
    plt.plot(x, df[actual], marker="o", label="Actual")
    plt.plot(x, df[pred], marker="o", label="Predicted")
    plt.title(title)
    plt.xlabel("Week")
    plt.ylabel(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def make_plots(output_dir: str | Path) -> None:
    outdir = Path(output_dir)
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    weekly = pd.read_csv(outdir / "predictions" / "holdout_weekly_predictions.csv")
    cohort = pd.read_csv(outdir / "predictions" / "holdout_cohort_week_predictions.csv")
    _line_plot(weekly, "actual_total_visits", "pred_total_visits", "Weekly Total Visits", plot_dir / "weekly_total_visits_actual_vs_pred.png")
    _line_plot(weekly, "actual_total_transactions", "pred_total_transactions", "Weekly Total Transactions", plot_dir / "weekly_total_transactions_actual_vs_pred.png")
    _line_plot(weekly, "actual_total_revenue", "pred_total_revenue", "Weekly Total Revenue", plot_dir / "weekly_total_revenue_actual_vs_pred.png")
    for val, name in [("actual_revenue", "actual"), ("pred_revenue", "pred")]:
        if not cohort.empty:
            piv = cohort.pivot_table(index="cohort_week", columns="calendar_week", values=val, aggfunc="sum", fill_value=0)
            plt.figure(figsize=(12, 6))
            sns.heatmap(piv, cmap="viridis")
            plt.title(f"Cohort Heatmap {name.title()} Revenue")
            plt.tight_layout()
            plt.savefig(plot_dir / f"cohort_heatmap_{name}_revenue.png")
            plt.close()
    if not cohort.empty:
        err = cohort.assign(abs_revenue_error=(cohort["pred_revenue"] - cohort["actual_revenue"]).abs()).groupby("tenure_week", as_index=False)["abs_revenue_error"].mean()
        plt.figure(figsize=(8, 4))
        plt.plot(err["tenure_week"], err["abs_revenue_error"], marker="o")
        plt.title("Cohort Revenue Error by Tenure")
        plt.tight_layout()
        plt.savefig(plot_dir / "cohort_error_by_tenure.png")
        plt.close()
        seg = cohort.groupby("tenure_week", as_index=False)[["pred_visits_per_customer", "pred_transactions_per_customer", "pred_avg_payment", "pred_revenue_per_customer"]].mean()
        plt.figure(figsize=(10, 6))
        for col in seg.columns[1:]:
            plt.plot(seg["tenure_week"], seg[col], label=col)
        plt.title("Average Predicted Segment Behavior by Tenure")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "segment_behavior_parameters.png")
        plt.close()
