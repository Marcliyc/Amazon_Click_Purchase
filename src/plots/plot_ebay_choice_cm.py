from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_diagnostic_plots(monthly: pd.DataFrame, loss_history: pd.DataFrame, params, workdir: Path) -> None:
    monthly = monthly.sort_values("month").copy()
    monthly["actual_cum"] = monthly["actual_ebay_purchases"].cumsum()
    monthly["pred_cum"] = monthly["pred_mean_ebay_purchases"].cumsum()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(monthly["month"], monthly["actual_ebay_purchases"], label="Actual")
    ax.plot(monthly["month"], monthly["pred_mean_ebay_purchases"], label="Predicted")
    ax.legend()
    ax.set_title("Monthly eBay purchases")
    fig.tight_layout()
    fig.savefig(workdir / "monthly_forecast_plot.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(monthly["month"], monthly["actual_cum"], label="Actual cum")
    ax.plot(monthly["month"], monthly["pred_cum"], label="Predicted cum")
    ax.legend()
    ax.set_title("Cumulative eBay purchases")
    fig.tight_layout()
    fig.savefig(workdir / "cumulative_forecast_plot.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_history["step"], loss_history["loss"])
    ax.set_title("Loss curve")
    fig.tight_layout()
    fig.savefig(workdir / "loss_curve_plot.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["pi", "mu0", "k"]
    amazon = [params.pi_amazon, params.mu0_amazon, params.k_amazon]
    ebay = [params.pi_ebay, params.mu0_ebay, params.k_ebay]
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], amazon, width=0.4, label="Amazon")
    ax.bar([i + 0.2 for i in x], ebay, width=0.4, label="eBay")
    ax.set_xticks(list(x), names)
    ax.legend()
    ax.set_title("Amazon vs eBay params")
    fig.tight_layout()
    fig.savefig(workdir / "amazon_vs_ebay_params.png", dpi=140)
    plt.close(fig)
