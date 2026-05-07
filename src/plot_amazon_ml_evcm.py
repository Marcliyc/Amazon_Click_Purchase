from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_loss(loss_history: pd.DataFrame, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_history["step"], loss_history["loss"], label="loss")
    if "nll" in loss_history:
        ax.plot(loss_history["step"], loss_history["nll"], label="nll", alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Objective")
    ax.set_title("Amazon ML EV/CM training loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "training_loss.png", dpi=140)
    plt.close(fig)


def plot_param_by_segment(machine_params: pd.DataFrame, segment_col: str, param: str, out_path: str | Path) -> None:
    frame = machine_params[[segment_col, param]].dropna().copy()
    if frame.empty:
        return
    summary = frame.groupby(segment_col, as_index=False)[param].mean().sort_values(param)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(summary[segment_col].astype(str), summary[param])
    ax.set_title(f"Mean {param} by {segment_col}")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_param_histograms(machine_params: pd.DataFrame, params: list[str], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for param in params:
        if param not in machine_params:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(machine_params[param].dropna(), bins=30)
        ax.set_title(f"Distribution of {param}")
        fig.tight_layout()
        fig.savefig(out / f"hist_{param}.png", dpi=140)
        plt.close(fig)


def plot_coefficients(coefficients: pd.DataFrame, out_path: str | Path) -> None:
    if coefficients.empty:
        return
    pivot = coefficients.pivot(index="feature", columns="parameter", values="coefficient")
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * pivot.shape[1]), max(6, 0.18 * pivot.shape[0])))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    ax.set_title("Linear covariate head coefficients (unconstrained scale)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_default_plots(loss_history: pd.DataFrame, machine_params: pd.DataFrame, coefficients: pd.DataFrame, plots_dir: str | Path) -> None:
    plots = Path(plots_dir)
    plots.mkdir(parents=True, exist_ok=True)
    plot_training_loss(loss_history, plots)
    plot_param_by_segment(machine_params, "household_income", "cm_pi", plots / "param_by_income.png")
    plot_param_by_segment(machine_params, "census_region", "cm_pi", plots / "param_by_region.png")
    plot_coefficients(coefficients, plots / "covariate_coefficients_heatmap.png")
    plot_param_histograms(machine_params, ["ev_r", "cm_mu0", "cm_k", "cm_pi"], plots)
