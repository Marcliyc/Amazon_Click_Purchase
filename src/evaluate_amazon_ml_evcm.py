from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .plot_amazon_ml_evcm import save_default_plots


def main(args=None):
    parser = argparse.ArgumentParser(description="Regenerate Amazon ML EV/CM diagnostics from a fit directory.")
    parser.add_argument("--fit-path", required=True, help="Directory produced by train_amazon_ml_evcm.py")
    parsed = parser.parse_args(args=args)

    fit = Path(parsed.fit_path)
    plots = fit / "plots"
    loss = pd.read_csv(fit / "training_loss.csv")
    machine_params = pd.read_csv(fit / "machine_parameter_predictions.csv")
    coefficients = pd.read_csv(fit / "covariate_coefficients.csv")
    save_default_plots(loss, machine_params, coefficients, plots)

    summary = pd.read_csv(fit / "segment_parameter_summary.csv")
    comparison = pd.read_csv(fit / "model_comparison.csv")
    print(f"Regenerated plots in {plots}")
    print("Model comparison:")
    print(comparison.to_string(index=False))
    print("Top segment summaries:")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
