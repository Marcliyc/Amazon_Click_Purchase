from __future__ import annotations

import argparse

from .config import load_config
from .cohort_builder import build_and_save
from .evaluate import evaluate_outputs
from .forecast import forecast_holdout
from .plots import make_plots
from .train import train_cbmt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/amazon_cbmt.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    build_and_save(cfg)
    train_cbmt(cfg)
    forecast_holdout(cfg)
    evaluate_outputs(cfg["data"]["output_dir"])
    make_plots(cfg["data"]["output_dir"])

if __name__ == "__main__":
    main()
