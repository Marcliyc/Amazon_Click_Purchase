from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.evaluate import evaluate_outputs
from src.plots import make_plots

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/amazon_cbmt.yaml"); args = ap.parse_args()
    cfg = load_config(args.config)
    print(evaluate_outputs(cfg["data"]["output_dir"])); make_plots(cfg["data"]["output_dir"])
