from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.cohort_builder import build_and_save

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/amazon_cbmt.yaml"); args = ap.parse_args()
    build_and_save(load_config(args.config))
