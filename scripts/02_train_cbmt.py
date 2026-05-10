from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.train import train_cbmt

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/amazon_cbmt.yaml"); args = ap.parse_args()
    print(train_cbmt(load_config(args.config)))
