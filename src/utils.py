from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def write_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def week_freq(week_start: str = "SUN") -> str:
    # Pandas W-* aliases are labelled by week end. A week starting SUN ends SAT.
    mapping = {"SUN": "W-SAT", "MON": "W-SUN", "TUE": "W-MON", "WED": "W-TUE", "THU": "W-WED", "FRI": "W-THU", "SAT": "W-FRI"}
    return mapping.get(str(week_start).upper(), "W-SAT")
