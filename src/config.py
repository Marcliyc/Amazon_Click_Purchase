from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "raw_path": "data/amazon_sessions.csv",
        "output_dir": "outputs/cbmt_amazon",
        "customer_id_col": "machine_id",
        "session_id_col": "site_session_id",
        "date_col": "event_date",
        "transaction_flag_col": "tran_flg",
        "payment_col": "totalprice",
        "pages_col": "pages_viewed",
        "duration_col": "duration",
        "quantity_col": None,
        "covariate_cols": [],
        "cohort_definition": "first_visit",
        "page_agg": "max",
        "duration_agg": "max",
    },
    "split": {"val_weeks": 12, "holdout_weeks": 12, "week_start": "SUN", "oracle_holdout_cohorts": False},
    "model": {
        "lookback_weeks": 20,
        "d_model": 128,
        "n_heads": 4,
        "n_encoder_layers": 2,
        "dropout": 0.1,
        "head_hidden_dim": 128,
        "batch_size": 512,
        "max_epochs": 200,
        "patience": 20,
        "lr_backbone": 3e-4,
        "lr_heads": 1e-3,
        "weight_decay": 1e-4,
        "aov_weight_decay_mult": 1000,
        "gradient_clip_norm": 1.0,
        "seed": 123,
        "loss_weights": {},
    },
}


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return deep_update(DEFAULT_CONFIG, user_cfg)


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(config), f, sort_keys=False)
