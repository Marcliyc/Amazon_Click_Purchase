from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

BEHAVIOR_COLUMNS = ["visits_per_customer", "transactions_per_customer", "avg_payment", "total_visits", "total_transactions", "total_revenue"]
TARGET_KEYS = ["visits_pc", "txns_pc", "avg_payment", "agg_visits", "agg_txns", "agg_revenue"]


@dataclass
class Splits:
    train_weeks: list[pd.Timestamp]
    val_weeks: list[pd.Timestamp]
    holdout_weeks: list[pd.Timestamp]


def temporal_splits(weeks: list[pd.Timestamp], val_weeks: int, holdout_weeks: int) -> Splits:
    ordered = sorted(pd.to_datetime(pd.Series(weeks)).drop_duplicates())
    hold = ordered[-holdout_weeks:] if holdout_weeks else []
    val = ordered[-(holdout_weeks + val_weeks): -holdout_weeks or None] if val_weeks else []
    train = ordered[: len(ordered) - len(val) - len(hold)]
    return Splits(train, val, hold)


def merge_aggregate(panel: pd.DataFrame, aggregate: pd.DataFrame) -> pd.DataFrame:
    return panel.merge(aggregate[["calendar_week", "total_visits", "total_transactions", "total_revenue"]], on="calendar_week", how="left").fillna(0.0)


class WindowedCohortDataset(Dataset):
    def __init__(self, panel: pd.DataFrame, feature_cols: list[str], target_weeks: list[pd.Timestamp], lookback_weeks: int = 20):
        self.panel = panel.copy()
        self.panel["calendar_week"] = pd.to_datetime(self.panel["calendar_week"])
        self.panel["cohort_week"] = pd.to_datetime(self.panel["cohort_week"])
        self.feature_cols = feature_cols
        self.lookback_weeks = lookback_weeks
        self.rows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        target_set = set(pd.to_datetime(target_weeks))
        for _, r in self.panel[["cohort_week", "calendar_week"]].drop_duplicates().iterrows():
            if r["calendar_week"] in target_set:
                self.rows.append((r["cohort_week"], r["calendar_week"]))
        self.by_key = {(pd.Timestamp(r["cohort_week"]), pd.Timestamp(r["calendar_week"])): r for r in self.panel.to_dict("records")}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        cohort_week, target_week = self.rows[idx]
        hist_weeks = [target_week - pd.Timedelta(weeks=i) for i in range(self.lookback_weeks, 0, -1)]
        seq = []
        prebirth = []
        for w in hist_weeks:
            row = self.by_key.get((cohort_week, w))
            if row is None:
                vals = [0.0] * (len(BEHAVIOR_COLUMNS) + len(self.feature_cols))
                prebirth.append(1.0)
            else:
                vals = [float(row.get(c, 0.0)) for c in BEHAVIOR_COLUMNS] + [float(row.get(c, 0.0)) for c in self.feature_cols]
                prebirth.append(float(w < cohort_week))
            seq.append(vals + [prebirth[-1]])
        row = self.by_key[(cohort_week, target_week)]
        y = np.array([
            row["visits_per_customer"], row["transactions_per_customer"], row["avg_payment"],
            row["total_visits"], row["total_transactions"], row["total_revenue"],
        ], dtype=np.float32)
        consistency = np.array([row["visits"], row["transactions"], row["revenue"], row["cohort_size"]], dtype=np.float32)
        return {
            "x_seq": torch.tensor(seq, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "consistency": torch.tensor(consistency, dtype=torch.float32),
            "cohort_week": str(cohort_week.date()),
            "target_week": str(target_week.date()),
        }
