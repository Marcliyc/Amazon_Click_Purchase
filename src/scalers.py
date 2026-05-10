from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET_COLUMNS = ["visits_per_customer", "transactions_per_customer", "avg_payment", "total_visits", "total_transactions", "total_revenue"]


class LogStandardScaler:
    def __init__(self) -> None:
        self.columns: list[str] = []
        self.scalers: dict[str, StandardScaler] = {}

    def fit(self, df: pd.DataFrame, columns: list[str]) -> "LogStandardScaler":
        self.columns = list(columns)
        for c in self.columns:
            s = StandardScaler()
            s.fit(np.log1p(np.asarray(df[c], dtype=float).reshape(-1, 1)))
            self.scalers[c] = s
        return self

    def transform_array(self, values, column: str):
        arr = np.asarray(values, dtype=float).reshape(-1, 1)
        return self.scalers[column].transform(np.log1p(np.clip(arr, 0, None))).reshape(np.asarray(values).shape)

    def inverse_array(self, values, column: str):
        arr = np.asarray(values, dtype=float).reshape(-1, 1)
        return np.expm1(self.scalers[column].inverse_transform(arr)).reshape(np.asarray(values).shape).clip(min=0)

    def transform_frame(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        out = df.copy()
        for c in columns or self.columns:
            out[c] = self.transform_array(out[c].values, c)
        return out
