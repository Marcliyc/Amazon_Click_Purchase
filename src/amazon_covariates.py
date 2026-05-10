from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

CATEGORICAL_COLUMNS = ["census_region", "racial_background", "country_of_origin"]
NUMERIC_COLUMNS = ["household_size", "household_income"]


@dataclass
class AmazonCovariateMetadata:
    categorical_columns: list[str]
    numeric_columns: list[str]
    category_levels: dict[str, list[str]]
    numeric_medians: dict[str, float]
    numeric_means: dict[str, float]
    numeric_stds: dict[str, float]
    rare_min_count: int
    standardize: bool
    feature_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AmazonCovariateMetadata":
        return cls(**payload)


def _first_non_null(series: pd.Series):
    nn = series.dropna()
    if len(nn) == 0:
        return np.nan
    mode = nn.mode(dropna=True)
    return mode.iloc[0] if len(mode) else nn.iloc[0]


def _machine_covariate_frame(df: pd.DataFrame, machine_ids: list | None = None) -> pd.DataFrame:
    cols = ["machine_id", *CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS]
    available = [c for c in cols if c in df.columns]
    frame = df[available].copy()
    grouped = frame.groupby("machine_id", as_index=False).agg({c: _first_non_null for c in available if c != "machine_id"})
    if machine_ids is not None:
        order = pd.DataFrame({"machine_id": machine_ids})
        grouped = order.merge(grouped, on="machine_id", how="left")
    return grouped


def fit_amazon_covariate_metadata(
    df: pd.DataFrame,
    rare_min_count: int = 10,
    standardize: bool = True,
) -> AmazonCovariateMetadata:
    machine = _machine_covariate_frame(df)
    category_levels: dict[str, list[str]] = {}
    numeric_medians: dict[str, float] = {}
    numeric_means: dict[str, float] = {}
    numeric_stds: dict[str, float] = {}

    for col in CATEGORICAL_COLUMNS:
        s = machine[col] if col in machine.columns else pd.Series(dtype=object)
        filled = s.fillna("Unknown").astype(str)
        counts = filled.value_counts(dropna=False)
        levels = sorted([str(k) for k, v in counts.items() if v >= rare_min_count and str(k) != "Unknown"])
        category_levels[col] = ["Unknown", "Other", *levels]

    for col in NUMERIC_COLUMNS:
        raw = pd.to_numeric(machine[col], errors="coerce") if col in machine.columns else pd.Series(dtype=float)
        median = float(raw.median()) if raw.notna().any() else 0.0
        filled = raw.fillna(median).astype(float)
        mean = float(filled.mean()) if len(filled) else 0.0
        std = float(filled.std(ddof=0)) if len(filled) else 1.0
        numeric_medians[col] = median
        numeric_means[col] = mean
        numeric_stds[col] = std if std > 1e-8 else 1.0

    feature_names = [*NUMERIC_COLUMNS]
    for col in CATEGORICAL_COLUMNS:
        feature_names.extend([f"{col}={level}" for level in category_levels[col]])

    return AmazonCovariateMetadata(
        categorical_columns=CATEGORICAL_COLUMNS.copy(),
        numeric_columns=NUMERIC_COLUMNS.copy(),
        category_levels=category_levels,
        numeric_medians=numeric_medians,
        numeric_means=numeric_means,
        numeric_stds=numeric_stds,
        rare_min_count=rare_min_count,
        standardize=standardize,
        feature_names=feature_names,
    )


def transform_amazon_covariates(
    df: pd.DataFrame,
    metadata: AmazonCovariateMetadata,
    machine_ids: list | None = None,
) -> tuple[np.ndarray, list[str], dict, pd.DataFrame]:
    machine = _machine_covariate_frame(df, machine_ids=machine_ids)
    blocks = []

    for col in metadata.numeric_columns:
        raw = pd.to_numeric(machine[col], errors="coerce") if col in machine.columns else pd.Series(np.nan, index=machine.index)
        vals = raw.fillna(metadata.numeric_medians[col]).astype(float).to_numpy()
        if metadata.standardize:
            vals = (vals - metadata.numeric_means[col]) / metadata.numeric_stds[col]
        blocks.append(vals.reshape(-1, 1))

    for col in metadata.categorical_columns:
        raw = machine[col] if col in machine.columns else pd.Series("Unknown", index=machine.index)
        vals = raw.fillna("Unknown").astype(str)
        allowed = set(metadata.category_levels[col])
        vals = vals.where(vals.isin(allowed), "Other")
        for level in metadata.category_levels[col]:
            blocks.append((vals == level).astype(float).to_numpy().reshape(-1, 1))

    X = np.concatenate(blocks, axis=1).astype(np.float32) if blocks else np.zeros((len(machine), 0), dtype=np.float32)
    machine_index = {mid: i for i, mid in enumerate(machine["machine_id"].tolist())}
    return X, metadata.feature_names, machine_index, machine


def build_amazon_covariates(
    df: pd.DataFrame,
    machine_ids: list | None = None,
    metadata: AmazonCovariateMetadata | None = None,
    rare_min_count: int = 10,
    standardize: bool = True,
) -> tuple[np.ndarray, list[str], dict, AmazonCovariateMetadata, pd.DataFrame]:
    if metadata is None:
        metadata = fit_amazon_covariate_metadata(df, rare_min_count=rare_min_count, standardize=standardize)
    X, feature_names, machine_index, machine_frame = transform_amazon_covariates(df, metadata, machine_ids=machine_ids)
    return X, feature_names, machine_index, metadata, machine_frame
