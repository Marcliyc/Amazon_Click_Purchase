import pandas as pd

from src.covariates import attach_covariates, numeric_feature_columns, CovariateScaler


def test_scaling_features_does_not_overwrite_targets_or_consistency_columns():
    panel = pd.DataFrame(
        {
            "cohort_week": pd.to_datetime(["2024-01-07", "2024-01-07"]),
            "calendar_week": pd.to_datetime(["2024-01-07", "2024-01-14"]),
            "cohort_size": [10.0, 10.0],
            "tenure_week": [0, 1],
            "visits": [5.0, 3.0],
            "transactions": [1.0, 0.0],
            "revenue": [20.0, 0.0],
            "visits_per_customer": [0.5, 0.3],
            "transactions_per_customer": [0.1, 0.0],
            "avg_payment": [20.0, 0.0],
            "revenue_per_customer": [2.0, 0.0],
            "total_visits": [5.0, 3.0],
            "total_transactions": [1.0, 0.0],
            "total_revenue": [20.0, 0.0],
        }
    )
    data = attach_covariates(panel)
    feature_cols = numeric_feature_columns(data)

    assert "cohort_size" not in feature_cols
    assert "tenure_week" not in feature_cols
    assert "total_visits" not in feature_cols
    assert "log1p_cohort_size" in feature_cols
    assert "log1p_tenure_week" in feature_cols

    scaled = CovariateScaler().fit(data, feature_cols).transform(data)
    for col in ["cohort_size", "tenure_week", "visits", "total_visits", "total_transactions", "total_revenue"]:
        assert scaled[col].tolist() == data[col].tolist()
