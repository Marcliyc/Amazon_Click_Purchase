import pandas as pd
import pytest
pytest.importorskip("torch")
from src.dataset import WindowedCohortDataset, merge_aggregate
from src.covariates import attach_covariates, numeric_feature_columns


def test_windowed_dataset_shapes():
    weeks = pd.date_range("2024-01-07", periods=5, freq="7D")
    panel = pd.DataFrame({"cohort_week": [weeks[0]] * 5, "calendar_week": weeks, "cohort_size": [2]*5, "tenure_week": range(5), "visits": [2,0,1,1,1], "transactions": [0,1,0,1,1], "revenue": [0,10,0,12,8]})
    panel["visits_per_customer"] = panel.visits / panel.cohort_size
    panel["transactions_per_customer"] = panel.transactions / panel.cohort_size
    panel["avg_payment"] = panel.revenue / panel.transactions.where(panel.transactions > 0, 1)
    panel["revenue_per_customer"] = panel.revenue / panel.cohort_size
    agg = pd.DataFrame({"calendar_week": weeks, "total_visits": [2,0,1,1,1], "total_transactions": [0,1,0,1,1], "total_revenue": [0,10,0,12,8]})
    data = attach_covariates(merge_aggregate(panel, agg))
    features = numeric_feature_columns(data)
    ds = WindowedCohortDataset(data, features, [weeks[-1]], lookback_weeks=3)
    item = ds[0]
    assert item["x_seq"].shape[0] == 3
    assert item["x_seq"].shape[1] > 0
    assert item["y"].shape[0] == 6
    assert item["consistency"].shape[0] == 4
