import pandas as pd

from src.segments import identify_early_first_purchase_customers
from src.evaluate import metrics_for_frame


def test_first_four_week_purchasers_are_assigned_to_evcm_segment():
    sessions = pd.DataFrame(
        {
            "machine_id": ["early", "late", "never"],
            "calendar_week": pd.to_datetime(["2024-01-07", "2024-02-11", "2024-01-14"]),
            "transaction": [1, 1, 0],
        }
    )
    cfg = {"data": {"customer_id_col": "machine_id"}, "evcm": {"early_purchase_weeks": 4}}

    segments = identify_early_first_purchase_customers(sessions, cfg).set_index("machine_id")

    assert segments.loc["early", "cbmt_segment"] == "early_purchase_evcm"
    assert segments.loc["late", "cbmt_segment"] == "transformer"
    assert segments.loc["never", "cbmt_segment"] == "transformer"


def test_metrics_can_be_reported_for_segments_and_combined_holdout():
    weekly = pd.DataFrame(
        {
            "actual_total_visits": [10.0],
            "pred_total_visits": [9.0],
            "actual_total_visits_transformer": [7.0],
            "pred_total_visits_transformer": [6.0],
            "actual_total_visits_evcm": [3.0],
            "pred_total_visits_evcm": [3.0],
        }
    )
    metrics = metrics_for_frame(
        weekly,
        {
            "combined_total_visits": ("actual_total_visits", "pred_total_visits"),
            "transformer_total_visits": ("actual_total_visits_transformer", "pred_total_visits_transformer"),
            "evcm_early_purchase_total_visits": ("actual_total_visits_evcm", "pred_total_visits_evcm"),
        },
    )

    assert set(metrics) == {"combined_total_visits", "transformer_total_visits", "evcm_early_purchase_total_visits"}
    assert metrics["evcm_early_purchase_total_visits"]["MAE"] == 0.0
