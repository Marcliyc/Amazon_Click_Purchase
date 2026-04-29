import pandas as pd

from src.pipelines.fit_ebay_choice_cm_jax import aggregate_monthly_purchases, aggregate_monthly_visits


def test_monthly_aggregation_deduplicates_sessions():
    df = pd.DataFrame(
        {
            "machine_id": [1, 1, 1, 2, 2],
            "user_session_id": ["s1", "s1", "s2", "s3", "s3"],
            "event_date": ["2024-01-03", "2024-01-03", "2024-01-21", "2024-02-02", "2024-02-02"],
            "event_time": ["10:00:00", "10:01:00", "11:00:00", "08:00:00", "08:05:00"],
            "tran_flg": [1, 0, 1, 1, 1],
        }
    )
    out = aggregate_monthly_purchases(df, "event_date", "tran_flg", "user_session_id")
    assert list(out["actual_ebay_purchases"]) == [2, 1]


def test_monthly_visit_aggregation_uses_session_dedup():
    df = pd.DataFrame(
        {
            "machine_id": [1, 1, 1],
            "user_session_id": ["a", "a", "b"],
            "event_date": ["2024-01-03", "2024-01-03", "2024-01-15"],
            "event_time": ["10:00:00", "10:01:00", "11:00:00"],
        }
    )
    out = aggregate_monthly_visits(df, "event_date", "user_session_id", output_col="actual_amazon_visits")
    assert int(out.loc[0, "actual_amazon_visits"]) == 2
