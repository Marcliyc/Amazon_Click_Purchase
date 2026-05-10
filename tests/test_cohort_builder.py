import pandas as pd

from src.data_preprocess import clean_raw_amazon, aggregate_to_sessions
from src.cohort_builder import build_cohort_week_panel


def cfg():
    return {"data": {"customer_id_col": "machine_id", "session_id_col": "session_id", "date_col": "event_date", "transaction_flag_col": "tran_flg", "payment_col": "totalprice", "pages_col": "pages_viewed", "duration_col": "duration", "covariate_cols": [], "cohort_definition": "first_visit", "page_agg": "max", "duration_agg": "max"}, "split": {"week_start": "SUN"}}


def test_cohort_builder_balanced_panel_and_metrics():
    raw = pd.DataFrame({
        "machine_id": ["a", "a", "b", "c", "c"],
        "session_id": [1, 2, 3, 4, 5],
        "event_date": ["2024-01-07", "2024-01-14", "2024-01-14", "2024-01-28", "2024-01-28"],
        "tran_flg": [0, 1, 1, 0, 1],
        "totalprice": [0, 10, 20, 0, 5],
        "pages_viewed": [1, 2, 3, 1, 1], "duration": [5, 6, 7, 1, 1],
    })
    cleaned, _ = clean_raw_amazon(raw, cfg())
    sessions = aggregate_to_sessions(cleaned, cfg())
    panel, agg, cohorts = build_cohort_week_panel(sessions, cfg())
    assert cohorts.set_index("machine_id").loc["a", "cohort_week"] == pd.Timestamp("2024-01-07")
    row = panel[(panel.cohort_week == pd.Timestamp("2024-01-07")) & (panel.calendar_week == pd.Timestamp("2024-01-14"))].iloc[0]
    assert row.visits == 1
    assert row.transactions == 1
    assert row.revenue == 10
    assert row.avg_payment == 10
    zero = panel[(panel.cohort_week == pd.Timestamp("2024-01-14")) & (panel.calendar_week == pd.Timestamp("2024-01-21"))].iloc[0]
    assert zero.visits == 0
    assert agg.loc[agg.calendar_week == pd.Timestamp("2024-01-14"), "total_revenue"].iloc[0] == 30
