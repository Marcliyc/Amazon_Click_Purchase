import pandas as pd

from src.forecast import add_predicted_cold_cohorts


def test_non_oracle_cold_cohorts_continue_across_holdout_weeks():
    panel = pd.DataFrame(
        {
            "cohort_week": pd.to_datetime(["2024-01-07", "2024-01-14"]),
            "calendar_week": pd.to_datetime(["2024-01-07", "2024-01-14"]),
            "cohort_size": [10.0, 20.0],
            "tenure_week": [0, 0],
            "visits": [1.0, 2.0],
        }
    )
    holdout_weeks = pd.to_datetime(["2024-01-21", "2024-01-28", "2024-02-04"]).tolist()

    expanded = add_predicted_cold_cohorts(panel, holdout_weeks)
    cold = expanded[expanded["cohort_week"].isin(holdout_weeks)].sort_values(["cohort_week", "calendar_week"])

    assert len(cold) == 6
    assert set(zip(cold["cohort_week"], cold["calendar_week"])) == {
        (pd.Timestamp("2024-01-21"), pd.Timestamp("2024-01-21")),
        (pd.Timestamp("2024-01-21"), pd.Timestamp("2024-01-28")),
        (pd.Timestamp("2024-01-21"), pd.Timestamp("2024-02-04")),
        (pd.Timestamp("2024-01-28"), pd.Timestamp("2024-01-28")),
        (pd.Timestamp("2024-01-28"), pd.Timestamp("2024-02-04")),
        (pd.Timestamp("2024-02-04"), pd.Timestamp("2024-02-04")),
    }
    first_cold = cold[cold["cohort_week"] == pd.Timestamp("2024-01-21")]
    assert first_cold["tenure_week"].tolist() == [0, 1, 2]
    assert first_cold["cohort_size"].tolist() == [15.0, 15.0, 15.0]
