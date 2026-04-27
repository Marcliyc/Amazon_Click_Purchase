import pandas as pd

from src.data_prep import load_raw_data, make_daily_visits, make_session_time_visits, split_calibration_holdout


def test_split_and_aggregations():
    df = pd.DataFrame(
        {
            "machine_id": [1, 1, 1, 2],
            "site_session_id": [10, 11, 12, 20],
            "visit_datetime": pd.to_datetime(["2024-01-01", "2024-01-01 10:00", "2024-01-03", "2024-01-04"], format="mixed"),
            "visit_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-03", "2024-01-04"]).date,
            "purchase": [0, 1, 0, 1],
            "purchase_session_count": [0, 1, 0, 1],
            "pages_viewed": [1, 2, 1, 2],
            "duration": [1, 2, 1, 2],
            "basket_tot": [0, 10, 0, 5],
            "prod_totprice": [0, 10, 0, 5],
        }
    )
    daily = make_daily_visits(df)
    assert "t" in daily.columns
    sess = make_session_time_visits(df)
    assert sess["t"].is_monotonic_increasing
    cal, hold, _ = split_calibration_holdout(daily, cutoff="2024-01-02")
    assert len(cal) > 0 and len(hold) > 0



def test_purchase_row_uses_tran_flag_only(tmp_path):
    p = tmp_path / "mini.csv"
    p.write_text(
        "machine_id,site_session_id,event_date,event_time,tran_flg,basket_tot,prod_totprice,prod_qty,pages_viewed,duration\n"
        "1,10,2024-01-01,00:00:00,,100,100,1,1,1\n"
        "1,11,2024-01-02,00:00:00,1,,,,1,1\n"
    )
    df = load_raw_data(p)
    assert df.loc[df["site_session_id"] == 10, "purchase_row"].iloc[0] == 0
    assert df.loc[df["site_session_id"] == 11, "purchase_row"].iloc[0] == 1
