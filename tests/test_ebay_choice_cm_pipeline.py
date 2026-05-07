import pandas as pd

from src.pipelines.fit_ebay_choice_cm_jax import _filter_domain, aggregate_monthly_purchases, aggregate_monthly_visits


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


def test_domain_filter_prevents_cross_site_mixing():
    df = pd.DataFrame(
        {
            "domain_name": ["amazon.com", "ebay.com", "ebay.com"],
            "x": [1, 2, 3],
        }
    )
    out = _filter_domain(df, "domain_name", "ebay.com")
    assert len(out) == 2


def test_session_summary_deduplicates_transactions_per_user_session():
    from src.pipelines.forecast_ebay_choice_cm_sessions import build_session_visit_transaction_summary

    df = pd.DataFrame(
        {
            "machine_id": [1, 1, 2],
            "user_session_id": ["s1", "s1", "s2"],
            "event_date": ["2024-01-03", "2024-01-03", "2024-01-15"],
            "tran_flg": [1, 1, 0],
        }
    )
    out = build_session_visit_transaction_summary(df, "event_date", "tran_flg", "user_session_id")
    assert list(out["user_session_id"]) == ["s1", "s2"]
    assert list(out["actual_visits"]) == [1, 1]
    assert list(out["actual_transactions"]) == [1, 0]
    assert list(out["transaction_row_count"]) == [2, 0]


def test_session_scoring_allocates_monthly_forecast_to_sessions():
    from src.ev_beta_choice_cm import EVBetaChoiceCMConfig, init_for_optimizer
    from src.pipelines.forecast_ebay_choice_cm_sessions import score_sessions_with_monthly_forecast

    cfg = EVBetaChoiceCMConfig(
        amazon_fixed={
            "r": 0.457522,
            "alpha": 7.372648,
            "s": 16.440086,
            "beta": 16.369219,
            "r_v": 1.939664,
            "r_tau": 22.712252,
            "psi": -0.080496,
            "pi": 0.208507,
            "mu0": 0.751528,
            "k": 0.730525,
        },
        ebay_init={"pi": 0.208507, "mu0": 0.751528, "k": 0.730525},
        choice={"initial_mean": 0.5, "initial_concentration": 20.0, "fix_concentration": True, "min_concentration": 2.0},
        priors={"lambda_pi": 0.1, "lambda_mu": 0.1, "lambda_k": 0.1},
        fit={"likelihood": "negative_binomial", "obs_scale_init": 20.0, "fit_shared_ev": True},
    )
    sessions = pd.DataFrame(
        {
            "user_session_id": ["a", "b"],
            "machine_id": [1, 2],
            "visit_start": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "visit_end": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "month": pd.to_datetime(["2024-01-31", "2024-01-31"]),
            "actual_visits": [1, 1],
            "actual_transactions": [0, 1],
            "transaction_row_count": [0, 1],
            "raw_row_count": [1, 1],
        }
    )
    out = score_sessions_with_monthly_forecast(sessions, init_for_optimizer(cfg), cfg, 2.0, "2024-01-01", "2024-01-31")
    assert "model_expected_transactions" in out.columns
    assert out["month_observed_sessions"].tolist() == [2, 2]
    assert out["model_expected_visits"].notna().all()
