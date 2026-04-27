import pandas as pd

from src.cm_model_jax import fit_cm_model_jax
from src.ev_model_jax import fit_ev_model_jax
from src.forecasting_jax import simulate_evcm_forecast_jax


def _toy_calibration_df():
    return pd.DataFrame(
        {
            "machine_id": [1, 1, 1, 2, 2],
            "t": [0.0, 2.0, 5.0, 1.0, 4.0],
            "purchase": [0, 1, 0, 0, 1],
        }
    )


def test_jax_fit_and_forecast_smoke():
    cal = _toy_calibration_df()
    ev_params, ev_info = fit_ev_model_jax(cal, T_cal_end=5.0, n_starts=1, seed=7)
    cm_params, cm_info = fit_cm_model_jax(cal, n_starts=1, seed=7)

    assert isinstance(ev_info["objective"], float)
    assert isinstance(cm_info["objective"], float)

    sim = simulate_evcm_forecast_jax(
        cal,
        T_cal_end=5.0,
        T_holdout_end=8.0,
        ev_params=ev_params,
        cm_params=cm_params,
        n_sims=5,
        seed=3,
    )

    assert "forecast_mean_cum_purchases" in sim.columns
    assert len(sim) > 0
