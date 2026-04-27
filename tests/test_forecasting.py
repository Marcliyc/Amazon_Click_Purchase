import pandas as pd

from src.cm_model import CMParams
from src.ev_model import EVParams
from src.forecasting import aggregate_actual_holdout, simulate_evcm_forecast


def test_forecast_shapes():
    cal = pd.DataFrame(
        {
            "machine_id": [1, 1, 2],
            "t": [0.0, 2.0, 1.0],
            "purchase": [0, 1, 0],
        }
    )
    hold = pd.DataFrame({"machine_id": [1, 2], "t": [5.0, 6.0], "purchase": [1, 0]})
    ev = EVParams(1.0, 1.0, 1.0, 1.0)
    cm = CMParams(0.3, 0.2, 1.0, 2.0, 0.0, 0.7)
    sim = simulate_evcm_forecast(cal, 2.0, 8.0, ev, cm, n_sims=20)
    act = aggregate_actual_holdout(hold, 2.0, 8.0)
    assert len(sim) == len(act)
    assert "forecast_mean_cum_purchases" in sim.columns
