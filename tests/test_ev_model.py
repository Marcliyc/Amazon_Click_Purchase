import numpy as np
import pandas as pd

from src.ev_model import EVParams, ev_customer_loglik_and_state, ev_loglik


def test_ev_loglik_finite():
    params = EVParams(1.0, 1.0, 1.0, 1.0)
    times = np.array([0.0, 2.0, 5.0])
    ll, state = ev_customer_loglik_and_state(times, 8.0, params)
    assert np.isfinite(ll)
    assert state.alpha > 0

    visits = pd.DataFrame({"machine_id": [1, 1, 1], "t": times})
    all_ll = ev_loglik(visits, 8.0, params)
    assert np.isfinite(all_ll)
