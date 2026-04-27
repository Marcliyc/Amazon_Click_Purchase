import numpy as np
import pandas as pd

from src.cm_model import CMParams, cm_customer_loglik_and_state, cm_loglik, cm_purchase_probability


def test_cm_prob_and_loglik():
    p = CMParams(r_v=0.4, mu0=0.2, k=1.0, r_tau=3.0, psi=0.0, pi=0.8)
    prob = cm_purchase_probability(1, 0, 0, p)
    assert 0 < prob < 1
    ll, state = cm_customer_loglik_and_state(np.array([0, 1, 0]), p)
    assert np.isfinite(ll)
    assert state.total_visits == 3

    v = pd.DataFrame({"machine_id": [1, 1, 1], "t": [0, 1, 2], "purchase": [0, 1, 0]})
    assert np.isfinite(cm_loglik(v, p))
