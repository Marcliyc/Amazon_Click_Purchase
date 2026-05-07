import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from src.amazon_covariates import build_amazon_covariates
from src.amazon_ml_evcm import PARAM_NAMES, covariate_parameter_head, init_head_params, loss_fn, make_training_data


def _raw_covariates():
    return pd.DataFrame(
        {
            "machine_id": [1, 1, 2, 3],
            "census_region": ["NE", "NE", "West", None],
            "household_size": [2, 2, 4, None],
            "household_income": [5, 5, 8, None],
            "racial_background": ["A", "A", "B", None],
            "country_of_origin": ["US", "US", "US", None],
            "tran_flg": [0, 1, 0, 0],
            "basket_tot": [0.0, 10.0, 0.0, 0.0],
        }
    )


def test_build_amazon_covariates_excludes_outcomes_and_handles_missing():
    X, feature_names, machine_index, metadata, machine_frame = build_amazon_covariates(
        _raw_covariates(), machine_ids=[1, 2, 3], rare_min_count=1
    )
    assert X.shape[0] == 3
    assert X.dtype == np.float32
    assert machine_index[1] == 0
    assert "tran_flg" not in feature_names
    assert "basket_tot" not in feature_names
    assert "Unknown" in metadata.category_levels["census_region"]
    assert machine_frame["machine_id"].tolist() == [1, 2, 3]


def test_covariate_head_zero_coefficients_matches_base_for_all_machines():
    params = init_head_params(n_features=2, seed=1, w_scale=0.0)
    X = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    theta = covariate_parameter_head(params, X, use_covariates=True)
    assert theta.shape == (2, len(PARAM_NAMES))
    assert jnp.allclose(theta[0], theta[1])
    assert jnp.all(theta[:, :8] > 0)
    assert jnp.all((theta[:, -1] > 0) & (theta[:, -1] < 1))


def test_amazon_ml_evcm_loss_jit_grad_finite():
    visits = pd.DataFrame(
        {
            "machine_id": [1, 1, 2, 2],
            "t": [0.0, 2.0, 0.5, 3.0],
            "purchase": [0, 1, 0, 0],
        }
    )
    X = np.zeros((2, 2), dtype=np.float32)
    data = make_training_data(visits, [1, 2], X, T_end=4.0)
    params = init_head_params(n_features=2, seed=1, w_scale=0.0)

    wrapped = lambda p: loss_fn(p, data, 1e-2, True)[0]
    value = jax.jit(wrapped)(params)
    grads = jax.grad(wrapped)(params)
    assert jnp.isfinite(value)
    assert jnp.isfinite(grads["base"]).all()
    assert jnp.isfinite(grads["W"]).all()


def test_load_base_constrained_from_param_csvs(tmp_path):
    from src.amazon_ml_evcm import load_base_constrained_from_csv

    ev = tmp_path / "params_ev.csv"
    cm = tmp_path / "params_cm.csv"
    ev.write_text("param,value\nr,1.1\nalpha,2.2\ns,3.3\nbeta,4.4\n")
    cm.write_text("param,value\nr_v,5.5\nmu0,6.6\nk,0.7\nr_tau,8.8\npsi,-0.1\npi,0.2\n")
    base = load_base_constrained_from_csv(ev, cm)
    assert base["ev_r"] == 1.1
    assert base["ev_beta"] == 4.4
    assert base["cm_mu0"] == 6.6
    assert base["cm_pi"] == 0.2


def test_holdout_forecast_reports_period_and_segment_mapes():
    from src.amazon_ml_evcm import machine_parameter_frame
    from src.train_amazon_ml_evcm import _forecast_holdout_purchases

    cal = pd.DataFrame(
        {
            "machine_id": [1, 1, 2],
            "t": [0.0, 1.0, 0.5],
            "purchase": [0, 1, 0],
            "visit_datetime": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )
    hold = pd.DataFrame(
        {
            "machine_id": [1, 2],
            "t": [3.0, 3.5],
            "purchase": [1, 0],
            "visit_datetime": pd.to_datetime(["2024-02-01", "2024-02-02"]),
        }
    )
    theta = np.asarray(
        [
            [0.5, 7.0, 16.0, 16.0, 1.9, 0.75, 0.73, 22.0, -0.08, 0.21],
            [0.5, 7.0, 16.0, 16.0, 1.9, 0.75, 0.73, 22.0, -0.08, 0.21],
        ],
        dtype=np.float32,
    )
    machine_params = machine_parameter_frame([1, 2], theta)
    machine_params["census_region"] = ["A", "B"]
    scored, by_period, segment_mape, metrics = _forecast_holdout_purchases(cal, hold, machine_params, ["census_region"])
    assert len(scored) == 2
    assert "holdout_incremental_mape" in metrics
    assert not by_period.empty
    assert set(segment_mape["segment_variable"]) == {"census_region"}
