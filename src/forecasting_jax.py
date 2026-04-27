from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .cm_model import CMState, cm_customer_loglik_and_state, cm_purchase_probability
from .ev_model import EVState, ev_customer_loglik_and_state


def _period_bins(T_cal_end: float, T_holdout_end: float, freq_days: float = 7.0):
    edges = np.arange(T_cal_end, T_holdout_end + freq_days + 1e-9, freq_days)
    if edges[-1] < T_holdout_end:
        edges = np.append(edges, T_holdout_end)
    return edges


def simulate_evcm_forecast_jax(
    visits_cal: pd.DataFrame,
    T_cal_end: float,
    T_holdout_end: float,
    ev_params,
    cm_params,
    n_sims: int = 1000,
    freq: str = "W",
    seed: int = 123,
):
    known = sorted(visits_cal["machine_id"].unique())
    freq_days = {"D": 1.0, "W": 7.0, "M": 30.0}.get(freq, 7.0)
    bins = _period_bins(T_cal_end, T_holdout_end, freq_days)
    sim_mat = np.zeros((n_sims, len(bins) - 1), dtype=float)
    sim_visits = np.zeros(n_sims, dtype=float)
    cal_groups = {k: g.sort_values("t") for k, g in visits_cal.groupby("machine_id")}

    key = jax.random.PRNGKey(seed)
    for s in range(n_sims):
        key, sim_key = jax.random.split(key)
        period_counts = np.zeros(len(bins) - 1, dtype=float)
        total_visits = 0
        cust_key = sim_key

        for mid in known:
            g = cal_groups[mid]
            _, ev_state = ev_customer_loglik_and_state(g["t"].to_numpy(), T_cal_end, ev_params, return_state_at_end=True)
            _, cm_state = cm_customer_loglik_and_state(g["purchase"].to_numpy(), cm_params)
            cust_key, k_lam = jax.random.split(cust_key)
            lam = float(jax.random.gamma(k_lam, ev_state.r) / ev_state.alpha)

            t = T_cal_end
            state = CMState(**vars(cm_state))
            ev_live = EVState(r=ev_state.r, alpha=ev_state.alpha, last_t=ev_state.last_t, total_visits_seen=ev_state.total_visits_seen)

            while True:
                cust_key, k_gap = jax.random.split(cust_key)
                gap = float(jax.random.exponential(k_gap) / max(lam, 1e-12))
                t_next = t + gap
                if t_next > T_holdout_end:
                    break
                total_visits += 1

                visit_index = state.total_visits + 1
                p = cm_purchase_probability(visit_index, state.prior_purchases, state.last_purchase_visit_index, cm_params)
                cust_key, k_buy = jax.random.split(cust_key)
                buy = int(jax.random.bernoulli(k_buy, p=jnp.asarray(p)))

                state.total_visits += 1
                if buy:
                    state.prior_purchases += 1
                    state.last_purchase_visit_index = state.total_visits
                    b = np.digitize([t_next], bins)[0] - 1
                    if 0 <= b < len(period_counts):
                        period_counts[b] += 1

                cust_key, k_c = jax.random.split(cust_key)
                c = float(jax.random.gamma(k_c, ev_params.s) / ev_params.beta)
                lam *= c
                t = t_next
                ev_live.total_visits_seen += 1
                ev_live.last_t = t_next

        sim_mat[s] = period_counts
        sim_visits[s] = total_visits

    q = np.quantile(sim_mat, [0.05, 0.5, 0.95], axis=0)
    out = pd.DataFrame(
        {
            "period_idx": np.arange(len(bins) - 1),
            "period_start_t": bins[:-1],
            "period_end_t": bins[1:],
            "forecast_mean_purchases": sim_mat.mean(axis=0),
            "forecast_p05_purchases": q[0],
            "forecast_p50_purchases": q[1],
            "forecast_p95_purchases": q[2],
        }
    )
    out["forecast_mean_cum_purchases"] = out["forecast_mean_purchases"].cumsum()
    out["forecast_p05_cum_purchases"] = out["forecast_p05_purchases"].cumsum()
    out["forecast_p50_cum_purchases"] = out["forecast_p50_purchases"].cumsum()
    out["forecast_p95_cum_purchases"] = out["forecast_p95_purchases"].cumsum()
    out.attrs["forecast_holdout_visits_mean"] = float(sim_visits.mean())
    return out
