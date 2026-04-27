from __future__ import annotations

import numpy as np
import pandas as pd

from .cm_model import CMParams, CMState, cm_customer_loglik_and_state, cm_purchase_probability
from .ev_model import EVParams, ev_customer_loglik_and_state


def _period_bins(T_cal_end: float, T_holdout_end: float, freq_days: float = 7.0):
    edges = np.arange(T_cal_end, T_holdout_end + freq_days + 1e-9, freq_days)
    if edges[-1] < T_holdout_end:
        edges = np.append(edges, T_holdout_end)
    return edges


def aggregate_actual_holdout(visits_holdout_known: pd.DataFrame, T_cal_end: float, T_holdout_end: float, freq: str = "W"):
    freq_days = {"D": 1.0, "W": 7.0, "M": 30.0}.get(freq, 7.0)
    bins = _period_bins(T_cal_end, T_holdout_end, freq_days)
    t = visits_holdout_known["t"].to_numpy()
    y = visits_holdout_known["purchase"].to_numpy()
    idx = np.digitize(t, bins) - 1
    rows = []
    for i in range(len(bins) - 1):
        mask = idx == i
        rows.append({"period_idx": i, "period_start_t": bins[i], "period_end_t": bins[i + 1], "actual_purchases": int(y[mask].sum())})
    df = pd.DataFrame(rows)
    df["actual_cum_purchases"] = df["actual_purchases"].cumsum()
    return df


def simulate_evcm_forecast(visits_cal, T_cal_end: float, T_holdout_end: float, ev_params: EVParams, cm_params: CMParams, n_sims: int = 1000, freq: str = "W", seed: int = 123):
    rng = np.random.default_rng(seed)
    known = sorted(visits_cal["machine_id"].unique())
    freq_days = {"D": 1.0, "W": 7.0, "M": 30.0}.get(freq, 7.0)
    bins = _period_bins(T_cal_end, T_holdout_end, freq_days)
    sim_mat = np.zeros((n_sims, len(bins) - 1), dtype=float)
    sim_visits = np.zeros(n_sims, dtype=float)

    cal_groups = {k: g.sort_values("t") for k, g in visits_cal.groupby("machine_id")}
    for s in range(n_sims):
        period_counts = np.zeros(len(bins) - 1, dtype=float)
        total_visits = 0
        for mid in known:
            g = cal_groups[mid]
            _, ev_state = ev_customer_loglik_and_state(g["t"].to_numpy(), T_cal_end, ev_params, return_state_at_end=True)
            _, cm_state = cm_customer_loglik_and_state(g["purchase"].to_numpy(), cm_params)
            lam = rng.gamma(shape=ev_state.r, scale=1.0 / ev_state.alpha)
            t = T_cal_end
            state = CMState(**vars(cm_state))

            while True:
                gap = rng.exponential(scale=1.0 / max(lam, 1e-12))
                t_next = t + gap
                if t_next > T_holdout_end:
                    break
                total_visits += 1
                visit_index = state.total_visits + 1
                p = cm_purchase_probability(visit_index, state.prior_purchases, state.last_purchase_visit_index, cm_params)
                buy = int(rng.random() < p)
                state.total_visits += 1
                if buy:
                    state.prior_purchases += 1
                    state.last_purchase_visit_index = state.total_visits
                    b = np.digitize([t_next], bins)[0] - 1
                    if 0 <= b < len(period_counts):
                        period_counts[b] += 1
                c = rng.gamma(shape=ev_params.s, scale=1.0 / ev_params.beta)
                lam *= c
                t = t_next
        sim_mat[s] = period_counts
        sim_visits[s] = total_visits

    q = np.quantile(sim_mat, [0.05, 0.5, 0.95], axis=0)
    out = pd.DataFrame({
        "period_idx": np.arange(len(bins) - 1),
        "period_start_t": bins[:-1],
        "period_end_t": bins[1:],
        "forecast_mean_purchases": sim_mat.mean(axis=0),
        "forecast_p05_purchases": q[0],
        "forecast_p50_purchases": q[1],
        "forecast_p95_purchases": q[2],
    })
    out["forecast_mean_cum_purchases"] = out["forecast_mean_purchases"].cumsum()
    out["forecast_p05_cum_purchases"] = out["forecast_p05_purchases"].cumsum()
    out["forecast_p50_cum_purchases"] = out["forecast_p50_purchases"].cumsum()
    out["forecast_p95_cum_purchases"] = out["forecast_p95_purchases"].cumsum()
    out.attrs["forecast_holdout_visits_mean"] = float(sim_visits.mean())
    return out
