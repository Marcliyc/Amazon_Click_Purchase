from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .cm_model import cm_customer_loglik_and_state
from .cm_model_jax import _cm_purchase_probability
from .ev_model import ev_customer_loglik_and_state


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
    max_steps_per_customer: int | None = None,
):
    """
    Accelerator-friendly JAX forecast simulation.

    Uses JAX `jit` + `vmap` + `lax.fori_loop` so the heavy simulation work runs in XLA
    (GPU/TPU/CPU accelerator backends), avoiding Python-side inner loops.
    """
    if len(visits_cal) == 0:
        raise ValueError("visits_cal is empty; cannot simulate forecast")

    freq_days = {"D": 1.0, "W": 7.0, "M": 30.0}.get(freq, 7.0)
    bins = _period_bins(T_cal_end, T_holdout_end, freq_days)
    n_periods = int(len(bins) - 1)

    grouped = {k: g.sort_values("t") for k, g in visits_cal.groupby("machine_id")}
    mids = sorted(grouped)

    ev_r, ev_a = [], []
    cm_total, cm_prior, cm_last = [], [], []
    for mid in mids:
        g = grouped[mid]
        _, ev_state = ev_customer_loglik_and_state(g["t"].to_numpy(), T_cal_end, ev_params, return_state_at_end=True)
        _, cm_state = cm_customer_loglik_and_state(g["purchase"].to_numpy(), cm_params)
        ev_r.append(ev_state.r)
        ev_a.append(ev_state.alpha)
        cm_total.append(cm_state.total_visits)
        cm_prior.append(cm_state.prior_purchases)
        cm_last.append(cm_state.last_purchase_visit_index)

    ev_r = jnp.asarray(np.asarray(ev_r, dtype=float))
    ev_a = jnp.asarray(np.asarray(ev_a, dtype=float))
    cm_total = jnp.asarray(np.asarray(cm_total, dtype=np.int32))
    cm_prior = jnp.asarray(np.asarray(cm_prior, dtype=np.int32))
    cm_last = jnp.asarray(np.asarray(cm_last, dtype=np.int32))

    if max_steps_per_customer is None:
        # Conservative upper bound for fixed-step simulation loops.
        hist_sizes = visits_cal.groupby("machine_id").size().to_numpy()
        max_steps_per_customer = int(max(64, np.percentile(hist_sizes, 99) * 8))

    use_x64 = bool(jax.config.read("jax_enable_x64"))
    fp_dtype = jnp.float64 if use_x64 else jnp.float32

    ev_param_vec = jnp.asarray([ev_params.r, ev_params.alpha, ev_params.s, ev_params.beta], dtype=fp_dtype)
    cm_param_vec = jnp.asarray([cm_params.r_v, cm_params.mu0, cm_params.k, cm_params.r_tau, cm_params.psi, cm_params.pi], dtype=fp_dtype)

    @jax.jit
    def _simulate_customer(key, r0, a0, total0, prior0, last0):
        key, k0 = jax.random.split(key)
        lam0 = jax.random.gamma(k0, r0) / jnp.maximum(a0, 1e-12)

        init = (
            key,
            jnp.asarray(T_cal_end, dtype=fp_dtype),
            lam0,
            total0,
            prior0,
            last0,
            jnp.ones((), dtype=jnp.bool_),
            jnp.zeros((n_periods,), dtype=fp_dtype),
            jnp.asarray(0.0, dtype=fp_dtype),
        )

        def step(_, carry):
            key, t, lam, total, prior, last, alive, counts, visits = carry

            key, k_gap = jax.random.split(key)
            gap = jax.random.exponential(k_gap) / jnp.maximum(lam, 1e-12)
            t_next = t + gap
            active = alive & (t_next <= T_holdout_end)

            visit_idx = total + 1
            p_buy = _cm_purchase_probability(visit_idx, prior, last, cm_param_vec)
            key, k_buy = jax.random.split(key)
            buy = jax.random.bernoulli(k_buy, p_buy) & active

            period_idx = jnp.floor((t_next - T_cal_end) / freq_days).astype(jnp.int32)
            period_idx = jnp.clip(period_idx, 0, n_periods - 1)
            add = jax.nn.one_hot(period_idx, n_periods, dtype=fp_dtype) * buy.astype(fp_dtype)
            counts = counts + add

            total = jnp.where(active, total + 1, total)
            prior = jnp.where(buy, prior + 1, prior)
            last = jnp.where(buy, total, last)
            visits = visits + active.astype(fp_dtype)

            key, k_c = jax.random.split(key)
            c = jax.random.gamma(k_c, ev_param_vec[2]) / jnp.maximum(ev_param_vec[3], 1e-12)
            lam = jnp.where(active, lam * c, lam)
            t = jnp.where(active, t_next, t)
            alive = active

            return (key, t, lam, total, prior, last, alive, counts, visits)

        out = jax.lax.fori_loop(0, int(max_steps_per_customer), step, init)
        return out[7], out[8]

    @jax.jit
    def _simulate_path(sim_key):
        cust_keys = jax.random.split(sim_key, ev_r.shape[0])
        counts, visits = jax.vmap(_simulate_customer)(cust_keys, ev_r, ev_a, cm_total, cm_prior, cm_last)
        return counts.sum(axis=0), visits.sum()

    keys = jax.random.split(jax.random.PRNGKey(seed), n_sims)
    sim_counts, sim_visits = jax.vmap(_simulate_path)(keys)

    sim_mat = np.asarray(sim_counts)
    sim_visits_np = np.asarray(sim_visits)
    q = np.quantile(sim_mat, [0.05, 0.5, 0.95], axis=0)

    out = pd.DataFrame(
        {
            "period_idx": np.arange(n_periods),
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
    out.attrs["forecast_holdout_visits_mean"] = float(sim_visits_np.mean())
    return out
