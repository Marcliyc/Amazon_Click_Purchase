from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxopt import LBFGS

EPS = 1e-9


@dataclass
class EVJaxParams:
    r: float
    alpha: float
    s: float
    beta: float


def softplus(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0)


def _params_from_theta(theta: jnp.ndarray) -> jnp.ndarray:
    return softplus(theta) + 1e-6


def _to_padded_times(visits_cal: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    groups = [g["t"].to_numpy(dtype=float) for _, g in visits_cal.groupby("machine_id")]
    if not groups:
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=np.int32)
    lengths = np.asarray([len(g) for g in groups], dtype=np.int32)
    max_len = int(lengths.max())
    padded = np.zeros((len(groups), max_len), dtype=float)
    for i, arr in enumerate(groups):
        padded[i, : len(arr)] = np.sort(arr)
    return padded, lengths


def _customer_loglik(times_row: jnp.ndarray, n: jnp.int32, T_end: float, p: jnp.ndarray) -> jnp.ndarray:
    r0, a0, s, beta = p

    def no_visits(_: None) -> jnp.ndarray:
        return jnp.array(0.0)

    def has_visits(_: None) -> jnp.ndarray:
        first_t = times_row[0]

        def body(carry, inputs):
            i, t_next = inputs
            r_cur, a_cur, t_prev, ll = carry
            active = i < (n - 1)
            d = jnp.maximum(t_next - t_prev, EPS)
            ll_inc = jnp.log(r_cur + EPS) + r_cur * jnp.log(a_cur + EPS) - (r_cur + 1.0) * jnp.log(a_cur + d + EPS)
            r_arr = r_cur + 1.0
            a_arr = a_cur + d
            denom = jnp.maximum(r_arr + s + 1.0, EPS)
            r_new = (r_arr * s) / denom
            a_new = (a_arr * beta) / denom
            r_cur = jnp.where(active, r_new, r_cur)
            a_cur = jnp.where(active, a_new, a_cur)
            t_prev = jnp.where(active, t_next, t_prev)
            ll = jnp.where(active, ll + ll_inc, ll)
            return (r_cur, a_cur, t_prev, ll), None

        idx = jnp.arange(times_row.shape[0] - 1)
        init = (r0, a0, first_t, jnp.array(0.0))
        (r_fin, a_fin, t_last, ll), _ = jax.lax.scan(body, init, (idx, times_row[1:]))
        c = jnp.maximum(T_end - jnp.where(n > 0, t_last, 0.0), 0.0)
        ll = ll + r_fin * (jnp.log(a_fin + EPS) - jnp.log(a_fin + c + EPS))
        return ll

    return jax.lax.cond(n == 0, no_visits, has_visits, operand=None)


def ev_loglik_jax(theta: jnp.ndarray, times: jnp.ndarray, lengths: jnp.ndarray, T_end: float) -> jnp.ndarray:
    p = _params_from_theta(theta)
    ll_by_customer = jax.vmap(_customer_loglik, in_axes=(0, 0, None, None))(times, lengths, T_end, p)
    return ll_by_customer.sum()


def fit_ev_model_jax(visits_cal: pd.DataFrame, T_cal_end: float, n_starts: int = 20, seed: int = 123):
    times_np, lengths_np = _to_padded_times(visits_cal)
    times = jnp.asarray(times_np)
    lengths = jnp.asarray(lengths_np)

    objective = lambda th: -ev_loglik_jax(th, times, lengths, T_cal_end)
    solver = LBFGS(fun=objective, maxiter=500, tol=1e-7)
    rng = np.random.default_rng(seed)
    best_params = None
    best_state = None
    best_value = np.inf

    for i in range(n_starts):
        print(f"EV fit {i} start", flush=True)
        x0 = rng.normal(0, 1, 4)
        params, state = solver.run(jnp.asarray(x0))
        value = float(state.value)
        if value < best_value:
            best_value = value
            best_params = params
            best_state = state

    params_arr = np.asarray(_params_from_theta(best_params))
    params = EVJaxParams(*(float(v) for v in params_arr))
    converged = bool(float(best_state.error) <= 1e-7 and int(best_state.iter_num) < 500)
    info = {
        "converged": converged,
        "message": f"jaxopt.LBFGS error={float(best_state.error):.3e}",
        "objective": best_value,
        "nit": int(best_state.iter_num),
    }
    return params, info
