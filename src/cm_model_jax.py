from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxopt import LBFGS

EPS = 1e-12
MAX_EFFECT = 1e12


@dataclass
class CMJaxParams:
    r_v: float
    mu0: float
    k: float
    r_tau: float
    psi: float
    pi: float


def softplus(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0)


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (1.0 + jnp.exp(-x))


def _params_from_theta(theta: jnp.ndarray) -> jnp.ndarray:
    r_v = softplus(theta[0]) + 1e-6
    mu0 = softplus(theta[1]) + 1e-6
    k = jnp.exp(jnp.clip(theta[2], -5.0, 5.0))
    r_tau = softplus(theta[3]) + 1e-6
    psi = theta[4]
    #pi = jnp.clip(sigmoid(theta[5]), 1e-6, 1.0 - 1e-6)
    pi = jnp.clip(theta[5], 0.05, 0.97)
    return jnp.array([r_v, mu0, k, r_tau, psi, pi])


def _to_padded_sequences(visits_cal: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    groups = [g.sort_values("t")["purchase"].to_numpy(dtype=int) for _, g in visits_cal.groupby("machine_id")]
    if not groups:
        return np.zeros((0, 0), dtype=int), np.zeros((0,), dtype=np.int32)
    lengths = np.asarray([len(g) for g in groups], dtype=np.int32)
    max_len = int(lengths.max())
    padded = np.zeros((len(groups), max_len), dtype=int)
    for i, seq in enumerate(groups):
        padded[i, : len(seq)] = seq
    return padded, lengths


def _geometric_visit_effect(mu0: jnp.ndarray, k: jnp.ndarray, start: jnp.ndarray, end: jnp.ndarray) -> jnp.ndarray:
    n = end - start + 1

    def k_is_one(_: None):
        return mu0 * n

    def k_not_one(_: None):
        # Stable log-domain evaluation of:
        #   mu0 * k**start * (k**n - 1) / (k - 1)
        # This avoids overflow when k > 1 and start/n are large.
        logk = jnp.log(jnp.maximum(k, EPS))
        num = jnp.expm1(n * logk)
        den = jnp.expm1(logk)
        sign = jnp.sign(num) * jnp.sign(den)
        log_abs = jnp.log(mu0 + EPS) + start * logk + jnp.log(jnp.abs(num) + EPS) - jnp.log(jnp.abs(den) + EPS)
        effect = sign * jnp.exp(jnp.clip(log_abs, -80.0, 80.0))
        return jnp.clip(effect, 0.0, MAX_EFFECT)

    return jax.lax.cond(jnp.abs(k - 1.0) < 1e-8, k_is_one, k_not_one, operand=None)


def _cm_purchase_probability(visit_idx: jnp.ndarray, prior_purchases: jnp.ndarray, last_purchase_visit_idx: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    r_v, mu0, k, r_tau, psi, pi = p
    n_ij = visit_idx - 1
    a_ij = r_v + _geometric_visit_effect(mu0, k, last_purchase_visit_idx + 1, visit_idx)
    b_ij = r_tau * jnp.exp(psi * prior_purchases)

    p0 = a_ij / jnp.maximum(a_ij + b_ij + n_ij, EPS)
    no_prev = pi * p0
    has_prev = (a_ij + prior_purchases) / jnp.maximum(a_ij + b_ij + n_ij, EPS)
    pbuy = jnp.where(prior_purchases == 0, no_prev, has_prev)
    return jnp.clip(pbuy, EPS, 1.0 - EPS)


def _customer_loglik(seq_row: jnp.ndarray, n: jnp.int32, p: jnp.ndarray) -> jnp.ndarray:
    def step(carry, y):
        total_visits, prior_purchases, last_purchase_idx, ll = carry
        visit_idx = total_visits + 1
        prob = _cm_purchase_probability(visit_idx, prior_purchases, last_purchase_idx, p)

        p0 = prob / jnp.maximum(p[5], EPS)
        ll_y0 = jnp.where(
            prior_purchases == 0,
            jnp.log((1.0 - p[5]) + p[5] * (1.0 - p0) + EPS),
            jnp.log(1.0 - prob + EPS),
        )
        ll = ll + jnp.where(y == 1, jnp.log(prob + EPS), ll_y0)

        total_visits = total_visits + 1
        bought = y == 1
        prior_purchases = jnp.where(bought, prior_purchases + 1, prior_purchases)
        last_purchase_idx = jnp.where(bought, total_visits, last_purchase_idx)
        return (total_visits, prior_purchases, last_purchase_idx, ll), None

    init = (jnp.array(0), jnp.array(0), jnp.array(0), jnp.array(0.0))

    def step_masked(carry, inputs):
        i, y = inputs
        def active_step(c):
            return step(c, y)[0]
        new_carry = jax.lax.cond(i < n, active_step, lambda c: c, carry)
        return new_carry, None

    idx = jnp.arange(seq_row.shape[0])
    (tv, pp, lp, ll), _ = jax.lax.scan(step_masked, init, (idx, seq_row))
    return ll


def cm_loglik_jax(theta: jnp.ndarray, purchases: jnp.ndarray, lengths: jnp.ndarray) -> jnp.ndarray:
    p = _params_from_theta(theta)
    ll_by_customer = jax.vmap(_customer_loglik, in_axes=(0, 0, None))(purchases, lengths, p)
    return ll_by_customer.sum()


def fit_cm_model_jax(visits_cal: pd.DataFrame, n_starts: int = 30, seed: int = 123):
    p_np, lengths_np = _to_padded_sequences(visits_cal)
    purchases = jnp.asarray(p_np)
    lengths = jnp.asarray(lengths_np)
    ever_purchase = visits_cal.groupby("machine_id")["purchase"].max().mean()

    objective = lambda th: -cm_loglik_jax(th, purchases, lengths)
    solver = LBFGS(fun=objective, maxiter=600, tol=1e-7)
    rng = np.random.default_rng(seed)
    best_params = None
    best_state = None
    best_value = np.inf

    for i in range(n_starts):
        print(f"CM fit {i} start", flush=True)
        pi0 = np.clip(ever_purchase, 0.05, 0.95)
        x0 = np.array([
            rng.normal(1, 1),
            rng.normal(1, 1),
            rng.normal(0, 0.2),
            rng.normal(1.5, 1),
            rng.normal(0, 0.2),
            rng.normal(0.5, 0.3),
            #np.log(pi0 / (1 - pi0)),
        ])
        params, state = solver.run(jnp.asarray(x0))
        value = float(state.value)
        if value < best_value:
            best_value = value
            best_params = params
            best_state = state

    params_arr = np.asarray(_params_from_theta(best_params))
    params = CMJaxParams(*(float(v) for v in params_arr))
    converged = bool(float(best_state.error) <= 1e-7 and int(best_state.iter_num) < 600)
    info = {
        "converged": converged,
        "message": f"jaxopt.LBFGS error={float(best_state.error):.3e}",
        "objective": best_value,
        "nit": int(best_state.iter_num),
    }
    return params, info
