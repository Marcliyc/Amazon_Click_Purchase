from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

EPS = 1e-12


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def inv_softplus(y: float) -> float:
    y = max(float(y), 1e-6)
    return float(np.log(np.expm1(y)))


@dataclass
class CMParams:
    r_v: float
    mu0: float
    k: float
    r_tau: float
    psi: float
    pi: float


@dataclass
class CMState:
    total_visits: int = 0
    prior_purchases: int = 0
    last_purchase_visit_index: int = 0


def geometric_visit_effect(mu0: float, k: float, start: int, end: int) -> float:
    if end < start:
        return 0.0
    n = end - start + 1
    if abs(k - 1.0) < 1e-8:
        return mu0 * n
    logk = np.log(max(k, 1e-12))
    if abs(logk * n) > 50:
        idx = np.arange(start, end + 1)
        return float(np.sum(mu0 * np.exp(np.clip(logk * idx, -50, 50))))
    return float(mu0 * (k**start) * ((k**n) - 1.0) / (k - 1.0))


def cm_purchase_probability(visit_index: int, prior_purchases: int, last_purchase_visit_index: int, params: CMParams) -> float:
    n_ij = visit_index - 1
    a_ij = params.r_v + geometric_visit_effect(params.mu0, params.k, last_purchase_visit_index + 1, visit_index)
    b_ij = params.r_tau * np.exp(params.psi * prior_purchases)

    if prior_purchases == 0:
        p0 = a_ij / max(a_ij + b_ij + n_ij, EPS)
        p = params.pi * p0
    else:
        p = (a_ij + prior_purchases) / max(a_ij + b_ij + n_ij, EPS)
    return float(np.clip(p, EPS, 1.0 - EPS))


def cm_customer_loglik_and_state(purchase_sequence: np.ndarray, params: CMParams, initial_state: CMState | None = None, update_state: bool = True):
    seq = np.asarray(purchase_sequence).astype(int)
    state = CMState(**vars(initial_state)) if initial_state else CMState()
    ll = 0.0
    for y in seq:
        visit_index = state.total_visits + 1
        p = cm_purchase_probability(visit_index, state.prior_purchases, state.last_purchase_visit_index, params)
        if y == 1:
            ll += np.log(p)
        else:
            if state.prior_purchases == 0:
                p0 = p / max(params.pi, EPS)
                ll += np.log((1.0 - params.pi) + params.pi * (1.0 - p0))
            else:
                ll += np.log(1.0 - p)
        if update_state:
            state.total_visits += 1
            if y == 1:
                state.prior_purchases += 1
                state.last_purchase_visit_index = state.total_visits
    return float(ll), state


def cm_loglik(visits: pd.DataFrame, params: CMParams) -> float:
    ll = 0.0
    for _, g in visits.groupby("machine_id"):
        l, _ = cm_customer_loglik_and_state(g.sort_values("t")["purchase"].to_numpy(), params)
        ll += l
    return float(ll)


def _params_from_theta(theta: np.ndarray) -> CMParams:
    r_v = float(softplus(theta[0]) + 1e-6)
    mu0 = float(softplus(theta[1]) + 1e-6)
    k = float(np.exp(np.clip(theta[2], -5, 5)))
    r_tau = float(softplus(theta[3]) + 1e-6)
    psi = float(theta[4])
    pi = float(np.clip(sigmoid(theta[5]), 1e-6, 1.0 - 1e-6))
    return CMParams(r_v, mu0, k, r_tau, psi, pi)


def fit_cm_model(visits_cal: pd.DataFrame, n_starts: int = 30, seed: int = 123):
    rng = np.random.default_rng(seed)
    ever_purchase = visits_cal.groupby("machine_id")["purchase"].max().mean()

    def obj(theta):
        p = _params_from_theta(theta)
        return -cm_loglik(visits_cal, p)

    # initialize positive transformed parameters around 1.0 as requested:
    # softplus(theta) ~= 1 for r_v, mu0, r_tau and exp(theta)=1 for k.
    sp1 = inv_softplus(1.0)
    pi0 = np.clip(ever_purchase, 0.05, 0.95)

    best = None
    for _ in range(n_starts):
        x0 = np.array([
            rng.normal(sp1, 0.5),   # r_v -> around 1 after softplus
            rng.normal(sp1, 0.5),   # mu0 -> around 1 after softplus
            rng.normal(0.0, 0.2),   # log(k) -> around 0 so k~1
            rng.normal(sp1, 0.5),   # r_tau -> around 1 after softplus
            rng.normal(0.0, 0.2),   # psi
            rng.normal(np.log(pi0 / (1 - pi0)), 0.5),
        ])
        res = minimize(obj, x0=x0, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res

    params = _params_from_theta(best.x)
    info = {
        "converged": bool(best.success),
        "message": str(best.message),
        "objective": float(best.fun),
        "nit": int(best.nit),
    }
    return params, info


def cm_holdout_loglik(visits_cal: pd.DataFrame, visits_holdout: pd.DataFrame, params: CMParams) -> float:
    hold = {k: v.sort_values("t") for k, v in visits_holdout.groupby("machine_id")}
    ll = 0.0
    for mid, g in visits_cal.groupby("machine_id"):
        _, state = cm_customer_loglik_and_state(g.sort_values("t")["purchase"].to_numpy(), params)
        if mid in hold:
            l, _ = cm_customer_loglik_and_state(hold[mid]["purchase"].to_numpy(), params, initial_state=state)
            ll += l
    return float(ll)
