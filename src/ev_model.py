from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log1p


EPS = 1e-9


def softplus(x: float | np.ndarray) -> float | np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


@dataclass
class EVParams:
    r: float
    alpha: float
    s: float
    beta: float


@dataclass
class EVState:
    r: float
    alpha: float
    last_t: float
    total_visits_seen: int


def ev_customer_loglik_and_state(times: np.ndarray, T_end: float, params: EVParams, return_state_at_end: bool = True):
    times = np.asarray(times, dtype=float)
    if times.size == 0:
        return 0.0, EVState(params.r, params.alpha + T_end, 0.0, 0)
    times = np.sort(times)

    r_cur, a_cur = params.r, params.alpha
    ll = 0.0
    for j in range(1, len(times)):
        d = max(times[j] - times[j - 1], EPS)
        ll += np.log(r_cur + EPS) + r_cur * np.log(a_cur + EPS) - (r_cur + 1) * np.log(a_cur + d + EPS)
        r_arr = r_cur + 1.0
        a_arr = a_cur + d
        denom = r_arr + params.s + 1.0
        r_cur = (r_arr * params.s) / max(denom, EPS)
        a_cur = (a_arr * params.beta) / max(denom, EPS)

    c = max(T_end - times[-1], 0.0)
    ll += r_cur * (np.log(a_cur + EPS) - np.log(a_cur + c + EPS))
    if return_state_at_end:
        return ll, EVState(r_cur, a_cur + c, float(times[-1]), int(len(times)))
    return ll, EVState(r_cur, a_cur, float(times[-1]), int(len(times)))


def ev_loglik(visits: pd.DataFrame, T_end: float, params: EVParams) -> float:
    ll = 0.0
    for _, g in visits.groupby("machine_id"):
        v, _ = ev_customer_loglik_and_state(g["t"].to_numpy(), T_end, params)
        ll += v
    return float(ll)


def _params_from_theta(theta: np.ndarray) -> EVParams:
    return EVParams(*(softplus(theta) + 1e-6))


def fit_ev_model(visits_cal: pd.DataFrame, T_cal_end: float, n_starts: int = 20, seed: int = 123):
    rng = np.random.default_rng(seed)
    best = None

    def obj(theta):
        p = _params_from_theta(theta)
        return -ev_loglik(visits_cal, T_cal_end, p)

    for _ in range(n_starts):
        x0 = rng.normal(0, 1, 4)
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


def ev_holdout_loglik(visits_cal, visits_holdout, T_cal_end: float, T_holdout_end: float, params: EVParams) -> float:
    hold_grp = {k: v.sort_values("t") for k, v in visits_holdout.groupby("machine_id")}
    ll = 0.0
    for mid, g in visits_cal.groupby("machine_id"):
        _, state = ev_customer_loglik_and_state(g["t"].to_numpy(), T_cal_end, params, return_state_at_end=True)
        r_cur, a_cur = state.r, state.alpha
        prev_t = T_cal_end
        h = hold_grp.get(mid)
        if h is not None and len(h):
            for t in h["t"].to_numpy():
                d = max(t - prev_t, EPS)
                ll += np.log(r_cur + EPS) + r_cur * np.log(a_cur + EPS) - (r_cur + 1) * np.log(a_cur + d + EPS)
                r_arr = r_cur + 1.0
                a_arr = a_cur + d
                denom = r_arr + params.s + 1.0
                r_cur = (r_arr * params.s) / max(denom, EPS)
                a_cur = (a_arr * params.beta) / max(denom, EPS)
                prev_t = t
        c = max(T_holdout_end - prev_t, 0.0)
        ll += r_cur * (np.log(a_cur + EPS) - np.log(a_cur + c + EPS))
    return float(ll)
