from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .cm_model_jax import _customer_loglik as _cm_customer_loglik
from .ev_model_jax import _customer_loglik as _ev_customer_loglik

EPS = 1e-8
PARAM_NAMES = [
    "ev_r",
    "ev_alpha",
    "ev_s",
    "ev_beta",
    "cm_r_v",
    "cm_mu0",
    "cm_k",
    "cm_r_tau",
    "cm_psi",
    "cm_pi",
]


@dataclass
class AmazonMLEVCMTrainingData:
    machine_ids: list
    times: np.ndarray
    purchases: np.ndarray
    lengths: np.ndarray
    X: np.ndarray
    T_end: float


def softplus(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0)


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (1.0 + jnp.exp(-x))


def inverse_softplus(x: float) -> float:
    x = max(float(x), EPS)
    return float(np.log(np.expm1(x)))


def logit(p: float) -> float:
    p = float(np.clip(p, EPS, 1.0 - EPS))
    return float(np.log(p) - np.log1p(-p))


def default_base_raw() -> np.ndarray:
    # Defaults are the previously estimated Amazon EV/CM values used elsewhere in this repo.
    constrained = {
        "ev_r": 0.457522,
        "ev_alpha": 7.372648,
        "ev_s": 16.440086,
        "ev_beta": 16.369219,
        "cm_r_v": 1.939664,
        "cm_mu0": 0.751528,
        "cm_k": 0.730525,
        "cm_r_tau": 22.712252,
        "cm_psi": -0.080496,
        "cm_pi": 0.208507,
    }
    return np.asarray(
        [
            inverse_softplus(constrained["ev_r"]),
            inverse_softplus(constrained["ev_alpha"]),
            inverse_softplus(constrained["ev_s"]),
            inverse_softplus(constrained["ev_beta"]),
            inverse_softplus(constrained["cm_r_v"]),
            inverse_softplus(constrained["cm_mu0"]),
            np.log(constrained["cm_k"]),
            inverse_softplus(constrained["cm_r_tau"]),
            constrained["cm_psi"],
            logit(constrained["cm_pi"]),
        ],
        dtype=np.float32,
    )


def init_head_params(n_features: int, seed: int = 123, w_scale: float = 0.0) -> dict[str, jnp.ndarray]:
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, w_scale, size=(n_features, len(PARAM_NAMES))).astype(np.float32)
    return {"base": jnp.asarray(default_base_raw()), "W": jnp.asarray(W)}


def apply_constraints(eta: jnp.ndarray) -> jnp.ndarray:
    ev = softplus(eta[:, 0:4]) + EPS
    cm_r_v = softplus(eta[:, 4:5]) + EPS
    cm_mu0 = softplus(eta[:, 5:6]) + EPS
    cm_k = jnp.exp(jnp.clip(eta[:, 6:7], -5.0, 5.0))
    cm_r_tau = softplus(eta[:, 7:8]) + EPS
    cm_psi = eta[:, 8:9]
    cm_pi = jnp.clip(sigmoid(eta[:, 9:10]), EPS, 1.0 - EPS)
    return jnp.concatenate([ev, cm_r_v, cm_mu0, cm_k, cm_r_tau, cm_psi, cm_pi], axis=1)


def covariate_parameter_head(params: dict[str, jnp.ndarray], X: jnp.ndarray, use_covariates: bool = True) -> jnp.ndarray:
    eta = jnp.broadcast_to(params["base"][None, :], (X.shape[0], params["base"].shape[0]))
    if use_covariates and X.shape[1] > 0:
        eta = eta + X @ params["W"]
    return apply_constraints(eta)


def make_training_data(visits: pd.DataFrame, machine_ids: list, X: np.ndarray, T_end: float | None = None) -> AmazonMLEVCMTrainingData:
    groups = {mid: g.sort_values("t") for mid, g in visits.groupby("machine_id")}
    lengths = np.asarray([len(groups.get(mid, [])) for mid in machine_ids], dtype=np.int32)
    max_len = int(lengths.max()) if len(lengths) else 0
    times = np.zeros((len(machine_ids), max_len), dtype=np.float32)
    purchases = np.zeros((len(machine_ids), max_len), dtype=np.int32)
    for i, mid in enumerate(machine_ids):
        if mid not in groups:
            continue
        g = groups[mid]
        n = len(g)
        times[i, :n] = g["t"].to_numpy(dtype=np.float32)
        purchases[i, :n] = g["purchase"].to_numpy(dtype=np.int32)
    if T_end is None:
        T_end = float(visits["t"].max()) if len(visits) else 0.0
    return AmazonMLEVCMTrainingData(machine_ids=machine_ids, times=times, purchases=purchases, lengths=lengths, X=X.astype(np.float32), T_end=float(T_end))


def loglik_by_machine(params: dict[str, jnp.ndarray], data: AmazonMLEVCMTrainingData, use_covariates: bool = True) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    theta = covariate_parameter_head(params, jnp.asarray(data.X), use_covariates=use_covariates)
    ev_params = theta[:, 0:4]
    cm_params = theta[:, 4:10]
    times = jnp.asarray(data.times)
    purchases = jnp.asarray(data.purchases)
    lengths = jnp.asarray(data.lengths)
    ev_ll = jax.vmap(_ev_customer_loglik, in_axes=(0, 0, None, 0))(times, lengths, data.T_end, ev_params)
    cm_ll = jax.vmap(_cm_customer_loglik, in_axes=(0, 0, 0))(purchases, lengths, cm_params)
    active = lengths > 0
    return jnp.where(active, ev_ll, 0.0), jnp.where(active, cm_ll, 0.0), theta


def loss_fn(params: dict[str, jnp.ndarray], data: AmazonMLEVCMTrainingData, lambda_beta: float = 1e-2, use_covariates: bool = True) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    ev_ll, cm_ll, theta = loglik_by_machine(params, data, use_covariates=use_covariates)
    penalty = float(lambda_beta) * jnp.sum(params["W"] ** 2)
    nll = -(ev_ll.sum() + cm_ll.sum())
    return nll + penalty, (nll, penalty, theta)


def machine_parameter_frame(machine_ids: list, theta: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(theta, columns=PARAM_NAMES)
    out.insert(0, "machine_id", machine_ids)
    return out
