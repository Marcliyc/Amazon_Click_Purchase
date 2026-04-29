from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax.scipy.special import gammaln

EPS = 1e-8


def softplus(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0)


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (1.0 + jnp.exp(-x))


def logit(p: jnp.ndarray) -> jnp.ndarray:
    p = jnp.clip(p, EPS, 1.0 - EPS)
    return jnp.log(p) - jnp.log1p(-p)


@dataclass
class EVBetaChoiceCMParams:
    r: Any
    alpha: Any
    s: Any
    beta: Any
    r_v: Any
    r_tau: Any
    psi: Any
    pi_amazon: Any
    mu0_amazon: Any
    k_amazon: Any
    pi_ebay: Any
    mu0_ebay: Any
    k_ebay: Any
    choice_a: Any
    choice_b: Any
    obs_scale: Any


@dataclass
class EVBetaChoiceCMConfig:
    amazon_fixed: dict[str, float]
    ebay_init: dict[str, float]
    choice: dict[str, Any]
    priors: dict[str, float]
    fit: dict[str, Any]


def build_initial_raw_params(config: EVBetaChoiceCMConfig) -> dict[str, float]:
    ei = config.ebay_init
    af = config.amazon_fixed
    choice_cfg = config.choice
    fit_cfg = config.fit

    init_mean = float(choice_cfg.get("initial_mean", 0.5))
    init_conc = float(choice_cfg.get("initial_concentration", 20.0))
    min_conc = float(choice_cfg.get("min_concentration", 2.0))

    raw = {
        "raw_pi_ebay": float(logit(jnp.asarray(ei["pi"]))),
        "raw_mu0_ebay": float(jnp.log(jnp.expm1(max(ei["mu0"], EPS)))),
        "raw_k_ebay": float(jnp.log(jnp.expm1(max(ei["k"], EPS)))),
        "raw_choice_mean": float(logit(jnp.asarray(init_mean))),
        "raw_choice_concentration": float(jnp.log(jnp.expm1(max(init_conc - min_conc, EPS)))),
        "raw_obs_scale": float(jnp.log(jnp.expm1(max(float(fit_cfg.get("obs_scale_init", 20.0)), EPS)))),
    }

    if bool(fit_cfg.get("fit_shared_ev", True)):
        raw.update(
            {
                "raw_r_v": float(jnp.log(jnp.expm1(max(af["r_v"], EPS)))),
                "raw_r_tau": float(jnp.log(jnp.expm1(max(af["r_tau"], EPS)))),
                "raw_psi": float(af["psi"]),
            }
        )
    return raw


def constrained_params(raw: dict[str, jnp.ndarray], config: EVBetaChoiceCMConfig) -> EVBetaChoiceCMParams:
    af = config.amazon_fixed
    min_conc = float(config.choice.get("min_concentration", 2.0))
    fix_conc = bool(config.choice.get("fix_concentration", True))
    conc_fixed = float(config.choice.get("initial_concentration", 20.0))

    choice_mean = sigmoid(raw["raw_choice_mean"])
    choice_concentration = conc_fixed if fix_conc else softplus(raw["raw_choice_concentration"]) + min_conc

    r_v = softplus(raw["raw_r_v"]) + EPS if "raw_r_v" in raw else jnp.asarray(af["r_v"])
    r_tau = softplus(raw["raw_r_tau"]) + EPS if "raw_r_tau" in raw else jnp.asarray(af["r_tau"])
    psi = raw["raw_psi"] if "raw_psi" in raw else jnp.asarray(af["psi"])

    return EVBetaChoiceCMParams(
        r=jnp.asarray(af["r"]),
        alpha=jnp.asarray(af["alpha"]),
        s=jnp.asarray(af["s"]),
        beta=jnp.asarray(af["beta"]),
        r_v=r_v,
        r_tau=r_tau,
        psi=psi,
        pi_amazon=jnp.asarray(af["pi"]),
        mu0_amazon=jnp.asarray(af["mu0"]),
        k_amazon=jnp.asarray(af["k"]),
        pi_ebay=sigmoid(raw["raw_pi_ebay"]),
        mu0_ebay=softplus(raw["raw_mu0_ebay"]) + EPS,
        k_ebay=softplus(raw["raw_k_ebay"]) + EPS,
        choice_a=choice_mean * choice_concentration,
        choice_b=(1.0 - choice_mean) * choice_concentration,
        obs_scale=softplus(raw["raw_obs_scale"]) + EPS,
    )


def _expected_visit_intentions(month_index: jnp.ndarray, r_v: jnp.ndarray, psi: jnp.ndarray, mu0: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(r_v * jnp.exp(psi * month_index) + mu0 * jnp.power(k, month_index), EPS)


def predicted_monthly_mean(raw: dict[str, jnp.ndarray], month_index: jnp.ndarray, total_customers: float, config: EVBetaChoiceCMConfig) -> jnp.ndarray:
    p = constrained_params(raw, config)
    month_index = month_index.astype(jnp.float32)

    visit_intentions = _expected_visit_intentions(month_index, p.r_v, p.psi, p.mu0_ebay, p.k_ebay)
    choice_mean = p.choice_a / jnp.maximum(p.choice_a + p.choice_b, EPS)
    cm_base = p.r / jnp.maximum(p.r + p.alpha + p.s + p.beta, EPS)
    expected_purchase_probability = jnp.clip(p.pi_ebay * cm_base + (1.0 - p.pi_ebay) * 0.01, EPS, 1.0)
    lam = total_customers * visit_intentions * choice_mean * expected_purchase_probability
    return jnp.maximum(lam, EPS)


def predicted_amazon_visits(raw: dict[str, jnp.ndarray], month_index: jnp.ndarray, total_customers: float, config: EVBetaChoiceCMConfig) -> jnp.ndarray:
    p = constrained_params(raw, config)
    month_index = month_index.astype(jnp.float32)
    visit_intentions = _expected_visit_intentions(month_index, p.r_v, p.psi, p.mu0_amazon, p.k_amazon)
    choice_mean = p.choice_a / jnp.maximum(p.choice_a + p.choice_b, EPS)
    visits = total_customers * visit_intentions * (1.0 - choice_mean)
    return jnp.maximum(visits, EPS)


def negbinom_logpmf_from_mean_dispersion(y: jnp.ndarray, mean: jnp.ndarray, dispersion: jnp.ndarray) -> jnp.ndarray:
    mean = jnp.maximum(mean, EPS)
    dispersion = jnp.maximum(dispersion, EPS)
    return (
        gammaln(y + dispersion)
        - gammaln(dispersion)
        - gammaln(y + 1.0)
        + dispersion * (jnp.log(dispersion) - jnp.log(dispersion + mean))
        + y * (jnp.log(mean) - jnp.log(dispersion + mean))
    )


def _poisson_ll(y: jnp.ndarray, mean: jnp.ndarray) -> jnp.ndarray:
    mean = jnp.maximum(mean, EPS)
    return jnp.sum(y * jnp.log(mean) - mean - gammaln(y + 1.0))


def loss_fn(
    raw: dict[str, jnp.ndarray],
    month_index: jnp.ndarray,
    y_ebay: jnp.ndarray,
    total_customers_ebay: float,
    config: EVBetaChoiceCMConfig,
    month_index_amazon: jnp.ndarray | None = None,
    y_amazon_visits: jnp.ndarray | None = None,
    total_customers_amazon: float | None = None,
) -> jnp.ndarray:
    lam = predicted_monthly_mean(raw, month_index, total_customers_ebay, config)
    p = constrained_params(raw, config)
    likelihood = config.fit.get("likelihood", "negative_binomial")

    if likelihood == "poisson":
        ll_ebay = _poisson_ll(y_ebay, lam)
    else:
        ll_ebay = jnp.sum(negbinom_logpmf_from_mean_dispersion(y_ebay, lam, jnp.asarray(p.obs_scale)))

    ll_amazon = 0.0
    if month_index_amazon is not None and y_amazon_visits is not None and total_customers_amazon is not None:
        amazon_mean = predicted_amazon_visits(raw, month_index_amazon, total_customers_amazon, config)
        ll_amazon = _poisson_ll(y_amazon_visits, amazon_mean)

    lp = config.priors
    prior_penalty = (
        float(lp.get("lambda_pi", 0.1)) * (logit(jnp.asarray(p.pi_ebay)) - logit(jnp.asarray(p.pi_amazon))) ** 2
        + float(lp.get("lambda_mu", 0.1)) * (jnp.log(jnp.asarray(p.mu0_ebay) + EPS) - jnp.log(jnp.asarray(p.mu0_amazon) + EPS)) ** 2
        + float(lp.get("lambda_k", 0.1)) * (jnp.log(jnp.asarray(p.k_ebay) + EPS) - jnp.log(jnp.asarray(p.k_amazon) + EPS)) ** 2
        + float(lp.get("lambda_choice_mean", 0.01)) * (p.choice_a / (p.choice_a + p.choice_b) - 0.5) ** 2
    )

    visit_weight = float(config.fit.get("amazon_visit_weight", 1.0))
    return -(ll_ebay + visit_weight * ll_amazon) + prior_penalty


def init_for_optimizer(config: EVBetaChoiceCMConfig) -> dict[str, jnp.ndarray]:
    raw = build_initial_raw_params(config)
    return {k: jnp.asarray(v, dtype=jnp.float32) for k, v in raw.items()}
