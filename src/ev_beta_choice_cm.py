from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
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
    r: float
    alpha: float
    s: float
    beta: float
    r_v: float
    r_tau: float
    psi: float
    pi_amazon: float
    mu0_amazon: float
    k_amazon: float
    pi_ebay: float
    mu0_ebay: float
    k_ebay: float
    choice_a: float
    choice_b: float
    obs_scale: float


@dataclass
class EVBetaChoiceCMConfig:
    amazon_fixed: dict[str, float]
    ebay_init: dict[str, float]
    choice: dict[str, Any]
    priors: dict[str, float]
    fit: dict[str, Any]


def build_initial_raw_params(config: EVBetaChoiceCMConfig) -> dict[str, float]:
    af = config.amazon_fixed
    ei = config.ebay_init
    choice_cfg = config.choice
    fit_cfg = config.fit

    init_mean = float(choice_cfg.get("initial_mean", 0.5))
    init_conc = float(choice_cfg.get("initial_concentration", 20.0))
    min_conc = float(choice_cfg.get("min_concentration", 2.0))

    return {
        "raw_pi_ebay": float(logit(jnp.asarray(ei["pi"]))),
        "raw_mu0_ebay": float(jnp.log(jnp.expm1(max(ei["mu0"], EPS)))),
        "raw_k_ebay": float(jnp.log(jnp.expm1(max(ei["k"], EPS)))),
        "raw_choice_mean": float(logit(jnp.asarray(init_mean))),
        "raw_choice_concentration": float(jnp.log(jnp.expm1(max(init_conc - min_conc, EPS)))),
        "raw_obs_scale": float(jnp.log(jnp.expm1(max(float(fit_cfg.get("obs_scale_init", 20.0)), EPS)))),
        "r": af["r"],
        "alpha": af["alpha"],
        "s": af["s"],
        "beta": af["beta"],
        "r_v": af["r_v"],
        "r_tau": af["r_tau"],
        "psi": af["psi"],
        "pi_amazon": af["pi"],
        "mu0_amazon": af["mu0"],
        "k_amazon": af["k"],
    }


def constrained_params(raw: dict[str, jnp.ndarray], config: EVBetaChoiceCMConfig) -> EVBetaChoiceCMParams:
    min_conc = float(config.choice.get("min_concentration", 2.0))
    fix_conc = bool(config.choice.get("fix_concentration", True))
    conc_fixed = float(config.choice.get("initial_concentration", 20.0))

    choice_mean = sigmoid(raw["raw_choice_mean"])
    choice_concentration = conc_fixed if fix_conc else softplus(raw["raw_choice_concentration"]) + min_conc

    choice_a = choice_mean * choice_concentration
    choice_b = (1.0 - choice_mean) * choice_concentration

    return EVBetaChoiceCMParams(
        r=raw["r"],
        alpha=raw["alpha"],
        s=raw["s"],
        beta=raw["beta"],
        r_v=raw["r_v"],
        r_tau=raw["r_tau"],
        psi=raw["psi"],
        pi_amazon=raw["pi_amazon"],
        mu0_amazon=raw["mu0_amazon"],
        k_amazon=raw["k_amazon"],
        pi_ebay=sigmoid(raw["raw_pi_ebay"]),
        mu0_ebay=softplus(raw["raw_mu0_ebay"]) + EPS,
        k_ebay=softplus(raw["raw_k_ebay"]) + EPS,
        choice_a=choice_a,
        choice_b=choice_b,
        obs_scale=softplus(raw["raw_obs_scale"]) + EPS,
    )


def predicted_monthly_mean(raw: dict[str, jnp.ndarray], month_index: jnp.ndarray, total_customers: float, config: EVBetaChoiceCMConfig) -> jnp.ndarray:
    p = constrained_params(raw, config)
    month_index = month_index.astype(jnp.float32)

    expected_visit_intentions = jnp.maximum(
        p.r_v * jnp.exp(p.psi * month_index) + p.mu0_ebay * jnp.power(p.k_ebay, month_index),
        EPS,
    )
    choice_mean = p.choice_a / jnp.maximum(p.choice_a + p.choice_b, EPS)
    cm_base = p.r / jnp.maximum(p.r + p.alpha + p.s + p.beta, EPS)
    expected_purchase_probability = jnp.clip(p.pi_ebay * cm_base + (1.0 - p.pi_ebay) * 0.01, EPS, 1.0)
    lam = total_customers * expected_visit_intentions * choice_mean * expected_purchase_probability
    return jnp.maximum(lam, EPS)


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


def loss_fn(raw: dict[str, jnp.ndarray], month_index: jnp.ndarray, y: jnp.ndarray, total_customers: float, config: EVBetaChoiceCMConfig) -> jnp.ndarray:
    lam = predicted_monthly_mean(raw, month_index, total_customers, config)
    p = constrained_params(raw, config)
    likelihood = config.fit.get("likelihood", "negative_binomial")

    if likelihood == "poisson":
        ll = jnp.sum(y * jnp.log(lam + EPS) - lam - gammaln(y + 1.0))
    else:
        ll = jnp.sum(negbinom_logpmf_from_mean_dispersion(y, lam, jnp.asarray(p.obs_scale)))

    lp = config.priors
    prior_penalty = (
        float(lp.get("lambda_pi", 0.1)) * (logit(jnp.asarray(p.pi_ebay)) - logit(jnp.asarray(p.pi_amazon))) ** 2
        + float(lp.get("lambda_mu", 0.1)) * (jnp.log(jnp.asarray(p.mu0_ebay) + EPS) - jnp.log(jnp.asarray(p.mu0_amazon) + EPS)) ** 2
        + float(lp.get("lambda_k", 0.1)) * (jnp.log(jnp.asarray(p.k_ebay) + EPS) - jnp.log(jnp.asarray(p.k_amazon) + EPS)) ** 2
        + float(lp.get("lambda_choice_mean", 0.01)) * (p.choice_a / (p.choice_a + p.choice_b) - 0.5) ** 2
    )
    return -ll + prior_penalty


def init_for_optimizer(config: EVBetaChoiceCMConfig) -> dict[str, jnp.ndarray]:
    raw = build_initial_raw_params(config)
    return {k: jnp.asarray(v, dtype=jnp.float32) for k, v in raw.items()}
