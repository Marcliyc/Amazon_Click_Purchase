from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import pandas as pd
import yaml

from .amazon_covariates import build_amazon_covariates
from .amazon_ml_evcm import PARAM_NAMES, init_head_params, loss_fn, machine_parameter_frame, make_training_data
from .data_prep import load_raw_data, make_daily_visits, make_session_time_visits, make_session_visits, split_calibration_holdout
from .plot_amazon_ml_evcm import save_default_plots


def _filter_domain(df: pd.DataFrame, domain_col: str | None, domain_value: str | None) -> pd.DataFrame:
    if not domain_col or domain_col not in df.columns or not domain_value:
        return df
    return df[df[domain_col].astype(str).str.strip().str.lower() == domain_value.strip().lower()].copy()


def _load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _coefficient_frame(params: dict, feature_names: list[str]) -> pd.DataFrame:
    W = pd.DataFrame(jax.device_get(params["W"]), index=feature_names, columns=PARAM_NAMES)
    return W.reset_index(names="feature").melt(id_vars="feature", var_name="parameter", value_name="coefficient")


def _segment_summary(machine_params: pd.DataFrame, visits: pd.DataFrame, segment_cols: list[str]) -> pd.DataFrame:
    frame = machine_params.copy()
    if "n_sessions" not in frame.columns or "purchase_rate" not in frame.columns:
        activity = visits.groupby("machine_id", as_index=False).agg(n_sessions=("purchase", "size"), purchases=("purchase", "sum"), purchase_rate=("purchase", "mean"))
        frame = frame.merge(activity, on="machine_id", how="left")
    rows = []
    for col in segment_cols:
        if col not in frame.columns:
            continue
        for segment, g in frame.groupby(col, dropna=False):
            row = {
                "segment_variable": col,
                "segment": str(segment),
                "n_machines": int(g["machine_id"].nunique()),
                "n_sessions": int(g["n_sessions"].fillna(0).sum()),
                "purchase_rate": float(g["purchase_rate"].mean()) if len(g) else 0.0,
            }
            for p in PARAM_NAMES:
                row[f"mean_predicted_{p}"] = float(g[p].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def main(args=None):
    parser = argparse.ArgumentParser(description="Train Amazon-only covariate-conditioned EV/CM model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--workdir", default=None)
    parsed = parser.parse_args(args=args)

    cfg = _load_config(parsed.config)
    out = Path(parsed.workdir or cfg["outputs"]["workdir"])
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    cov_cfg = cfg["covariates"]
    fit_cfg = cfg["fit"]

    raw = load_raw_data(data_cfg["input_path"])
    raw = _filter_domain(raw, data_cfg.get("domain_col"), data_cfg.get("amazon_domain_value"))
    sessions = make_session_visits(raw)
    visits = make_daily_visits(sessions) if data_cfg.get("visit_unit", "session") == "daily" else make_session_time_visits(sessions)
    cal, val, split = split_calibration_holdout(visits, cutoff=data_cfg.get("cutoff"), calibration_fraction=float(data_cfg.get("calibration_fraction", 0.8)))

    machine_ids = sorted(cal["machine_id"].unique().tolist())
    X, feature_names, _, metadata, machine_covariates = build_amazon_covariates(
        raw,
        machine_ids=machine_ids,
        rare_min_count=int(cov_cfg.get("rare_min_count", 10)),
        standardize=bool(cov_cfg.get("standardize_covariates", True)),
    )
    train_data = make_training_data(cal, machine_ids, X)
    val_known = val[val["machine_id"].isin(machine_ids)].copy()
    val_data = make_training_data(val_known, machine_ids, X, T_end=float(visits["t"].max()) if len(visits) else train_data.T_end)

    print(f"machines={len(machine_ids)} sessions={len(cal)} purchases={int(cal['purchase'].sum())} covariate_features={len(feature_names)}")
    print(f"parameters={PARAM_NAMES}")
    print(f"train_period={split.global_start}..{split.calibration_end} validation_period={split.calibration_end}..{split.holdout_end}")

    params = init_head_params(len(feature_names), seed=int(fit_cfg.get("seed", 123)), w_scale=float(fit_cfg.get("w_init_scale", 0.0)))
    opt = optax.adam(float(fit_cfg.get("learning_rate", 0.01)))
    opt_state = opt.init(params)
    use_covariates = bool(cov_cfg.get("use_covariates", True))
    lambda_beta = float(fit_cfg.get("lambda_beta", 0.01))

    @jax.jit
    def step(p, state):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, train_data, lambda_beta, use_covariates)
        updates, state = opt.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return p, state, loss, aux

    history = []
    for step_idx in range(int(fit_cfg.get("num_steps", 1000))):
        params, opt_state, loss, aux = step(params, opt_state)
        nll, penalty, _ = aux
        if step_idx % max(int(fit_cfg.get("log_every", 100)), 1) == 0 or step_idx == int(fit_cfg.get("num_steps", 1000)) - 1:
            print(f"step={step_idx} loss={float(loss):.4f} nll={float(nll):.4f} penalty={float(penalty):.4f}")
        history.append({"step": step_idx, "loss": float(loss), "nll": float(nll), "penalty": float(penalty)})

    (_, (train_nll, train_penalty, theta)) = loss_fn(params, train_data, lambda_beta, use_covariates)
    (val_loss, (val_nll, _, _)) = loss_fn(params, val_data, lambda_beta, use_covariates)
    base_params = {"base": params["base"], "W": jnp.zeros_like(params["W"])}
    (base_loss, (base_nll, _, _)) = loss_fn(base_params, train_data, lambda_beta, False)

    machine_params = machine_parameter_frame(machine_ids, jax.device_get(theta))
    machine_params = machine_params.merge(machine_covariates, on="machine_id", how="left")
    activity = cal.groupby("machine_id", as_index=False).agg(n_sessions=("purchase", "size"), purchases=("purchase", "sum"), purchase_rate=("purchase", "mean"))
    machine_params = machine_params.merge(activity, on="machine_id", how="left")
    coefficients = _coefficient_frame(params, feature_names)
    segments = _segment_summary(machine_params, cal, ["household_income", "household_size", "census_region", "racial_background", "country_of_origin"])

    loss_history = pd.DataFrame(history)
    comparison = pd.DataFrame(
        [
            {"model": "homogeneous_base", "num_parameters": len(PARAM_NAMES), "train_nll": float(base_nll), "validation_nll": None, "AIC": 2 * len(PARAM_NAMES) + 2 * float(base_nll), "BIC": None},
            {"model": "amazon_ml_evcm", "num_parameters": int(params["base"].size + params["W"].size), "train_nll": float(train_nll), "validation_nll": float(val_nll), "AIC": 2 * int(params["base"].size + params["W"].size) + 2 * float(train_nll), "BIC": None},
        ]
    )

    machine_params.to_csv(out / "machine_parameter_predictions.csv", index=False)
    segments.to_csv(out / "segment_parameter_summary.csv", index=False)
    coefficients.to_csv(out / "covariate_coefficients.csv", index=False)
    loss_history.to_csv(out / "training_loss.csv", index=False)
    comparison.to_csv(out / "model_comparison.csv", index=False)
    _save_json(out / "covariate_metadata.json", metadata.to_dict())
    _save_json(out / "config_used.json", cfg)
    _save_json(out / "fitted_params.json", {"base": jax.device_get(params["base"]).tolist(), "W": jax.device_get(params["W"]).tolist(), "param_names": PARAM_NAMES, "feature_names": feature_names})

    save_default_plots(loss_history, machine_params, coefficients, plots)
    print(f"Wrote Amazon ML EV/CM outputs to {out}")


if __name__ == "__main__":
    main()
