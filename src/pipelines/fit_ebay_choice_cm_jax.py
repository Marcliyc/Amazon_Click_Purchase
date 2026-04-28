from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import pandas as pd

from src.ev_beta_choice_cm import EVBetaChoiceCMConfig, constrained_params, init_for_optimizer, loss_fn, predicted_monthly_mean
from src.plots.plot_ebay_choice_cm import save_diagnostic_plots


def load_config(path: str):
    spec = importlib.util.spec_from_file_location("ebay_cfg", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.CONFIG


def _resolve_session_key(df: pd.DataFrame, preferred: str | None) -> str:
    for col in [preferred, "user_session_id", "site_session_id"]:
        if col and col in df.columns:
            return col
    return "fallback_session_key"


def aggregate_monthly_purchases(df: pd.DataFrame, date_col: str, purchase_col: str, session_col: str | None = None) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    key = _resolve_session_key(out, session_col)
    if key == "fallback_session_key":
        out[key] = out["machine_id"].astype(str) + "_" + out[date_col].dt.strftime("%Y-%m-%d") + "_" + out.get("event_time", "00:00:00").astype(str)
    out[purchase_col] = out[purchase_col].fillna(0).astype(int)

    sess = (
        out.groupby(key, as_index=False)
        .agg(month=(date_col, lambda x: pd.to_datetime(x.iloc[0]).to_period("M").to_timestamp("M")), purchase=(purchase_col, "max"))
    )
    monthly = sess.groupby("month", as_index=False)["purchase"].sum().rename(columns={"purchase": "actual_ebay_purchases"})
    return monthly.sort_values("month").reset_index(drop=True)


def _mape(a: pd.Series, p: pd.Series) -> float:
    denom = a.replace(0, pd.NA)
    return float((((a - p).abs() / denom).dropna()).mean() * 100.0) if denom.notna().any() else 0.0


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workdir", default=None)
    parsed = parser.parse_args(args=args)

    cfg = load_config(parsed.config)
    workdir = Path(parsed.workdir or cfg["outputs"]["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)

    print("Warning: fitting EV-Beta-Bernoulli-Choice-CM from aggregate monthly eBay purchase counts only.")
    print("Choice heterogeneity and eBay-specific EV/CM parameters may be weakly identified.")
    print("Use priors, fixed concentration, or additional session-level moments if available.")

    data_cfg = cfg["data"]
    dr = cfg["date_range"]
    ebay = pd.read_csv(data_cfg["ebay_path"])
    ebay[data_cfg["date_col"]] = pd.to_datetime(ebay[data_cfg["date_col"]])
    ebay = ebay[(ebay[data_cfg["date_col"]] >= dr["start"]) & (ebay[data_cfg["date_col"]] <= dr["holdout_end"])].copy()

    monthly = aggregate_monthly_purchases(ebay, data_cfg["date_col"], data_cfg["purchase_col"], data_cfg.get("session_id_col"))
    monthly["split"] = monthly["month"].apply(lambda d: "calibration" if d <= pd.to_datetime(dr["calibration_end"]) else "holdout")
    monthly["month_index"] = range(len(monthly))

    total_customers = float(ebay["machine_id"].nunique())
    model_cfg = EVBetaChoiceCMConfig(
        amazon_fixed=cfg["amazon_fixed"],
        ebay_init=cfg["ebay_init"],
        choice=cfg["choice"],
        priors=cfg["priors"],
        fit=cfg["fit"],
    )
    params = init_for_optimizer(model_cfg)

    train_mask = monthly["split"] == "calibration"
    x_train = jnp.asarray(monthly.loc[train_mask, "month_index"].to_numpy(), dtype=jnp.float32)
    y_train = jnp.asarray(monthly.loc[train_mask, "actual_ebay_purchases"].to_numpy(), dtype=jnp.float32)

    optimizer = optax.adam(float(cfg["fit"].get("learning_rate", 1e-2)))
    opt_state = optimizer.init(params)

    @jax.jit
    def step(p, s):
        loss, grads = jax.value_and_grad(loss_fn)(p, x_train, y_train, total_customers, model_cfg)
        updates, s = optimizer.update(grads, s, p)
        p = optax.apply_updates(p, updates)
        return p, s, loss

    history = []
    num_steps = int(cfg["fit"].get("num_steps", 1000))
    for i in range(num_steps):
        params, opt_state, loss = step(params, opt_state)
        history.append({"step": i, "loss": float(loss)})

    p_final = constrained_params(params, model_cfg)

    x_all = jnp.asarray(monthly["month_index"].to_numpy(), dtype=jnp.float32)
    pred = predicted_monthly_mean(params, x_all, total_customers, model_cfg)
    monthly["pred_mean_ebay_purchases"] = [float(v) for v in pred]
    monthly["pred_p05_ebay_purchases"] = pd.NA
    monthly["pred_p50_ebay_purchases"] = pd.NA
    monthly["pred_p95_ebay_purchases"] = pd.NA
    monthly["ape"] = ((monthly["actual_ebay_purchases"] - monthly["pred_mean_ebay_purchases"]).abs() / monthly["actual_ebay_purchases"].replace(0, pd.NA))

    cal = monthly[monthly["split"] == "calibration"]
    hold = monthly[monthly["split"] == "holdout"]
    cal_mape = _mape(cal["actual_ebay_purchases"], cal["pred_mean_ebay_purchases"])
    hold_mape = _mape(hold["actual_ebay_purchases"], hold["pred_mean_ebay_purchases"]) if len(hold) else 0.0

    params_initial = init_for_optimizer(model_cfg)
    pd.DataFrame({"param": list(params_initial.keys()), "raw_value": [float(v) for v in params_initial.values()]}).to_csv(workdir / "params_initial.csv", index=False)

    fitted_json = {
        "constrained": {k: float(v) for k, v in p_final.__dict__.items()},
        "raw": {k: float(v) for k, v in params.items()},
    }
    with open(workdir / "params_fitted.json", "w", encoding="utf-8") as f:
        json.dump(fitted_json, f, indent=2)
    pd.DataFrame({"param": list(fitted_json["constrained"].keys()), "value": list(fitted_json["constrained"].values())}).to_csv(workdir / "params_fitted.csv", index=False)

    monthly.to_csv(workdir / "monthly_fit.csv", index=False)
    pd.DataFrame(history).to_csv(workdir / "loss_curve.csv", index=False)
    pd.DataFrame(history).to_csv(workdir / "parameter_trace_or_history.csv", index=False)

    with open(workdir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"seed": cfg["fit"].get("seed", 123), "git_commit": _git_hash(), "config": cfg}, f, indent=2)

    save_diagnostic_plots(monthly, pd.DataFrame(history), p_final, workdir)

    final_cum_err = float(monthly["pred_mean_ebay_purchases"].sum() - monthly["actual_ebay_purchases"].sum())
    print(f"Fitted eBay pi: {p_final.pi_ebay:.6f}")
    print(f"Fitted eBay mu0: {p_final.mu0_ebay:.6f}")
    print(f"Fitted eBay k: {p_final.k_ebay:.6f}")
    print(f"Fitted choice mean: {p_final.choice_a / (p_final.choice_a + p_final.choice_b):.6f}")
    print(f"Fitted choice concentration: {(p_final.choice_a + p_final.choice_b):.6f}")
    print(f"Calibration MAPE: {cal_mape:.3f}")
    print(f"Holdout MAPE: {hold_mape:.3f}")
    print(f"Final cumulative error: {final_cum_err:.3f}")


if __name__ == "__main__":
    main()
