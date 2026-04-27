from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .cm_model import cm_holdout_loglik, cm_loglik, fit_cm_model
from .data_prep import make_daily_visits, make_session_time_visits, make_session_visits, load_raw_data, split_calibration_holdout
from .ev_model import ev_holdout_loglik, ev_loglik, fit_ev_model
from .forecasting import aggregate_actual_holdout, simulate_evcm_forecast
from .metrics import forecast_error_metrics, information_criteria


def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cutoff", default=None)
    p.add_argument("--calib-frac", type=float, default=0.5)
    p.add_argument("--visit-unit", choices=["daily", "session"], default="daily")
    p.add_argument("--freq", default="W")
    p.add_argument("--n-sims", type=int, default=1000)
    p.add_argument("--ev-starts", type=int, default=20)
    p.add_argument("--cm-starts", type=int, default=30)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max-machines", type=int, default=None)
    p.add_argument("--no-plots", action="store_true")
    return p


def main(args=None):
    args = _build_parser().parse_args(args=args)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data(args.input)
    session = make_session_visits(raw)
    visits = make_daily_visits(session) if args.visit_unit == "daily" else make_session_time_visits(session)

    visits.to_csv(out_dir / "processed_visits.csv", index=False)

    if args.max_machines:
        keep = visits["machine_id"].drop_duplicates().head(args.max_machines)
        visits = visits[visits["machine_id"].isin(keep)]

    cal, hold, split = split_calibration_holdout(visits, cutoff=args.cutoff, calibration_fraction=args.calib_frac)
    cal_machines = set(cal["machine_id"].unique())
    hold_known = hold[hold["machine_id"].isin(cal_machines)].copy()
    hold_new = hold[~hold["machine_id"].isin(cal_machines)].copy()

    T_cal_end = float(cal["t"].max()) if len(cal) else 0.0
    T_holdout_end = float(visits["t"].max()) if len(visits) else T_cal_end

    print(f"Calibration period: 0 to {T_cal_end:.2f} (n={len(cal)})")
    print(f"Holdout period: {T_cal_end:.2f} to {T_holdout_end:.2f} (n={len(hold_known)})")

    ev_params, ev_info = fit_ev_model(cal, T_cal_end, n_starts=args.ev_starts, seed=args.seed)
    cm_params, cm_info = fit_cm_model(cal, n_starts=args.cm_starts, seed=args.seed)

    ll_ev_cal = ev_loglik(cal, T_cal_end, ev_params)
    ll_cm_cal = cm_loglik(cal, cm_params)
    ll_ev_hold = ev_holdout_loglik(cal, hold_known, T_cal_end, T_holdout_end, ev_params)
    ll_cm_hold = cm_holdout_loglik(cal, hold_known, cm_params)

    #save params and log-likelihoods for both calibration and holdout periods
    pd.DataFrame({"param": ["r", "alpha", "s", "beta"], "value": [ev_params.r, ev_params.alpha, ev_params.s, ev_params.beta]}).to_csv(out_dir / "params_ev.csv", index=False)
    pd.DataFrame({"param": ["r_v", "mu0", "k", "r_tau", "psi", "pi"], "value": [cm_params.r_v, cm_params.mu0, cm_params.k, cm_params.r_tau, cm_params.psi, cm_params.pi]}).to_csv(out_dir / "params_cm.csv", index=False)

    print(f"Calibration LL - EV: {ll_ev_cal:.2f}, CM: {ll_cm_cal:.2f}, Joint: {ll_ev_cal + ll_cm_cal:.2f}"
          f" | Holdout LL - EV: {ll_ev_hold:.2f}, CM: {ll_cm_hold:.2f}, Joint: {ll_ev_hold + ll_cm_hold:.2f}")
    

    sim = simulate_evcm_forecast(cal, T_cal_end, T_holdout_end, ev_params, cm_params, n_sims=args.n_sims, freq=args.freq, seed=args.seed)

    print(f"Simulated forecast for holdout period with {args.n_sims} simulations.")
    actual = aggregate_actual_holdout(hold_known, T_cal_end, T_holdout_end, freq=args.freq)
    by_period = actual.merge(sim, on=["period_idx", "period_start_t", "period_end_t"], how="left")
    err, by_period = forecast_error_metrics(by_period)

    print(f"Forecast error metrics: {err}")
    n_ev = int(cal.groupby("machine_id").size().clip(lower=1).sum())
    n_cm = int(len(cal))
    joint_ll = ll_ev_cal + ll_cm_cal
    ic = information_criteria(joint_ll, k=10, n=n_ev + n_cm)

    by_period.to_csv(out_dir / "forecast_by_period.csv", index=False)
    #pd.DataFrame({"param": ["r", "alpha", "s", "beta"], "value": [ev_params.r, ev_params.alpha, ev_params.s, ev_params.beta]}).to_csv(out_dir / "params_ev.csv", index=False)
    #pd.DataFrame({"param": ["r_v", "mu0", "k", "r_tau", "psi", "pi"], "value": [cm_params.r_v, cm_params.mu0, cm_params.k, cm_params.r_tau, cm_params.psi, cm_params.pi]}).to_csv(out_dir / "params_cm.csv", index=False)

    metrics = {
        "data": {
            "n_raw_rows": int(len(raw)),
            "n_session_visits": int(len(session)),
            "n_daily_visits": int(len(visits)),
            "n_machines_total": int(visits["machine_id"].nunique()),
            "n_machines_calibration": split.n_machines_calibration,
            "n_machines_holdout_known": split.n_machines_holdout_known,
            "n_machines_holdout_new": split.n_machines_holdout_new,
            "calibration_start": split.global_start,
            "calibration_end": split.calibration_end,
            "holdout_end": split.holdout_end,
            "new_machine_holdout_purchases": float(hold_new["purchase"].sum()),
        },
        "fit": {
            "LL_EV_cal": ll_ev_cal,
            "LL_CM_cal": ll_cm_cal,
            "LL_joint_cal": joint_ll,
            "LL_EV_holdout": ll_ev_hold,
            "LL_CM_holdout": ll_cm_hold,
            "LL_joint_holdout": ll_ev_hold + ll_cm_hold,
            "AIC_joint": ic["AIC"],
            "BIC_joint": ic["BIC"],
            "CAIC_joint": ic["CAIC"],
        },
        "forecast": {
            **err,
            "actual_holdout_cum_purchases_final": float(by_period["actual_cum_purchases"].iloc[-1]),
            "forecast_holdout_cum_purchases_final": float(by_period["forecast_mean_cum_purchases"].iloc[-1]),
            "actual_holdout_visits": int(len(hold_known)),
            "forecast_holdout_visits_mean": float(sim.attrs.get("forecast_holdout_visits_mean", 0.0)),
        },
        "conversion": {
            "actual_holdout_conversion_rate": float(hold_known["purchase"].mean()) if len(hold_known) else 0.0,
            "predicted_holdout_conversion_rate": float(by_period["forecast_mean_purchases"].sum() / max(sim.attrs.get("forecast_holdout_visits_mean", 1.0), 1.0)),
            "conversion_relative_error_pct": None,
        },
        "optimization": {
            "ev_converged": ev_info["converged"],
            "cm_converged": cm_info["converged"],
            "ev_message": ev_info["message"],
            "cm_message": cm_info["message"],
        },
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
