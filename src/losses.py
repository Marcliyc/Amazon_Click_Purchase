from __future__ import annotations

import torch
import torch.nn.functional as F

HEAD_ORDER = ["visits_pc", "txns_pc", "avg_payment", "agg_visits", "agg_txns", "agg_revenue"]


def implied_cohort_revenue(cohort_size, transactions_per_customer, avg_payment):
    return cohort_size * transactions_per_customer * avg_payment


def implied_cohort_counts(cohort_size, visits_per_customer, transactions_per_customer):
    return cohort_size * visits_per_customer, cohort_size * transactions_per_customer


def cbmt_loss(pred: dict[str, torch.Tensor], y: torch.Tensor, consistency: torch.Tensor, weights: dict[str, float] | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    weights = weights or {}
    losses = {}
    total = torch.tensor(0.0, device=y.device)
    for j, key in enumerate(HEAD_ORDER):
        # log-space base losses stabilize sparse count/revenue targets while preserving nonnegative original-scale outputs.
        l = F.huber_loss(torch.log1p(pred[key]), torch.log1p(torch.clamp(y[:, j], min=0)))
        losses[key] = l
        total = total + float(weights.get(key if key not in {"visits_pc", "txns_pc"} else {"visits_pc":"visits_pc","txns_pc":"txns_pc"}[key], 1.0)) * l
    actual_visits, actual_txns, actual_revenue, cohort_size = consistency[:, 0], consistency[:, 1], consistency[:, 2], consistency[:, 3]
    pred_visits, pred_txns = implied_cohort_counts(cohort_size, pred["visits_pc"], pred["txns_pc"])
    pred_revenue = implied_cohort_revenue(cohort_size, pred["txns_pc"], pred["avg_payment"])
    cons = {
        "visit_consistency": F.mse_loss(torch.log1p(pred_visits), torch.log1p(actual_visits)),
        "txn_consistency": F.mse_loss(torch.log1p(pred_txns), torch.log1p(actual_txns)),
        "revenue_consistency": F.mse_loss(torch.log1p(pred_revenue), torch.log1p(actual_revenue)),
    }
    for k, l in cons.items():
        losses[k] = l
        total = total + float(weights.get(k, 1.0)) * l
    return total, {k: float(v.detach().cpu()) for k, v in losses.items()} | {"loss": float(total.detach().cpu())}
