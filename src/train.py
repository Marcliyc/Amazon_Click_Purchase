from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import save_config
from .cohort_builder import build_and_save
from .covariates import attach_covariates, build_cohort_covariates, numeric_feature_columns, CovariateScaler
from .dataset import WindowedCohortDataset, merge_aggregate, temporal_splits
from .losses import cbmt_loss
from .model_cbmt import CBMTTransformer
from .utils import ensure_dir, set_seed, write_json


def prepare_model_tables(config: dict) -> tuple[pd.DataFrame, list[str], object, object]:
    outdir = Path(config["data"]["output_dir"])
    if not (outdir / "cohort_week_panel.csv").exists():
        build_and_save(config)
    panel = pd.read_csv(outdir / "cohort_week_panel.csv", parse_dates=["cohort_week", "calendar_week"])
    agg = pd.read_csv(outdir / "aggregate_week_panel.csv", parse_dates=["calendar_week"])
    sessions = pd.read_csv(outdir / "cleaned_sessions.csv", parse_dates=["calendar_week"])
    cohorts = pd.read_csv(outdir / "customer_cohorts.csv", parse_dates=["cohort_week"])
    cohort_features = build_cohort_covariates(sessions, cohorts, config)
    panel = merge_aggregate(attach_covariates(panel, cohort_features), agg)
    splits = temporal_splits(panel["calendar_week"].unique().tolist(), config["split"].get("val_weeks", 12), config["split"].get("holdout_weeks", 12))
    feature_cols = numeric_feature_columns(panel)
    train_mask = panel["calendar_week"].isin(splits.train_weeks)
    cov_scaler = CovariateScaler().fit(panel.loc[train_mask], feature_cols)
    panel = cov_scaler.transform(panel)
    return panel, feature_cols, splits, cov_scaler


def train_cbmt(config: dict) -> dict:
    set_seed(int(config["model"].get("seed", 123)))
    outdir = ensure_dir(config["data"]["output_dir"])
    model_dir = ensure_dir(outdir / "models")
    panel, feature_cols, splits, cov_scaler = prepare_model_tables(config)
    lookback = int(config["model"].get("lookback_weeks", 20))
    train_ds = WindowedCohortDataset(panel, feature_cols, splits.train_weeks, lookback)
    val_ds = WindowedCohortDataset(panel, feature_cols, splits.val_weeks, lookback)
    batch_size = int(config["model"].get("batch_size", 512))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    input_dim = len(feature_cols) + 7
    model = CBMTTransformer(input_dim=input_dim, **{k: config["model"][k] for k in ["d_model", "n_heads", "n_encoder_layers", "dropout", "head_hidden_dim"] if k in config["model"]})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    wd = float(config["model"].get("weight_decay", 1e-4))
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": config["model"].get("lr_backbone", 3e-4), "weight_decay": wd},
        {"params": model.visit_head.parameters(), "lr": config["model"].get("lr_heads", 1e-3), "weight_decay": wd},
        {"params": model.txn_head.parameters(), "lr": config["model"].get("lr_heads", 1e-3), "weight_decay": wd},
        {"params": model.payment_head.parameters(), "lr": config["model"].get("lr_heads", 1e-3), "weight_decay": wd * config["model"].get("aov_weight_decay_mult", 1000)},
        {"params": model.aggregate_heads.parameters(), "lr": config["model"].get("lr_heads", 1e-3), "weight_decay": wd},
        {"params": model.input_projection.parameters(), "lr": config["model"].get("lr_backbone", 3e-4), "weight_decay": wd},
    ])
    best = float("inf")
    patience = int(config["model"].get("patience", 20))
    bad = 0
    curves = []
    for epoch in range(1, int(config["model"].get("max_epochs", 200)) + 1):
        model.train(); train_losses=[]
        for batch in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            x, y, cons = batch["x_seq"].to(device), batch["y"].to(device), batch["consistency"].to(device)
            opt.zero_grad(); pred = model(x); loss, _ = cbmt_loss(pred, y, cons, config["model"].get("loss_weights", {})); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["model"].get("gradient_clip_norm", 1.0)))
            opt.step(); train_losses.append(float(loss.detach().cpu()))
        model.eval(); val_losses=[]
        with torch.no_grad():
            for batch in val_loader:
                loss, _ = cbmt_loss(model(batch["x_seq"].to(device)), batch["y"].to(device), batch["consistency"].to(device), config["model"].get("loss_weights", {}))
                val_losses.append(float(loss.cpu()))
        train_loss = sum(train_losses) / max(len(train_losses), 1); val_loss = sum(val_losses) / max(len(val_losses), 1)
        curves.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best:
            best = val_loss; bad = 0
            torch.save({"model_state_dict": model.state_dict(), "input_dim": input_dim, "feature_cols": feature_cols, "config": config}, model_dir / "cbmt_best.pt")
        else:
            bad += 1
        if bad >= patience:
            break
    pd.DataFrame(curves).to_csv(outdir / "training_curve.csv", index=False)
    joblib.dump({"covariate_scaler": cov_scaler, "feature_cols": feature_cols}, model_dir / "scalers.pkl")
    joblib.dump({"feature_cols": feature_cols}, model_dir / "encoders.pkl")
    save_config(config, model_dir / "config_used.yaml")
    diag = {"best_val_loss": best, "device": str(device), "train_rows": len(train_ds), "val_rows": len(val_ds), "holdout_weeks": [str(w.date()) for w in splits.holdout_weeks]}
    write_json(diag, model_dir / "validation_metrics.json")
    return diag
