# Amazon EV/CM Implementation

This repository implements an EV/CM (Moe–Fader style) model for Amazon browsing and purchase sessions.

## Run

```bash
python -m src.evcm_pipeline \
  --input data/amazon_sessions.csv \
  --output reports/evcm \
  --visit-unit daily \
  --freq W \
  --n-sims 300 \
  --ev-starts 5 \
  --cm-starts 5 \
  --seed 123
```

## Tests

```bash
pytest
```


## JAX/CUDA12 run

```bash
# Optional: install GPU JAX wheel
JAX_ACCELERATOR=cuda12 bash scripts/setup_codex_jax.sh

python -m src.evcm_pipeline \
  --input data/amazon_sessions.csv \
  --output reports/evcm_jax \
  --visit-unit daily \
  --freq W \
  --n-sims 300 \
  --ev-starts 5 \
  --cm-starts 5 \
  --engine jax \
  --x64 \
  --seed 123
```

## CBMT-style Amazon weekly cohort forecasting

This repository also includes a Customer-Based Multi-Task Transformer (CBMT)-style pipeline for forecasting Amazon cohort-week visits, transactions, and revenue/payment volume.

### Configure columns

Edit `configs/amazon_cbmt.yaml` to map your raw file and columns:

```yaml
data:
  raw_path: data/amazon_sessions.csv
  customer_id_col: machine_id
  session_id_col: site_session_id
  date_col: event_date
  transaction_flag_col: tran_flg
  payment_col: prod_totprice   # use totalprice if that is your total product price field
  covariate_cols:
    - household_income
    - household_size
    - census_region
```

The default cohort definition is `first_visit`, which supports visit forecasting. Set `data.cohort_definition: first_purchase` to cohort customers by their first transaction instead.

### Run the CBMT pipeline

```bash
python scripts/01_build_weekly_cohorts.py --config configs/amazon_cbmt.yaml
python scripts/02_train_cbmt.py --config configs/amazon_cbmt.yaml
python scripts/03_forecast_holdout.py --config configs/amazon_cbmt.yaml
python scripts/04_evaluate_and_plot.py --config configs/amazon_cbmt.yaml
```

Or run end to end:

```bash
python -m src.run_all --config configs/amazon_cbmt.yaml
```

Outputs are written under `outputs/cbmt_amazon/`:

- `cleaned_sessions.csv`, `customer_cohorts.csv`, `cohort_week_panel.csv`, and `aggregate_week_panel.csv`
- `models/cbmt_best.pt`, `models/scalers.pkl`, `models/encoders.pkl`, `training_curve.csv`, and validation diagnostics
- `predictions/holdout_weekly_predictions.csv`, `predictions/holdout_cohort_week_predictions.csv`, and `predictions/holdout_metrics.json`
- plots for holdout actual-vs-predicted totals, cohort revenue heatmaps, tenure error, and segment-style behavior interpretation

The forecasting script performs rolling one-step-ahead holdout prediction and feeds predictions into later holdout histories rather than using future actual outcomes. By default it avoids oracle holdout cohorts; set `split.oracle_holdout_cohorts: true` only for diagnostic isolation of repeat behavior/payment forecasting.
