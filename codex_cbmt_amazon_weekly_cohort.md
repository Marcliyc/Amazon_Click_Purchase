# Codex Implementation Spec: CBMT-Style Multi-Task Learning for Amazon Weekly Cohort Forecasting

## Goal

Implement a Customer-Based Multi-Task Transformer (CBMT)-style forecasting pipeline for my Amazon browsing/session dataset. The method should adapt the paper's idea of jointly forecasting upstream customer behavior processes and downstream revenue outcomes using a shared temporal representation plus task-specific heads.

For my Amazon data, implement the method at **customer cohort x calendar week** granularity, using available time-invariant covariates such as household, geography, and other customer attributes. Use **total product price** as the purchase/payment amount. Forecast weekly holdout outcomes:

1. **Total visits** by week.
2. **Total transactions** by week.
3. **Total revenue / transaction volume** by week, where revenue is based on `totalprice` or an equivalent total product price field.

The final deliverable should include reproducible preprocessing, model training, walk-forward forecasting, evaluation, plots, and saved prediction tables.

---

## Background and Modeling Adaptation

The paper models cohort-level customer behavior using a multi-task Transformer. It jointly predicts upstream behavioral primitives such as acquisition, repeat purchasing, and spending/order value, and uses downstream revenue consistency as an auxiliary objective. Adapt this to Amazon as follows:

### Original CBMT-style quantities

- `Acquisition`: number of newly acquired customers in cohort week.
- `ROPC`: repeat orders per cohort member in later weeks.
- `AOV`: average order value conditional on purchase.
- `Aggregate sales`: downstream weekly revenue.
- `Cohort sales = cohort_size * orders_per_customer * AOV`.

### Amazon adaptation

Use weekly acquisition cohorts based on the first observed Amazon activity or first observed purchase depending on data availability. Prefer first observed Amazon visit/session if the dataset contains all visits; otherwise use first observed transaction.

For cohort `i` and calendar week `t`, define:

- `cohort_week`: week when customer first appears in Amazon data.
- `calendar_week`: week of observation.
- `tenure_week = calendar_week - cohort_week` in weeks.
- `cohort_size_i`: number of unique customers/users/machines in cohort `i`.
- `visits_i_t`: number of Amazon visits/sessions by cohort `i` in week `t`.
- `transactions_i_t`: number of transactions by cohort `i` in week `t`.
- `revenue_i_t`: sum of `totalprice` over transactions by cohort `i` in week `t`.
- `visits_per_customer_i_t = visits_i_t / cohort_size_i`.
- `transactions_per_customer_i_t = transactions_i_t / cohort_size_i`.
- `avg_payment_i_t = revenue_i_t / max(transactions_i_t, 1)`.
- `revenue_per_customer_i_t = revenue_i_t / cohort_size_i`.

Use the model to forecast the following cohort-week upstream-ish processes:

1. `x_visit_i_t = visits_per_customer_i_t`
2. `x_txn_i_t = transactions_per_customer_i_t`
3. `x_pay_i_t = avg_payment_i_t`

Also forecast the downstream/stochastic aggregate weekly series:

4. `z_visit_t = total visits in week t`
5. `z_txn_t = total transactions in week t`
6. `z_rev_t = total revenue in week t`

The key deterministic consistency relation should be:

```text
pred_revenue_i_t = cohort_size_i * pred_transactions_per_customer_i_t * pred_avg_payment_i_t
pred_transactions_i_t = cohort_size_i * pred_transactions_per_customer_i_t
pred_visits_i_t = cohort_size_i * pred_visits_per_customer_i_t
```

Aggregate weekly predictions are:

```text
pred_total_visits_t = sum_i pred_visits_i_t
pred_total_transactions_t = sum_i pred_transactions_i_t
pred_total_revenue_t = sum_i pred_revenue_i_t
```

Important: the user asks to “predict the visits, transactions, and transaction volumes.” Interpret “transaction volumes” as total revenue/payment volume unless there is an explicit quantity/unit field. If there is a product quantity field, also output an optional `total_units` target, but do not make it required.

---

## Expected Input Data

Assume a raw Amazon session/transaction table with fields similar to:

- Customer/session IDs:
  - `machine_id`
  - `user_session_id`
  - `site_session_id`
- Time:
  - `event_date` or `event_time`
- Site/domain/product:
  - `domain_id`
  - `prod_*` fields, if present
- Visit/session behavior:
  - `pages_viewed`
  - `duration`
- Transaction:
  - `tran_flg` or equivalent transaction flag
  - `totalprice` or equivalent total product price
- Time-invariant covariates:
  - household covariates
  - geographic covariates
  - demographic/customer-level stable attributes

Implement flexible column mapping via a YAML config so I can adjust names without changing code.

Example config:

```yaml
data:
  raw_path: data/amazon.csv
  output_dir: outputs/cbmt_amazon
  customer_id_col: machine_id
  session_id_col: site_session_id
  date_col: event_date
  transaction_flag_col: tran_flg
  payment_col: totalprice
  pages_col: pages_viewed
  duration_col: duration
  covariate_cols:
    - household_income
    - household_size
    - state
    - zip3
    - region
    - device_type

split:
  baseline_start: null
  train_start: null
  val_weeks: 12
  holdout_weeks: 12
  week_start: SUN

model:
  lookback_weeks: 20
  d_model: 128
  n_heads: 4
  n_encoder_layers: 2
  n_decoder_layers: 1
  dropout: 0.1
  head_hidden_dim: 128
  batch_size: 512
  max_epochs: 200
  patience: 20
  lr_backbone: 0.0003
  lr_heads: 0.001
  weight_decay: 0.0001
  aov_weight_decay_mult: 1000
  loss_weights:
    visits_pc: 1.0
    txns_pc: 1.0
    avg_payment: 1.0
    agg_visits: 1.0
    agg_txns: 1.0
    agg_revenue: 1.0
    revenue_consistency: 4.0
    visit_consistency: 1.0
    txn_consistency: 1.0
```

---

## Repository Structure

Create the following files:

```text
.
├── README.md
├── requirements.txt
├── configs/
│   └── amazon_cbmt.yaml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocess.py
│   ├── cohort_builder.py
│   ├── covariates.py
│   ├── scalers.py
│   ├── dataset.py
│   ├── model_cbmt.py
│   ├── losses.py
│   ├── train.py
│   ├── forecast.py
│   ├── evaluate.py
│   ├── plots.py
│   └── utils.py
├── scripts/
│   ├── 01_build_weekly_cohorts.py
│   ├── 02_train_cbmt.py
│   ├── 03_forecast_holdout.py
│   └── 04_evaluate_and_plot.py
└── tests/
    ├── test_cohort_builder.py
    ├── test_dataset_shapes.py
    └── test_revenue_consistency.py
```

---

## Step 1: Data Cleaning and Weekly Panel Construction

Implement `src/data_preprocess.py` and `src/cohort_builder.py`.

### Cleaning rules

1. Parse the date column into pandas datetime.
2. Drop rows with missing customer ID or date.
3. Set invalid/negative `totalprice` to missing or zero depending on context:
   - If `tran_flg == 1` and `totalprice` is missing, set it to 0 but flag it in diagnostics.
   - If `tran_flg == 0`, force `totalprice = 0`.
4. Deduplicate carefully:
   - If multiple rows represent product lines within the same transaction/session, aggregate to session level first.
   - Do not double-count transactions if `tran_flg` is repeated across product rows.
5. Define `calendar_week` using weekly periods. Make the start day configurable.

### Visit/session level aggregation

If raw rows are product/page rows, first aggregate to `customer_id x session_id x calendar_week`:

- `visit = 1`
- `pages_viewed = max/sum pages_viewed`, configurable but default to max if session-level already repeated.
- `duration = max/sum duration`, configurable.
- `transaction = max(tran_flg)`.
- `payment = sum(totalprice)` over purchased products. If session has no transaction, payment is 0.

### Cohort definition

Define each customer's cohort as:

```text
cohort_week = min(calendar_week where customer has an Amazon visit/session)
```

Then construct `cohort_size` as the number of distinct customers in each `cohort_week`.

Also support an optional `cohort_definition: first_purchase` mode:

```text
cohort_week = min(calendar_week where customer has tran_flg == 1)
```

Default to `first_visit` because the user wants to predict visits as well as transactions.

### Weekly cohort panel

Construct a balanced cohort-week panel for all `cohort_week <= calendar_week <= max_week`.

For each `cohort_week i` and `calendar_week t`:

- `cohort_size_i`
- `tenure_week`
- `visits_i_t`
- `transactions_i_t`
- `revenue_i_t`
- `visits_per_customer_i_t`
- `transactions_per_customer_i_t`
- `avg_payment_i_t`
- `revenue_per_customer_i_t`

Add zero rows for cohort-week cells with no activity:

- visits = 0
- transactions = 0
- revenue = 0
- visits per customer = 0
- transactions per customer = 0
- avg payment = 0

### Aggregate weekly targets

Create a separate weekly aggregate table:

- `week`
- `total_visits`
- `total_transactions`
- `total_revenue`
- optional `active_customers`
- optional `conversion_rate = total_transactions / total_visits`
- optional `avg_payment = total_revenue / total_transactions`

---

## Step 2: Covariates

Implement `src/covariates.py`.

### Deterministic time covariates known at forecast time

Add:

- week-of-year categorical embedding index
- month categorical embedding index
- quarter categorical embedding index
- year index or normalized calendar trend
- linear time trend
- quadratic time trend
- holiday flags, at least:
  - Black Friday week
  - Cyber Monday week
  - Thanksgiving week
  - Christmas week
  - Prime Day if identifiable or manually passed as dates
- tenure week
- tenure week squared
- log1p tenure week
- acquisition month
- acquisition week-of-year

### Customer/cohort covariates

Use the time-invariant covariates provided by the user. Since covariates are customer-level but the panel is cohort-level, aggregate them to cohort-level features:

For numeric covariates:

- mean
- median
- standard deviation where useful
- missing rate

For categorical covariates:

- mode / top category
- category proportions for low-cardinality columns
- target-safe frequency encoding based only on training data, if cardinality is high

Important: avoid leakage. Fit encoders and scalers only on the train period and apply to validation/holdout.

### Optional behavioral covariates from browsing

Include cohort-week behavioral summaries as inputs if available and if they are known up to the forecast origin:

- mean/sum `pages_viewed`
- mean/sum `duration`
- visits with product page views
- product category mix from `prod_*` fields

These should be lagged/autoregressive like the behavioral processes, not future-known deterministic covariates.

---

## Step 3: Train/Validation/Holdout Split

Implement temporal splits:

1. Sort all weeks.
2. Hold out the final `holdout_weeks` weeks.
3. Use the preceding `val_weeks` weeks for validation.
4. Use all prior weeks for training.

The model should never train on holdout actual outcomes.

Output split diagnostics:

```text
train weeks: ... to ...
validation weeks: ... to ...
holdout weeks: ... to ...
number of cohorts observed in train
number of cold cohorts first appearing in holdout
number of rows in cohort-week panel
```

---

## Step 4: Windowed Dataset

Implement `src/dataset.py`.

Use an autoregressive lookback window of `p = 20` weeks by default, corresponding to the paper's 20-week history.

For each cohort `i` and forecast origin week `t`, create input window:

```text
D_i_t = {
  behavioral_history: [x_visit, x_txn, x_pay, z_visit, z_txn, z_rev] for weeks t-p+1 ... t,
  deterministic_covariates: C_i_t for weeks t-p+1 ... t,
  cohort_covariates: static cohort features,
}
```

Targets for `t+1`:

```text
y_visit_pc_i_t1
y_txn_pc_i_t1
y_avg_payment_i_t1
y_agg_visits_t1
y_agg_txns_t1
y_agg_revenue_t1
y_cohort_visits_i_t1
y_cohort_txns_i_t1
y_cohort_revenue_i_t1
```

### Zero padding

For pre-birth weeks before a cohort exists, pad behavior with zeros and include an `is_prebirth` indicator. For newly acquired/cold cohorts, histories are all zeros except deterministic and cohort-level covariates if known.

### Scaling

Implement train-fitted scalers:

- Use `log1p` transform for count/revenue-like quantities before MinMax or Standard scaling:
  - visits per customer
  - transactions per customer
  - average payment
  - aggregate visits
  - aggregate transactions
  - aggregate revenue
- Save scalers to disk.
- The model can output transformed values. Inverse-transform before computing consistency losses in original scale, then optionally re-transform the consistency target to stabilize gradients.

Use nonnegative outputs. Prefer `softplus` or clamp after inverse transform. Avoid negative visits, transactions, or revenue.

---

## Step 5: Model Architecture

Implement `src/model_cbmt.py` in PyTorch.

### Shared encoder

Inputs:

- sequence of numeric behavioral features
- sequence of deterministic time/tenure features
- categorical embeddings for week/month/acquisition period/other categorical covariates
- static cohort covariates broadcast across the sequence or injected after pooling

Architecture:

1. Numeric projection layer into `d_model`.
2. Categorical embeddings concatenated/projection into `d_model`.
3. Add sinusoidal or learned positional encoding.
4. Transformer encoder with:
   - `n_encoder_layers`
   - `n_heads`
   - feed-forward dim `4 * d_model`
   - dropout
   - batch-first tensors.
5. Use the final token representation or attention pooling over the sequence as shared representation `h_i_t`.

### Task-specific heads

Create separate 3-layer MLP heads for:

1. `visits_per_customer`
2. `transactions_per_customer`
3. `avg_payment`
4. `aggregate_visits`
5. `aggregate_transactions`
6. `aggregate_revenue`

Each head receives:

```text
concat(shared_representation, task_specific_lag_history_summary, static_cohort_features)
```

For task-specific lag history summary, include last value, mean over window, slope over window, and possibly the flattened last few lags.

Use `softplus` on outputs if predicting in original scale, or output unconstrained values if predicting transformed/scaled targets and inverse-transform later.

---

## Step 6: Losses

Implement `src/losses.py`.

Base losses are MSE or Huber losses for the six heads:

```text
L_visit_pc
L_txn_pc
L_avg_payment
L_agg_visits
L_agg_txns
L_agg_revenue
```

Add consistency losses in original scale:

```text
pred_cohort_visits = cohort_size * pred_visits_per_customer
pred_cohort_txns = cohort_size * pred_transactions_per_customer
pred_cohort_revenue = cohort_size * pred_transactions_per_customer * pred_avg_payment
```

Consistency losses:

```text
L_visit_consistency = MSE(log1p(pred_cohort_visits), log1p(actual_cohort_visits))
L_txn_consistency = MSE(log1p(pred_cohort_txns), log1p(actual_cohort_txns))
L_revenue_consistency = MSE(log1p(pred_cohort_revenue), log1p(actual_cohort_revenue))
```

Optionally add aggregate consistency for each batch/week if batch construction supports grouping by target week:

```text
sum_i pred_cohort_visits_i_t ≈ actual_total_visits_t
sum_i pred_cohort_txns_i_t ≈ actual_total_transactions_t
sum_i pred_cohort_revenue_i_t ≈ actual_total_revenue_t
```

Total loss:

```text
L = w1 * L_visit_pc
  + w2 * L_txn_pc
  + w3 * L_avg_payment
  + w4 * L_agg_visits
  + w5 * L_agg_txns
  + w6 * L_agg_revenue
  + w7 * L_visit_consistency
  + w8 * L_txn_consistency
  + w9 * L_revenue_consistency
```

Use config weights. Default revenue consistency should be relatively high, e.g. 4.0, because weekly revenue is the final key target.

---

## Step 7: Training

Implement `src/train.py` and `scripts/02_train_cbmt.py`.

Requirements:

- AdamW optimizer.
- Lower LR for Transformer/shared backbone and higher LR for heads.
- Stronger weight decay for the avg-payment head, because payment/order value is likely noisy.
- Gradient clipping.
- Early stopping on validation aggregate revenue sMAPE or combined validation loss.
- Save:
  - best model checkpoint
  - config used
  - scalers
  - covariate encoders
  - training curve CSV
  - validation metrics JSON

Recommended default optimizer groups:

```python
optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": lr_backbone, "weight_decay": weight_decay},
    {"params": model.visit_head.parameters(), "lr": lr_heads, "weight_decay": weight_decay},
    {"params": model.txn_head.parameters(), "lr": lr_heads, "weight_decay": weight_decay},
    {"params": model.payment_head.parameters(), "lr": lr_heads, "weight_decay": weight_decay * 1000},
    {"params": model.aggregate_heads.parameters(), "lr": lr_heads, "weight_decay": weight_decay},
])
```

---

## Step 8: Walk-Forward Holdout Forecasting

Implement `src/forecast.py` and `scripts/03_forecast_holdout.py`.

Use rolling-origin one-step-ahead forecasting:

1. At first holdout week, use the most recent observed train/validation histories.
2. Predict one week ahead for all active cohorts.
3. For weeks after the first holdout week, append predictions to the history wherever actual values are unavailable.
4. For new/cold cohorts in holdout, create zero behavioral histories and deterministic covariates. If actual cohort size is not known at forecast time, forecast cohort size/acquisition using the model or use predicted new customers from the visit/acquisition process. For this implementation:
   - If first-visit cohort sizes are known in evaluation only, do not leak them into forecasting unless `oracle_cohort_size_for_holdout: true`.
   - Default non-leaky mode: train an auxiliary weekly new-customer/cohort-size head or simple baseline model to forecast new cohort sizes.
5. Generate cohort-week predictions.
6. Aggregate by calendar week to weekly predictions:
   - total visits
   - total transactions
   - total revenue

### Cold cohort handling

Implement two modes:

- `oracle_holdout_cohorts = false` default:
  - forecast new cohort size for each future week using a simple seasonal/trend model or an additional model head.
  - create one new cohort each future week with predicted size.
- `oracle_holdout_cohorts = true` diagnostic mode:
  - use actual holdout cohort sizes to isolate repeat/transaction/payment forecasting performance.
  - label all plots and metrics clearly as oracle mode.

---

## Step 9: Evaluation

Implement `src/evaluate.py` and `scripts/04_evaluate_and_plot.py`.

Evaluate on the holdout set at weekly aggregate level:

- total visits
- total transactions
- total revenue

Metrics:

- sMAPE
- MAPE with epsilon guard
- MAE
- RMSE
- WAPE
- MASE if a seasonal naive baseline is implemented

Also evaluate cohort-week level:

- visits per customer
- transactions per customer
- average payment
- cohort visits
- cohort transactions
- cohort revenue

Baselines:

1. Seasonal naive: same value as 52 weeks ago if available, else same as 4 weeks ago.
2. Rolling mean baseline: mean of last 4 or 8 weeks.
3. Optional LightGBM baseline with lag features for weekly aggregate outcomes.

Output:

```text
outputs/cbmt_amazon/
├── predictions/
│   ├── holdout_weekly_predictions.csv
│   ├── holdout_cohort_week_predictions.csv
│   └── holdout_metrics.json
├── models/
│   ├── cbmt_best.pt
│   ├── scalers.pkl
│   └── encoders.pkl
└── plots/
    ├── weekly_total_visits_actual_vs_pred.png
    ├── weekly_total_transactions_actual_vs_pred.png
    ├── weekly_total_revenue_actual_vs_pred.png
    ├── cohort_heatmap_actual_revenue.png
    ├── cohort_heatmap_pred_revenue.png
    ├── cohort_error_by_tenure.png
    └── segment_behavior_parameters.png
```

---

## Step 10: Plots and Interpretation

Implement `src/plots.py`.

Required plots:

1. Actual vs predicted weekly total visits on holdout.
2. Actual vs predicted weekly total transactions on holdout.
3. Actual vs predicted weekly total revenue on holdout.
4. Cohort-week heatmap of actual revenue.
5. Cohort-week heatmap of predicted revenue.
6. Error by tenure week.
7. Segment-level interpretation plot using covariates:
   - Choose several meaningful segments from household/geographic covariates.
   - For each segment, plot predicted visits per customer, transactions per customer, avg payment, and revenue per customer over tenure.

The segment plot is important because I want to interpret how behavior changes across model inputs/covariates.

If the Transformer is too hard to interpret directly, use post-hoc segment simulations:

1. Select representative covariate profiles, e.g. high-income vs low-income, urban vs suburban, region, etc.
2. Construct synthetic cohort feature vectors while holding calendar/tenure fixed.
3. Feed them through the trained model with identical behavioral histories.
4. Compare predicted trajectories.

---

## Step 11: CLI Commands

The pipeline should run as:

```bash
python scripts/01_build_weekly_cohorts.py --config configs/amazon_cbmt.yaml
python scripts/02_train_cbmt.py --config configs/amazon_cbmt.yaml
python scripts/03_forecast_holdout.py --config configs/amazon_cbmt.yaml
python scripts/04_evaluate_and_plot.py --config configs/amazon_cbmt.yaml
```

Also support a single end-to-end command:

```bash
python -m src.run_all --config configs/amazon_cbmt.yaml
```

---

## Step 12: Requirements

Use these packages in `requirements.txt`:

```text
pandas
numpy
scikit-learn
scipy
pyyaml
matplotlib
seaborn
tqdm
joblib
torch
holidays
lightgbm
pytest
```

LightGBM is optional for baselines. The core implementation should work without GPU but use CUDA if available.

---

## Step 13: Tests

Add minimal tests:

### `test_cohort_builder.py`

- Given a tiny dataset with 3 customers across 4 weeks, verify cohort assignment.
- Verify visits, transactions, revenue, and avg payment calculations.
- Verify zero-filled cohort-week rows exist.

### `test_dataset_shapes.py`

- Verify that windowed dataset returns tensors with expected shapes:
  - sequence length = lookback weeks
  - numeric feature dimension > 0
  - targets include all six heads and consistency targets.

### `test_revenue_consistency.py`

- Given known predictions for cohort size, transactions per customer, and avg payment, verify implied cohort revenue and aggregate weekly revenue calculations.

---

## Important Implementation Details and Edge Cases

1. **Do not leak holdout actuals** into features, scalers, encoders, or cold-cohort sizes unless explicitly in oracle diagnostic mode.
2. **Use `totalprice` only as payment/revenue**, not as a feature from the future.
3. **Keep visits and transactions distinct**:
   - visits are sessions or site visits.
   - transactions are purchases.
   - conversion behavior is learned through transaction-per-customer and optionally transactions-per-visit.
4. **Use nonnegative forecasts** for visits, transactions, and revenue.
5. **Handle sparse cohorts** by zero padding and using cohort_size in consistency calculations.
6. **Save intermediate tables** so bugs can be inspected:
   - cleaned session table
   - customer cohort assignment table
   - cohort-week panel
   - aggregate-week panel
   - train/val/holdout split info
7. **Make all paths configurable**.
8. **Use random seeds** for reproducibility.
9. **Use mixed precision only if stable**; default to standard FP32.
10. **Write clear README instructions** including how to change the column mapping.

---

## Acceptance Criteria

The implementation is complete when:

1. The full pipeline runs from raw Amazon data to holdout forecasts.
2. The model outputs weekly holdout predictions for total visits, total transactions, and total revenue.
3. The model outputs cohort-week predictions for visits, transactions, and revenue.
4. The evaluation script reports sMAPE, WAPE, MAE, and RMSE for all three aggregate weekly targets.
5. Plots are saved under `outputs/cbmt_amazon/plots/`.
6. The code supports time-invariant household/geographic covariates aggregated to cohort-level features.
7. The revenue consistency loss is implemented and tested.
8. A non-leaky walk-forward forecast is implemented.
9. A small synthetic test dataset passes all tests.

---

## Suggested First Implementation Order

1. Build `cohort_builder.py` and verify cohort-week panel manually.
2. Build train/validation/holdout split and scalers.
3. Build `WindowedCohortDataset` and tests.
4. Implement simple MLP multi-task model first to validate end-to-end training.
5. Replace shared MLP with Transformer encoder.
6. Add consistency losses.
7. Add walk-forward forecasting.
8. Add evaluation/plots.
9. Add segment interpretation.
10. Add baselines.

---

## Notes for Codex

Prioritize correctness and inspectability over cleverness. Make the pipeline modular and robust to imperfect column names. Include docstrings and comments explaining the cohort construction and consistency losses. Use clear, explicit tensor names to avoid confusing visits, transactions, average payment, and revenue.
