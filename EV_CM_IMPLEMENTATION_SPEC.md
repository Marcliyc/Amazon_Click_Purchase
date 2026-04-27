# EV/CM Model Implementation Specification for Amazon Clickstream Data

This document is a task specification for implementing the Moe–Fader EV/CM framework on the Amazon browsing dataset in this repository.

The goal is to write production-quality Python code that:

1. loads and cleans the Amazon clickstream dataset,
2. constructs customer-level visit and purchase histories,
3. splits the data into calibration and holdout periods without leakage,
4. fits the EV model for evolving visit timing,
5. fits the CM model for dynamic conversion conditional on visits,
6. reports log-likelihood and model-fit statistics,
7. forecasts cumulative purchases over the holdout period using EV/CM simulation, and
8. reports MAPE and related forecast diagnostics.

The implementation should be self-contained and reproducible from the command line.

---

## 1. Dataset schema

The input data is a table with approximately 50,722 rows and the following columns:

```text
machine_id           int64
site_session_id      int64
user_session_id      int64
domain_id            int64
pages_viewed         int64
duration             int64
event_date           object
event_time           object
tran_flg             float64, mostly missing
prod_category_id     float64, mostly missing
prod_name            object, mostly missing
prod_qty             float64, mostly missing
prod_totprice        float64, mostly missing
basket_tot           float64, mostly missing
census_region        int64
household_size       int64
household_income     int64
racial_background    int64
country_of_origin    int64
domain_name          object
```

Primary modeling identifiers:

- Customer / household / panelist: `machine_id`
- Visit/session identifier: `site_session_id`
- Timestamp: `event_date` + `event_time`
- Purchase indicator: `tran_flg`
- Optional transaction/revenue fields: `prod_qty`, `prod_totprice`, `basket_tot`

The EV/CM model predicts purchase incidence, not basket size or revenue. Use a binary purchase indicator as the primary outcome.

---

## 2. Output repository structure

Create or modify the repository to include the following structure:

```text
.
├── data/
│   └── amazon_clickstream.csv              # user-provided raw data, not committed if private
├── src/
│   ├── __init__.py
│   ├── data_prep.py                        # loading, cleaning, visit construction, split
│   ├── ev_model.py                         # evolving visit model likelihood and simulation
│   ├── cm_model.py                         # dynamic conversion model likelihood
│   ├── forecasting.py                      # EV/CM holdout simulation and aggregation
│   ├── metrics.py                          # LL, AIC, BIC, MAPE, forecast diagnostics
│   └── evcm_pipeline.py                    # CLI entry point
├── notebooks/
│   └── 01_evcm_diagnostics.ipynb           # optional, generated from saved outputs
├── reports/
│   └── evcm/
│       ├── metrics.json
│       ├── params_ev.csv
│       ├── params_cm.csv
│       ├── forecast_by_period.csv
│       ├── visit_history_summary.csv
│       ├── purchase_history_summary.csv
│       └── figures/
│           ├── cumulative_purchases_forecast.png
│           ├── weekly_purchases_forecast.png
│           ├── cumulative_visits_forecast.png
│           └── calibration_fit_diagnostics.png
├── tests/
│   ├── test_data_prep.py
│   ├── test_ev_model.py
│   ├── test_cm_model.py
│   └── test_forecasting.py
├── requirements.txt
└── README.md
```

Required dependencies:

```text
numpy
pandas
scipy
matplotlib
tqdm
pytest
```

Optional but useful:

```text
numdifftools
joblib
pyarrow
```

---

## 3. Data preprocessing

### 3.1 Load and timestamp construction

Implement:

```python
load_raw_data(path: str | Path) -> pd.DataFrame
```

Steps:

1. Read CSV or parquet.
2. Combine `event_date` and `event_time` into `event_datetime`.
3. Convert `event_date` to calendar date.
4. Sort by `machine_id`, `event_datetime`, `site_session_id`.
5. Keep the original index only as metadata; do not use it for modeling.

Example:

```python
df["event_datetime"] = pd.to_datetime(
    df["event_date"].astype(str) + " " + df["event_time"].astype(str),
    errors="coerce",
)
```

Raise an informative error if any timestamp cannot be parsed.

### 3.2 Purchase flag construction

Define a row-level purchase flag:

```python
df["purchase_row"] = df["tran_flg"].fillna(0).astype(float) > 0
```

Also treat a row as purchase-related if `basket_tot`, `prod_totprice`, or `prod_qty` is non-null and positive, but use `tran_flg` as the primary signal if available.

Recommended robust rule:

```python
df["purchase_row"] = (
    (df["tran_flg"].fillna(0).astype(float) > 0)
    | (df["basket_tot"].fillna(0).astype(float) > 0)
    | (df["prod_totprice"].fillna(0).astype(float) > 0)
    | (df["prod_qty"].fillna(0).astype(float) > 0)
)
```

### 3.3 Session-level deduplication

Raw rows may contain multiple product rows for the same session. First aggregate to one row per `machine_id, site_session_id`.

Implement:

```python
make_session_visits(df: pd.DataFrame) -> pd.DataFrame
```

Session aggregation rules:

- `machine_id`: group key
- `site_session_id`: group key
- `visit_datetime`: minimum `event_datetime` in the session
- `visit_date`: calendar date of `visit_datetime`
- `purchase`: max/any of `purchase_row`
- `purchase_session_count`: 1 if purchase, else 0
- `pages_viewed`: max or sum; prefer max if duplicated product rows are suspected
- `duration`: max or sum; prefer max if duplicated product rows are suspected
- `basket_tot`: max, not sum, because basket total may be repeated on product rows
- `prod_totprice`: sum across product rows
- demographic fields: first non-null value

### 3.4 Daily visit aggregation

The original Moe–Fader application treats the customer history as a sequence of days with visits. Default to daily aggregation to avoid zero-length intervisit gaps when a customer has multiple sessions on the same day.

Implement:

```python
make_daily_visits(session_df: pd.DataFrame) -> pd.DataFrame
```

Daily aggregation rules:

- one row per `machine_id, visit_date`
- `visit_datetime`: earliest session timestamp that day
- `t`: numeric time in days from the global start date
- `purchase`: 1 if any session that day purchased, else 0
- `purchase_session_count`: number of purchasing sessions that day
- `pages_viewed`: sum over sessions that day
- `duration`: sum over sessions that day
- `basket_tot`: sum of unique session-level basket totals or sum of session-level `basket_tot`
- `prod_totprice`: sum over session-level product totals

The primary model outcome is `purchase`, a binary purchase-visit indicator. Keep `purchase_session_count` and revenue fields for optional reporting only.

Allow a CLI option:

```text
--visit-unit daily   # default
--visit-unit session # optional, uses fractional days from event_datetime
```

If `--visit-unit session` is used, ensure all intervisit gaps are strictly positive. If multiple visits have identical timestamps, jitter by a tiny epsilon or aggregate them.

---

## 4. Calibration/holdout split

Use a global time-based split. Do not randomly split rows, because that leaks future customer history into training.

Implement:

```python
split_calibration_holdout(
    visits: pd.DataFrame,
    cutoff: str | None = None,
    calibration_fraction: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]
```

Rules:

1. If `--cutoff YYYY-MM-DD` is provided, use that date as the end of calibration.
2. Otherwise, use the first 50% of the observed date range as calibration and the remaining 50% as holdout.
3. Report:
   - global start date,
   - calibration end date,
   - holdout end date,
   - number of machines in calibration,
   - number of known machines with holdout visits,
   - number of new machines appearing only in holdout.

For forecasting, use only machines with at least one calibration visit as the primary evaluation set. New machines in holdout require an acquisition model and should be excluded from the primary EV/CM forecast. Still report their actual purchases separately under `new_machine_holdout_purchases`.

---

## 5. EV model: Evolving Visit Timing

### 5.1 Model definition

Use the uncorrelated four-parameter EV model as the required implementation.

For customer `i`, visit rate evolves over visits:

```text
lambda_i1 ~ Gamma(r, alpha)
intervisit_gap_ij | lambda_ij ~ Exponential(lambda_ij)
lambda_i,j+1 = lambda_ij * c_ij
c_ij ~ Gamma(s, beta)
```

Important convention:

- Use Gamma(shape, rate), not shape/scale.
- In NumPy/SciPy random draws, `scale = 1 / rate`.

The four EV parameters are:

```text
r      > 0   initial Gamma shape for visit rate
alpha  > 0   initial Gamma rate for visit rate
s      > 0   Gamma shape for visit-rate multiplier
beta   > 0   Gamma rate for visit-rate multiplier
```

### 5.2 EV likelihood recursion

For each customer, condition on the first observed visit. Model repeat intervisit gaps only.

Let current posterior approximation be:

```text
lambda ~ Gamma(r_cur, alpha_cur)
```

For an observed intervisit gap `d > 0`, the gamma-exponential mixture density is:

```text
f(d | r_cur, alpha_cur)
= r_cur * alpha_cur^r_cur / (alpha_cur + d)^(r_cur + 1)
```

Log density:

```text
log_f = log(r_cur) + r_cur * log(alpha_cur) - (r_cur + 1) * log(alpha_cur + d)
```

After observing an arrival with gap `d`, Bayesian update before the multiplier:

```text
r_arrival     = r_cur + 1
alpha_arrival = alpha_cur + d
```

Then apply the stochastic multiplier update using moment matching for the product of two independent Gamma variables:

```text
r_next     = (r_arrival * s) / (r_arrival + s + 1)
alpha_next = (alpha_arrival * beta) / (r_arrival + s + 1)
```

Right-censoring survival from last observed visit to period end with censor gap `c >= 0`:

```text
S(c | r_cur, alpha_cur) = (alpha_cur / (alpha_cur + c))^r_cur
log_S = r_cur * (log(alpha_cur) - log(alpha_cur + c))
```

When returning the posterior state at the calibration cutoff for forecasting, condition on no visit during the censoring interval:

```text
r_post = r_cur
alpha_post = alpha_cur + c
```

Do not apply the multiplier after a censoring interval, because no arrival occurred.

### 5.3 EV functions

Implement:

```python
@dataclass
class EVParams:
    r: float
    alpha: float
    s: float
    beta: float

@dataclass
class EVState:
    r: float
    alpha: float
    last_t: float
    total_visits_seen: int

def ev_customer_loglik_and_state(
    times: np.ndarray,
    T_end: float,
    params: EVParams,
    return_state_at_end: bool = True,
) -> tuple[float, EVState]

def ev_loglik(
    visits: pd.DataFrame,
    T_end: float,
    params: EVParams,
) -> float

def fit_ev_model(
    visits_cal: pd.DataFrame,
    T_cal_end: float,
    n_starts: int = 20,
    seed: int = 123,
) -> tuple[EVParams, dict]
```

Use transformed parameters for optimization:

```text
r     = softplus(theta_r)     + eps
alpha = softplus(theta_alpha) + eps
s     = softplus(theta_s)     + eps
beta  = softplus(theta_beta)  + eps
```

Optimization:

- Use `scipy.optimize.minimize(..., method="L-BFGS-B")`.
- Run multiple random starts.
- Return the best result.
- Report convergence status, objective value, gradient norm if available, and parameter estimates.

### 5.4 EV holdout likelihood

Implement optional but recommended:

```python
ev_holdout_loglik(
    visits_cal: pd.DataFrame,
    visits_holdout: pd.DataFrame,
    T_cal_end: float,
    T_holdout_end: float,
    params: EVParams,
) -> float
```

For each known machine:

1. Compute posterior EV state at `T_cal_end` using calibration visits and calibration censoring.
2. Evaluate holdout intervisit gaps from `T_cal_end` to the first holdout visit, then between holdout visits.
3. Add survival from last holdout visit to `T_holdout_end`.

---

## 6. CM model: Dynamic Conversion Conditional on Visits

### 6.1 Model definition

The CM model predicts whether visit `j` results in a purchase, conditional on the visit occurring.

The full six-parameter CM model:

```text
r_v    > 0      baseline purchase propensity
mu0    > 0      initial incremental visit effect
k      > 0      evolution of visit effects
r_tau  > 0      initial purchasing threshold
psi    real     threshold evolution with prior purchases
pi     in (0,1) fraction of non-hard-core-never-buyers
```

For customer `i` at visit `j`, using 1-based visit index:

```text
x_ij  = number of prior purchase visits before visit j
n_ij  = number of prior visits before visit j = j - 1
lp_ij = index of most recent prior purchase visit, or 0 if none
```

The net visit-effect shape is:

```text
a_ij = r_v + sum_{u = lp_ij + 1}^{j} mu0 * k^u
```

The threshold shape is:

```text
b_ij = r_tau * exp(psi * x_ij)
```

The baseline beta-Bernoulli posterior mean without the hard-core never-buyer mixture is:

```text
p_base = (a_ij + x_ij) / (a_ij + b_ij + n_ij)
```

For customers with no prior purchases (`x_ij == 0`), account for hard-core never-buyers:

```text
Pr(purchase at visit j) = pi * a_ij / (a_ij + b_ij + n_ij)
Pr(no purchase at visit j) = (1 - pi) + pi * (1 - a_ij / (a_ij + b_ij + n_ij))
```

For customers with at least one prior purchase (`x_ij > 0`):

```text
Pr(purchase at visit j) = (a_ij + x_ij) / (a_ij + b_ij + n_ij)
Pr(no purchase at visit j) = 1 - Pr(purchase at visit j)
```

Clamp probabilities to `[1e-12, 1 - 1e-12]` before taking logs.

### 6.2 Stable geometric sum

Implement a numerically stable function for:

```text
sum_{u = start}^{end} mu0 * k^u
```

Use the direct sum for small ranges, but implement the closed form for speed:

```text
if abs(k - 1) < 1e-8:
    mu0 * (end - start + 1)
else:
    mu0 * k^start * (k^(end - start + 1) - 1) / (k - 1)
```

Guard against overflow if `k^j` becomes too large. A safe practical approach:

- calculate in log-space when needed,
- or constrain `log(k)` to a reasonable range during optimization, e.g. `log(k) in [-5, 5]`.

### 6.3 CM functions

Implement:

```python
@dataclass
class CMParams:
    r_v: float
    mu0: float
    k: float
    r_tau: float
    psi: float
    pi: float

@dataclass
class CMState:
    total_visits: int
    prior_purchases: int
    last_purchase_visit_index: int

def cm_purchase_probability(
    visit_index: int,
    prior_purchases: int,
    last_purchase_visit_index: int,
    params: CMParams,
) -> float

def cm_customer_loglik_and_state(
    purchase_sequence: np.ndarray,
    params: CMParams,
    initial_state: CMState | None = None,
    update_state: bool = True,
) -> tuple[float, CMState]

def cm_loglik(
    visits: pd.DataFrame,
    params: CMParams,
) -> float

def fit_cm_model(
    visits_cal: pd.DataFrame,
    n_starts: int = 30,
    seed: int = 123,
) -> tuple[CMParams, dict]
```

Optimization transforms:

```text
r_v   = softplus(theta_rv)   + eps
mu0   = softplus(theta_mu0)  + eps
k     = exp(theta_log_k)
r_tau = softplus(theta_rt)   + eps
psi   = theta_psi
pi    = sigmoid(theta_pi)
```

Use multiple starts. Reasonable starting values include:

```text
r_v    around 0.05 to 1.0
mu0    around 0.05 to 1.0
k      around 0.8 to 1.2
r_tau  around 1.0 to 10.0
psi    around -0.2 to 0.2
pi     around observed fraction of machines that ever purchase, but bounded away from 0 and 1
```

### 6.4 CM holdout likelihood

Implement:

```python
cm_holdout_loglik(
    visits_cal: pd.DataFrame,
    visits_holdout: pd.DataFrame,
    params: CMParams,
) -> float
```

For each known machine:

1. Build the CM state from calibration visits.
2. Evaluate the actual holdout purchase sequence conditional on actual holdout visits.
3. Update the state sequentially using the actual holdout outcomes.

This holdout CM likelihood measures conversion prediction conditional on actual future visits, not visit forecasting.

---

## 7. Joint EV/CM fitting

Because the EV model describes visit timing and the CM model describes conversion conditional on visits, fit them separately by default:

```python
ev_params = fit_ev_model(visits_cal, T_cal_end)
cm_params = fit_cm_model(visits_cal)
```

Then report joint log-likelihood as:

```text
LL_joint_cal = LL_EV_cal + LL_CM_cal
LL_joint_holdout = LL_EV_holdout + LL_CM_holdout
```

The main EV/CM model should have:

- 4 EV parameters for the uncorrelated EV model
- 6 CM parameters for the full conversion model
- 10 total parameters

The slide deck also refers to an 11-parameter EV/CM model when using a correlated EV extension. Do not implement the correlated EV extension unless explicitly requested. Leave a TODO stub for it.

---

## 8. Forecasting cumulative purchases

### 8.1 Forecast target

Primary forecast target:

```text
Cumulative number of purchase visits among known calibration machines during the holdout period.
```

A "purchase visit" is a visit/day with `purchase == 1`.

Also report but do not optimize against:

```text
cumulative purchase sessions
cumulative revenue / basket total
new-machine holdout purchases
```

### 8.2 EV/CM simulation

Implement:

```python
simulate_evcm_forecast(
    visits_cal: pd.DataFrame,
    T_cal_end: float,
    T_holdout_end: float,
    ev_params: EVParams,
    cm_params: CMParams,
    n_sims: int = 1000,
    freq: str = "W",
    seed: int = 123,
) -> pd.DataFrame
```

For each simulation path and each known machine:

1. Build EV posterior state at `T_cal_end`, conditioned on no additional calibration visit before cutoff.
2. Build CM state from calibration visits.
3. Draw a latent visit rate:

   ```python
   lambda_cur = rng.gamma(shape=ev_state.r, scale=1.0 / ev_state.alpha)
   ```

4. Starting at `t = T_cal_end`, simulate future visits until `T_holdout_end`:

   ```python
   gap = rng.exponential(scale=1.0 / lambda_cur)
   t_next = t + gap
   if t_next > T_holdout_end:
       break
   ```

5. At each simulated visit:
   - increment visit index,
   - compute CM purchase probability,
   - draw purchase with Bernoulli probability `p`,
   - update CM state,
   - update EV latent rate by drawing multiplier:

     ```python
     c = rng.gamma(shape=ev_params.s, scale=1.0 / ev_params.beta)
     lambda_cur *= c
     ```

6. Store simulated visit and purchase events.
7. Aggregate simulated purchases into holdout periods, e.g. weekly.
8. Average across simulation paths.

Return a table with:

```text
period_start
period_end
actual_purchases
forecast_mean_purchases
forecast_p05_purchases
forecast_p50_purchases
forecast_p95_purchases
actual_cum_purchases
forecast_mean_cum_purchases
forecast_p05_cum_purchases
forecast_p50_cum_purchases
forecast_p95_cum_purchases
ape_cumulative
```

### 8.3 Actual holdout aggregation

Actual holdout purchases should be aggregated from the same known-machine population used for simulation.

Implement:

```python
aggregate_actual_holdout(
    visits_holdout_known: pd.DataFrame,
    T_cal_end: float,
    T_holdout_end: float,
    freq: str = "W",
) -> pd.DataFrame
```

Use the same time bins as simulation.

---

## 9. Evaluation metrics

Implement and report the following.

### 9.1 Log-likelihood

Report:

```text
LL_EV_cal
LL_CM_cal
LL_joint_cal
LL_EV_holdout
LL_CM_holdout
LL_joint_holdout
```

If EV holdout LL is not implemented, report `null` and clearly state why.

### 9.2 Information criteria

For each fitted model:

```text
AIC = -2 * LL + 2 * k
BIC = -2 * LL + k * log(N)
```

Where:

- `k_EV = 4`
- `k_CM = 6`
- `k_joint = 10`
- `N_EV` = number of modeled repeat-visit gaps plus number of customer censoring terms
- `N_CM` = number of visit rows used in CM
- `N_joint = N_EV + N_CM`

Optionally report CAIC:

```text
CAIC = -2 * LL + k * (log(N) + 1)
```

### 9.3 Forecast error

Primary MAPE over cumulative purchases:

```text
APE_t = abs(forecast_cum_t - actual_cum_t) / max(actual_cum_t, eps)
MAPE = mean_t APE_t
```

Use:

```text
eps = 1.0
```

Also report:

```text
final_cumulative_error_pct =
    (forecast_cum_final - actual_cum_final) / max(actual_cum_final, eps)

MAE_cumulative =
    mean_t abs(forecast_cum_t - actual_cum_t)

RMSE_cumulative =
    sqrt(mean_t (forecast_cum_t - actual_cum_t)^2)

MAPE_incremental =
    mean_t abs(forecast_increment_t - actual_increment_t) / max(actual_increment_t, eps)
```

If early holdout periods have zero actual cumulative purchases, either use `eps=1.0` or skip those periods. Record the choice in `metrics.json`.

### 9.4 Conversion diagnostics

Conditional on actual holdout visits:

```text
actual_holdout_conversion_rate
predicted_holdout_conversion_rate
conversion_relative_error_pct
CM_holdout_loglik_per_visit
```

### 9.5 Visit diagnostics

For EV simulation:

```text
actual_holdout_visits
forecast_holdout_visits_mean
final_visit_error_pct
EV_holdout_loglik_per_gap
```

---

## 10. CLI entry point

Implement:

```bash
python -m src.evcm_pipeline \
  --input data/amazon_clickstream.csv \
  --output reports/evcm \
  --cutoff 1998-06-30 \
  --visit-unit daily \
  --freq W \
  --n-sims 1000 \
  --ev-starts 20 \
  --cm-starts 30 \
  --seed 123
```

Arguments:

```text
--input              path to raw csv/parquet
--output             output directory
--cutoff             optional calibration cutoff date
--calib-frac         default 0.5 if cutoff absent
--visit-unit         daily or session
--freq               W, D, or M for forecast aggregation
--n-sims             number of simulation paths
--ev-starts          random starts for EV optimization
--cm-starts          random starts for CM optimization
--seed               random seed
--max-machines       optional debug subsample, default None
--no-plots           skip plots
```

---

## 11. Required report contents

`reports/evcm/metrics.json` should contain at least:

```json
{
  "data": {
    "n_raw_rows": 50722,
    "n_session_visits": null,
    "n_daily_visits": null,
    "n_machines_total": null,
    "n_machines_calibration": null,
    "n_machines_holdout_known": null,
    "n_machines_holdout_new": null,
    "calibration_start": null,
    "calibration_end": null,
    "holdout_end": null
  },
  "fit": {
    "LL_EV_cal": null,
    "LL_CM_cal": null,
    "LL_joint_cal": null,
    "LL_EV_holdout": null,
    "LL_CM_holdout": null,
    "LL_joint_holdout": null,
    "AIC_joint": null,
    "BIC_joint": null,
    "CAIC_joint": null
  },
  "forecast": {
    "MAPE_cumulative": null,
    "MAE_cumulative": null,
    "RMSE_cumulative": null,
    "final_cumulative_error_pct": null,
    "actual_holdout_cum_purchases_final": null,
    "forecast_holdout_cum_purchases_final": null,
    "actual_holdout_visits": null,
    "forecast_holdout_visits_mean": null
  },
  "conversion": {
    "actual_holdout_conversion_rate": null,
    "predicted_holdout_conversion_rate": null,
    "conversion_relative_error_pct": null
  },
  "optimization": {
    "ev_converged": null,
    "cm_converged": null,
    "ev_message": null,
    "cm_message": null
  }
}
```

Parameter CSVs:

`params_ev.csv`

```text
param,value
r,...
alpha,...
s,...
beta,...
```

`params_cm.csv`

```text
param,value
r_v,...
mu0,...
k,...
r_tau,...
psi,...
pi,...
```

`forecast_by_period.csv`

```text
period_start,period_end,actual_purchases,forecast_mean_purchases,forecast_p05_purchases,forecast_p50_purchases,forecast_p95_purchases,actual_cum_purchases,forecast_mean_cum_purchases,forecast_p05_cum_purchases,forecast_p50_cum_purchases,forecast_p95_cum_purchases,ape_cumulative
```

---

## 12. Baselines

Implement these if time permits; otherwise leave clear TODOs.

### 12.1 Historical conversion + EV visits

Use EV simulation for future visits. At each simulated visit, purchase probability is the customer's historical calibration conversion rate with beta smoothing:

```text
p_i = (purchases_i + a0) / (visits_i + a0 + b0)
```

Default:

```text
a0 = 1
b0 = 10
```

### 12.2 EG/BB

Stationary exponential-gamma visit model plus beta-binomial conversion.

### 12.3 Logistic RFM benchmark

Conditional on actual visits, fit logistic regression with:

```text
number of past visits
number of past purchases
number of visits since last purchase
days since last visit
days since last purchase
```

This is only a conversion benchmark and does not forecast visits unless combined with EV/EG.

---

## 13. Unit tests

Write tests for the following.

### Data prep

- Multiple raw rows for the same `site_session_id` collapse into one session.
- Multiple sessions on the same day collapse into one daily visit.
- `purchase` is 1 if any raw/session row has a purchase flag.
- Time split does not leak holdout rows into calibration.

### EV model

- EV log-likelihood is finite for small synthetic histories.
- EV update after an arrival matches the formulas in Section 5.2.
- EV censoring adds `alpha += censor_gap` to posterior state but does not change shape.
- Simulation returns no visits beyond `T_holdout_end`.

### CM model

- CM purchase probability is in `[0, 1]`.
- For customers with no prior purchases, hard-core never-buyer mixture reduces purchase probability by factor `pi`.
- After first purchase, `pi` no longer enters the purchase probability.
- `k = 1` reduces the evolving visit-effect sum to a linear accumulation.

### Forecasting

- Forecast aggregation bins match actual aggregation bins.
- Cumulative purchases are nondecreasing.
- MAPE calculation handles zero actual counts through `eps`.

---

## 14. Numerical and modeling cautions

1. Use the Gamma rate parameterization internally. SciPy random Gamma uses scale, so always pass `scale=1 / rate`.
2. Fit on calibration only.
3. Evaluate primary forecast only on machines observed in calibration.
4. Keep new holdout-only machines separate.
5. The model predicts purchase incidence, not revenue.
6. Use daily visit aggregation by default.
7. Use multiple optimization starts.
8. Clamp probabilities before taking logs.
9. Record all random seeds.
10. Save all intermediate cleaned datasets or summaries so results are auditable.

---

## 15. Definition of done

The task is complete when the following command runs end-to-end:

```bash
python -m src.evcm_pipeline \
  --input data/amazon_clickstream.csv \
  --output reports/evcm \
  --visit-unit daily \
  --freq W \
  --n-sims 1000 \
  --seed 123
```

And produces:

```text
reports/evcm/metrics.json
reports/evcm/params_ev.csv
reports/evcm/params_cm.csv
reports/evcm/forecast_by_period.csv
reports/evcm/figures/cumulative_purchases_forecast.png
reports/evcm/figures/weekly_purchases_forecast.png
```

The final README should briefly explain:

- how the raw data was converted into visit histories,
- the calibration/holdout split,
- estimated EV and CM parameters,
- in-sample and holdout log-likelihood,
- cumulative purchase MAPE,
- the final cumulative forecast error,
- and any deviations from this specification.
