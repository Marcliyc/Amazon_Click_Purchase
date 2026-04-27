# Codex Implementation Spec: JAX EV/CM Model for Amazon Clickstream Data

This document is a task brief for Codex. Add it to the repository root, preferably as `EV_CM_JAX_CODEX_SPEC.md`. If you want Codex to treat the instructions as persistent repository guidance, copy the implementation rules and deliverables into `AGENTS.md` as well.

The goal is to implement a scalable JAX version of the Moe--Fader EV/CM framework for an Amazon clickstream dataset with roughly 385K+ raw rows and 10K+ machines. The implementation should split the data temporally, fit the evolving visits model, fit the conversion model conditional on visits, report log-likelihood and related fit statistics, and forecast holdout cumulative purchases over time with MAPE and companion diagnostics.

---

## 1. Conceptual model summary

The EV/CM framework has two separable parts:

1. **EV, Evolving Visits:** models when each machine/customer returns to the site. It uses a latent visit rate, heterogeneous across customers and evolving after each observed repeat visit.
2. **CM, Conversion Model:** models whether a visit converts into a purchase. Purchase probability is driven by accumulated visit effects, an evolving purchase threshold, Bayesian updating from past purchases/non-purchases, and a hard-core never-buyer mixture.

Fit EV on visit timing histories. Fit CM on purchase incidence conditional on observed visits. Use both together for holdout purchase forecasting: simulate future visits from EV, then simulate purchase/non-purchase outcomes from CM on those simulated visits.

---

## 2. Data assumptions and raw schema

The repository will include a CSV or Parquet file with at least these columns:

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

Primary fields:

- Customer or panelist id: `machine_id`
- Session id: `site_session_id`
- Visit timestamp: `event_date` + `event_time`
- Purchase incidence: `tran_flg` plus transaction fields as robustness checks
- Optional future covariates: demographics, pages viewed, duration, session-level features, previous-history features

The model predicts **purchase incidence**, not basket value. Revenue fields should be saved for optional descriptive reporting but should not be used as the primary target in the first implementation.

---

## 3. Required repository structure

Codex should create or modify the repo to look like this:

```text
.
├── data/
│   └── amazon_clickstream.csv              # or .parquet; user-provided, usually not committed
├── src/
│   ├── __init__.py
│   ├── config.py                           # dataclass/YAML config loading
│   ├── data_prep.py                        # loading, cleaning, visit construction, splits, arrays
│   ├── jax_utils.py                        # parameter transforms, padding, batching, device utilities
│   ├── ev_model_jax.py                     # EV likelihood, states, simulation
│   ├── cm_model_jax.py                     # CM likelihood, states, simulation
│   ├── fit_jax.py                          # Adam/LBFGS fitting utilities and multistart
│   ├── forecasting_jax.py                  # combined EV/CM holdout simulation
│   ├── metrics.py                          # LL, AIC, BIC, MAPE, sMAPE, MAE, calibration diagnostics
│   └── evcm_jax_pipeline.py                # CLI entry point
├── tests/
│   ├── test_data_prep.py
│   ├── test_ev_model_jax.py
│   ├── test_cm_model_jax.py
│   ├── test_forecasting_jax.py
│   └── test_pipeline_smoke.py
├── reports/
│   └── evcm_jax/
│       ├── metrics.json
│       ├── params_ev.csv
│       ├── params_cm.csv
│       ├── fit_history_ev.csv
│       ├── fit_history_cm.csv
│       ├── forecast_by_period.csv
│       ├── customer_state_calibration.parquet
│       ├── visit_history_summary.csv
│       ├── purchase_history_summary.csv
│       └── figures/
│           ├── cumulative_purchases_forecast.png
│           ├── weekly_purchases_forecast.png
│           ├── cumulative_visits_forecast.png
│           ├── calibration_conversion_fit.png
│           └── ev_posterior_rate_segments.png
├── requirements.txt
└── README.md
```

Keep the code modular. The CLI should run the full pipeline from raw data to reports, but every model function should also be testable independently.

---

## 4. Environment and JAX requirements

Use the provided `requirements.txt` for a portable CPU install. For GPU, install a CUDA-enabled JAX wheel using a setup script or manual command appropriate to the machine:

```bash
# CPU, portable local development
pip install -r requirements.txt

# NVIDIA GPU, choose one depending on driver/CUDA support
pip install -U "jax[cuda13]"
# or
pip install -U "jax[cuda12]"
```

Implementation requirements:

- Use JAX arrays inside likelihood/simulation functions. Do not pass Pandas objects into `jax.jit` functions.
- Use `jax.value_and_grad` for gradients.
- Use `jax.jit` for full-batch likelihood and forecasting kernels.
- Use `jax.vmap` across customers where possible and `jax.lax.scan` across visits or simulated events.
- Enable `jax_enable_x64=True` by default for stable likelihood optimization. Expose a config option to disable x64 if GPU double precision is too slow.
- Log available devices at the start of the run.

Example startup:

```python
import jax
jax.config.update("jax_enable_x64", True)
print("JAX devices:", jax.devices())
```

Recommended memory environment variables for GPU runs:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
```

---

## 5. CLI contract

Implement a command line entry point:

```bash
python -m src.evcm_jax_pipeline \
  --data data/amazon_clickstream.csv \
  --output reports/evcm_jax \
  --aggregation daily \
  --split-date 1998-07-01 \
  --holdout-end 1998-10-31 \
  --period W \
  --n-sim 500 \
  --seed 123 \
  --x64 true
```

Arguments:

- `--data`: CSV or Parquet input path.
- `--output`: report directory.
- `--aggregation`: `daily` or `session`. Default `daily` to match the original EV application and avoid zero intervisit gaps from multiple same-day sessions.
- `--split-date`: first date of holdout. All visits before this date are calibration.
- `--holdout-end`: final date included in holdout forecasts.
- `--period`: period for reporting actual/predicted purchases, e.g. `D`, `W`, `M`.
- `--n-sim`: number of Monte Carlo paths for EV/CM forecasting.
- `--seed`: PRNG seed.
- `--x64`: whether to enable JAX 64-bit mode.
- `--fit-cm-conditional-only`: optional flag to only evaluate/predict conversions conditional on observed holdout visits. Default false; full forecast should simulate visits and conversions.
- `--max-customers`: optional debugging subset.

Add `python -m src.evcm_jax_pipeline --help` support.

---

## 6. Data preprocessing

### 6.1 Load raw data

Create:

```python
def load_raw_data(path: str | Path) -> pd.DataFrame:
    ...
```

Requirements:

1. Accept CSV or Parquet.
2. Combine `event_date` and `event_time` into `event_datetime`.
3. Convert `event_date` to normalized calendar date.
4. Sort by `machine_id`, `event_datetime`, `site_session_id`.
5. Keep the raw index only as metadata; do not model from the row index.
6. Raise an informative error if any timestamp cannot be parsed.

Example:

```python
df["event_datetime"] = pd.to_datetime(
    df["event_date"].astype(str) + " " + df["event_time"].astype(str),
    errors="coerce",
)
if df["event_datetime"].isna().any():
    bad = df.loc[df["event_datetime"].isna(), ["event_date", "event_time"]].head()
    raise ValueError(f"Unparseable timestamps, examples:\n{bad}")
```

### 6.2 Purchase flag construction

Create a robust row-level flag:

```python
df["purchase_row"] = (
    (df["tran_flg"].fillna(0).astype(float) > 0)
    | (df["basket_tot"].fillna(0).astype(float) > 0)
    | (df["prod_totprice"].fillna(0).astype(float) > 0)
    | (df["prod_qty"].fillna(0).astype(float) > 0)
)
```

Also save diagnostics comparing purchase definitions:

- count where `tran_flg > 0`
- count where transaction fields are positive
- count where definitions disagree
- number of purchase sessions and purchase days

Use the robust flag by default, but expose a config option `purchase_rule = tran_flg_only | robust`.

### 6.3 Session aggregation

Create:

```python
def make_session_visits(df: pd.DataFrame) -> pd.DataFrame:
    ...
```

Group by `machine_id, site_session_id`.

Aggregation rules:

- `visit_datetime`: minimum `event_datetime`
- `visit_date`: date of `visit_datetime`
- `purchase`: max/any of `purchase_row`
- `purchase_session_count`: 1 if purchase else 0
- `pages_viewed`: max if duplicated product rows are suspected; otherwise sum. Default `max`.
- `duration`: max if duplicated product rows are suspected; otherwise sum. Default `max`.
- `basket_tot`: max at session level, because basket total can be repeated on product rows
- `prod_totprice`: sum across product rows
- demographic fields: first non-null value

### 6.4 Daily aggregation

Create:

```python
def make_daily_visits(session_df: pd.DataFrame) -> pd.DataFrame:
    ...
```

Group by `machine_id, visit_date`.

Aggregation rules:

- one row per `machine_id, visit_date`
- `visit_datetime`: earliest session timestamp that day
- `purchase`: 1 if any session that day purchased, else 0
- `purchase_session_count`: number of purchasing sessions that day
- `pages_viewed`: sum over sessions that day
- `duration`: sum over sessions that day
- `basket_tot`: sum over session-level basket totals
- `prod_totprice`: sum over session-level product totals

### 6.5 Time origin and split

Set:

```python
global_start_date = visits["visit_date"].min()
visits["t"] = (visits["visit_date"] - global_start_date).dt.days.astype(float)
```

Use a temporal split:

```text
calibration: visit_date < split_date
holdout:     split_date <= visit_date <= holdout_end
```

Do not random-split rows; that leaks future behavior.

Training inclusion rules:

- Include machines with at least one calibration visit in EV/CM fitting.
- Machines first observed in holdout are cold-start customers. Exclude them from the main individual-level holdout forecast unless a separate cold-start model is implemented.
- Save counts for: total machines, calibration machines, holdout machines, cold-start holdout machines.

---

## 7. JAX array representation

Create customer-level histories sorted by time.

For calibration, arrays should include:

```python
customer_ids: np.ndarray[int64]          # shape [N]
times: np.ndarray[float64]               # padded shape [N, T_max], visit times in days
purchases: np.ndarray[int8]              # padded shape [N, T_max], 0/1 purchase on visit
mask: np.ndarray[bool]                   # padded shape [N, T_max]
lengths: np.ndarray[int32]               # shape [N]
cal_end_t: float                         # days from global start to split_date
holdout_end_t: float                     # days from global start to holdout_end
```

Because 385K rows are modest for JAX but sequence lengths can be uneven, implement two modes:

1. **Simple padded mode**: use `[N, T_max]` arrays. This is acceptable if `N * T_max` is not too large.
2. **Bucketed padded mode**: group customers into length buckets, e.g. 1, 2, 3--4, 5--8, 9--16, 17--32, 33--64, 65+. Fit by summing log-likelihood over buckets. Use this if `N * T_max` is wasteful.

Start with simple padded mode and add bucketed mode if tests show memory issues.

Important constraints:

- Avoid Python loops over customers inside the objective.
- A Python loop over a small number of length buckets is acceptable.
- Use `jax.lax.scan` over the visit dimension and vectorize over customers.
- Use masks to ignore padded entries.

---

## 8. Parameter transforms and numerical stability

Never optimize constrained parameters directly. Use unconstrained raw parameters and transforms.

### EV raw parameters

Model parameters:

```text
r     > 0
alpha > 0
s     > 0
beta  > 0
```

Use rate parameterization for the gamma prior: if `lambda ~ Gamma(shape=r, rate=alpha)`, then `E[lambda] = r / alpha`.

Transform:

```python
r     = softplus(raw_r) + eps
alpha = softplus(raw_alpha) + eps
s     = softplus(raw_s) + eps
beta  = softplus(raw_beta) + eps
```

### CM raw parameters

Model parameters:

```text
r_v   > 0      baseline visit-effect shape
mu0   > 0      initial incremental visit-effect shape
k     > 0      evolution multiplier for visit effects
r_tau > 0      purchase-threshold shape
psi   real     threshold evolution coefficient
pi    in (0,1) non-hardcore buyer mixture probability
```

Transform:

```python
r_v   = softplus(raw_r_v) + eps
mu0   = softplus(raw_mu0) + eps
k     = softplus(raw_k) + eps
r_tau = softplus(raw_r_tau) + eps
psi   = raw_psi
pi    = sigmoid(raw_pi) * (1 - 2 * eps) + eps
```

Do not hard-clip parameters inside the likelihood except for tiny log-safety epsilons. Hard clipping can create flat gradients and boundary solutions. Use weak penalties if needed.

Recommended eps: `1e-12` for float64, `1e-7` for float32.

---

## 9. EV model details

### 9.1 Notation

For customer `i`, let observed calibration visit times be:

```text
t_i0 < t_i1 < ... < t_i,J-1
```

The first observed visit is treated as given/conditioned on. EV likelihood is for repeat intervisit gaps and right-censoring after the last calibration visit.

Intervisit gaps:

```text
d_ij = t_ij - t_i,j-1, for j >= 1
```

At any point, the individual visit rate is integrated out under a gamma prior/posterior state:

```text
lambda_i ~ Gamma(shape = r_state, rate = alpha_state)
```

The marginal gap distribution is Lomax/Pareto-II:

```text
f(d | r, alpha) = r * alpha^r / (alpha + d)^(r + 1)
S(c | r, alpha) = (alpha / (alpha + c))^r
```

### 9.2 EV likelihood recurrence

For each customer:

1. Initialize state at first observed visit:

```text
r_state = r
alpha_state = alpha
```

2. For each repeat gap `d > 0`, add:

```text
log f(d | r_state, alpha_state)
= log(r_state) + r_state * log(alpha_state) - (r_state + 1) * log(alpha_state + d)
```

3. Update after a repeat visit:

```text
r_post = r_state + 1
alpha_post = alpha_state + d
r_state_next = (r_post * s) / (r_post + s + 1)
alpha_state_next = (alpha_post * beta) / (r_post + s + 1)
```

4. Add right-censoring from last observed calibration visit to `cal_end_t`:

```text
c = max(cal_end_t - t_last, 0)
log S(c | r_state, alpha_state)
= r_state * (log(alpha_state) - log(alpha_state + c))
```

Customers with only one calibration visit contribute only the censoring term.

### 9.3 EV JAX functions

Implement:

```python
@dataclass(frozen=True)
class EVParams:
    r: jnp.ndarray
    alpha: jnp.ndarray
    s: jnp.ndarray
    beta: jnp.ndarray


def unpack_ev(raw: jnp.ndarray, eps: float) -> EVParams:
    ...


def ev_loglik_customer(params: EVParams, times_i, mask_i, cal_end_t) -> jnp.ndarray:
    ...


def ev_loglik(raw_params, times, mask, cal_end_t) -> jnp.ndarray:
    # returns scalar total log-likelihood
    ...


def ev_negative_loglik(raw_params, times, mask, cal_end_t) -> jnp.ndarray:
    return -ev_loglik(...)
```

Use `vmap(ev_loglik_customer)` and `sum` across customers.

### 9.4 EV posterior state at calibration end

Implement:

```python
def ev_calibration_state(params: EVParams, times, mask, cal_end_t) -> dict[str, jnp.ndarray]:
    ...
```

Return per-customer state after processing calibration:

```text
r_state[N]
alpha_state[N]
last_visit_t[N]
```

This state is needed for holdout simulation.

### 9.5 EV simulation

To sample next gap from the marginal predictive survival distribution:

```text
U ~ Uniform(0, 1)
D = alpha_state * ((1 - U)^(-1 / r_state) - 1)
```

Then if `last_visit_t + D <= holdout_end_t`, record a simulated visit, update EV state using the recurrence, and continue. If not, stop.

Implement simulation with `jax.lax.scan` up to a configured `max_sim_visits_per_customer`. This max can be a safe upper bound such as the 99.9th percentile observed calibration visits per customer times a multiplier. Also return a truncation warning if any path hits the max.

---

## 10. CM model details

### 10.1 Notation

For customer `i` at visit `j` using 1-indexing:

```text
n_ij = j - 1                         # prior visits before current visit
x_ij = number of prior purchases      # purchases before current visit
lp_ij = visit index of last purchase  # 0 if no previous purchase
```

The net visit-effect shape is:

```text
V_ij = r_v + mu0 * sum_{u=lp_ij + 1}^{j} k^u
```

Use a stable geometric-sum helper:

```python
def geom_sum_k(k, start_u, end_u):
    # sum_{u=start_u}^{end_u} k^u, inclusive
    # if abs(k - 1) is small, return end_u - start_u + 1
```

The purchase-threshold shape is:

```text
tau_ij = r_tau * exp(psi * x_ij)
```

For a non-hardcore customer, after Bayesian updating from past conversion outcomes, the purchase probability at the current visit is:

```text
p_star_ij = (V_ij + x_ij) / (V_ij + tau_ij + n_ij)
```

This is a dynamic beta-binomial predictive probability. `V_ij` and `tau_ij` change with the customer's visit and purchase history.

### 10.2 Hard-core never-buyer mixture

`pi` is the initial probability that a customer belongs to the non-hardcore group. A hard-core never-buyer has purchase probability exactly zero forever.

Maintain a per-customer state:

```text
q_ij = posterior probability customer is non-hardcore before visit j
```

Initialize `q = pi`.

At each visit:

- If the customer has purchased before (`x_ij > 0`), set `q = 1` and use `Pr(y=1) = p_star_ij`.
- If the customer has not purchased before (`x_ij = 0`), use:

```text
Pr(y=1) = q_ij * p_star_ij
Pr(y=0) = 1 - q_ij * p_star_ij
```

Bayes update after observing the current visit:

```text
if y = 1: q_next = 1
if y = 0 and x_ij = 0:
    q_next = q_ij * (1 - p_star_ij) / (1 - q_ij * p_star_ij)
if x_ij > 0 and y = 0:
    q_next = 1
```

Also update history states:

```text
n_next = n_ij + 1
x_next = x_ij + y
lp_next = j if y == 1 else lp_ij
```

Clamp probabilities only for logs:

```python
p_log = jnp.clip(p, eps, 1 - eps)
ll = y * log(p_log) + (1 - y) * log1p(-p_log)
```

### 10.3 CM JAX functions

Implement:

```python
@dataclass(frozen=True)
class CMParams:
    r_v: jnp.ndarray
    mu0: jnp.ndarray
    k: jnp.ndarray
    r_tau: jnp.ndarray
    psi: jnp.ndarray
    pi: jnp.ndarray


def unpack_cm(raw: jnp.ndarray, eps: float) -> CMParams:
    ...


def cm_loglik_customer(params: CMParams, purchases_i, mask_i) -> jnp.ndarray:
    ...


def cm_loglik(raw_params, purchases, mask) -> jnp.ndarray:
    ...


def cm_negative_loglik(raw_params, purchases, mask) -> jnp.ndarray:
    return -cm_loglik(...)
```

Use `jax.lax.scan` over visit positions and `vmap` across customers.

### 10.4 CM calibration state for forecasting

After processing calibration visits, return:

```text
n_prior[N]
x_prior[N]
last_purchase_index[N]
q_nonhardcore[N]
next_visit_index[N] = n_prior + 1
```

This state must initialize conversion simulation in holdout.

---

## 11. Optimization and fitting

### 11.1 Fit EV and CM separately first

Fit EV and CM separately in v1:

```text
EV: maximize log p(observed calibration visit timings)
CM: maximize log p(observed calibration purchase indicators | observed calibration visits)
```

Joint estimation can be added later, but separate estimation is easier to debug and follows the natural decomposition.

### 11.2 Optimizers

Implement two-stage fitting:

1. Adam warm start using Optax:
   - 500--3000 iterations
   - learning rate default `1e-2`, reduce on plateau
   - optional mini-batching for CM only during warm start
2. Full-batch L-BFGS using `jaxopt.LBFGS`:
   - run until gradient norm and objective change converge
   - max iterations default 500

Use multiple starts because boundary-like parameter values can occur:

```text
EV starts: 10
CM starts: 20
```

Save all starts to `fit_history_ev.csv` and `fit_history_cm.csv`, including:

```text
start_id, optimizer, converged, n_iter, final_neg_ll, grad_norm, raw_params, transformed_params
```

### 11.3 Initialization suggestions

EV initialization from observed calibration gaps:

```python
mean_gap = np.mean(gaps)
rate_hat = 1 / max(mean_gap, 1e-6)
r_init = 1.0
alpha_init = r_init / rate_hat
s_init = 10.0
beta_init = alpha_init
```

CM initialization from purchase rate:

```python
pbar = purchases.sum() / mask.sum()
r_v_init = max(pbar, 1e-4)
r_tau_init = max(1 - pbar, 1e-4)
mu0_init = 1e-3
k_init = 1.0
psi_init = 0.0
pi_init = min(max(number_of_buyers / number_of_visitors, 1e-3), 0.999)
```

Random starts should jitter on the raw scale.

### 11.4 Fit statistics

Report for EV:

```text
LL_EV
N_customers_EV
N_repeat_gaps
N_ev_params = 4
AIC_EV = 2k - 2LL
BIC_EV = k log(N_repeat_gaps or N_customers) - 2LL
```

Report for CM:

```text
LL_CM
N_visit_observations
N_purchase_observations
N_buyers
N_cm_params = 6
AIC_CM
BIC_CM
calibration_purchase_rate
mean_predicted_purchase_probability
```

For BIC, use the number of modeled observations:

- EV: number of repeat gaps plus number of censoring terms. Also save a variant using number of customers.
- CM: number of calibration visit observations.

If Hessian-based standard errors are implemented, save them, but do not block the main pipeline on standard errors.

---

## 12. Forecasting holdout purchases

### 12.1 Actual holdout aggregation

From observed holdout visits, aggregate by reporting period:

```text
period_start
actual_visits
actual_purchases
actual_cumulative_visits
actual_cumulative_purchases
```

Use only machines with calibration histories for the main forecast unless reporting cold-start separately.

### 12.2 Full EV/CM Monte Carlo forecast

For each simulation path and each customer:

1. Start from the EV calibration state: `r_state`, `alpha_state`, `last_visit_t`.
2. Start from the CM calibration state: `n_prior`, `x_prior`, `last_purchase_index`, `q_nonhardcore`, `next_visit_index`.
3. Simulate EV visits until `holdout_end_t`.
4. For each simulated visit, compute CM conversion probability and sample purchase.
5. Update both EV and CM states after each simulated visit.
6. Aggregate simulated visits and purchases by reporting period.

Output mean forecast and uncertainty intervals across simulation paths:

```text
period_start
actual_visits
pred_visits_mean
pred_visits_p05
pred_visits_p50
pred_visits_p95
actual_purchases
pred_purchases_mean
pred_purchases_p05
pred_purchases_p50
pred_purchases_p95
actual_cumulative_visits
pred_cumulative_visits_mean
pred_cumulative_visits_p05
pred_cumulative_visits_p50
pred_cumulative_visits_p95
actual_cumulative_purchases
pred_cumulative_purchases_mean
pred_cumulative_purchases_p05
pred_cumulative_purchases_p50
pred_cumulative_purchases_p95
```

### 12.3 MAPE and companion metrics

Primary requested metric:

```text
MAPE_cumulative_purchases = mean(abs(actual_cum - pred_cum) / max(actual_cum, epsilon))
```

Set `epsilon = 1.0` by default because early cumulative purchase counts may be zero or very small. Also report:

```text
sMAPE_cumulative_purchases
MAE_cumulative_purchases
RMSE_cumulative_purchases
final_cumulative_purchase_error
final_cumulative_purchase_pct_error
MAPE_period_purchases
coverage_90pct_cumulative_purchases
```

Do not report only MAPE; cumulative series can make MAPE look artificially good late in the holdout.

### 12.4 Conditional conversion diagnostic

Also implement an optional diagnostic using actual observed holdout visits:

```text
Given observed holdout visits, forecast purchases using only CM state updates.
```

This separates conversion-model errors from visit-forecasting errors. Save this as:

```text
forecast_by_period_cm_conditional.csv
```

---

## 13. Future covariates and ML extension hooks

The first implementation should fit the covariate-free EV/CM model. However, design the code so that covariates can be added later.

Add feature preparation utilities:

```python
def make_static_customer_features(visits: pd.DataFrame) -> pd.DataFrame:
    ...

def make_dynamic_visit_features(visits: pd.DataFrame) -> pd.DataFrame:
    ...
```

Suggested static features:

- census region
- household size
- household income
- racial background
- country of origin
- first-30-day visits
- first-30-day pages viewed
- first-30-day duration

Suggested dynamic features:

- pages viewed on current visit
- duration on current visit
- time since previous visit
- cumulative visits
- cumulative purchases
- visits since last purchase

Future parameterization options:

```text
log r_v_i      = base_r_v + f_rv(z_i)
log mu0_i      = base_mu0 + f_mu0(z_i)
log r_tau_i    = base_tau + f_tau(z_i)
logit pi_i     = base_pi + f_pi(z_i)
log EV rate_i  = EV state + f_ev(z_i)
```

For v1, do not fit these functions. But write function signatures so a later Codex task can add:

- scikit-learn gradient boosting or logistic regression covariate baselines
- Flax MLP covariate heads
- two-stage residualization or embedding models

Add simple ML baselines for comparison if time permits:

1. Logistic regression on hand-crafted visit-history features.
2. Gradient boosting classifier on visit-history features.
3. Poisson or negative-binomial model for period-level purchase counts.

These baselines should not replace EV/CM; they are only forecast benchmarks.

---

## 14. Testing requirements

Codex must add tests. The implementation is not done unless tests pass.

### 14.1 Data prep tests

Use a tiny synthetic dataframe with duplicated product rows and multiple sessions on the same day. Verify:

- timestamps parse correctly
- session deduplication is correct
- daily aggregation is correct
- purchase flags are correct
- split produces no leakage

### 14.2 EV tests

Create a toy customer with visits at `[0, 2, 5]` and `cal_end_t = 10`. Verify:

- log-likelihood is finite
- gradients are finite
- one-customer JAX result matches a straightforward NumPy implementation within tolerance
- customer with one visit contributes only survival/censoring
- simulation returns nonnegative visit times and no times beyond holdout end

### 14.3 CM tests

Create tiny histories:

```text
[0, 0, 1, 0]
[0, 0, 0]
[1, 0, 1]
```

Verify:

- log-likelihood is finite
- gradients are finite
- probabilities are in `(0, 1)` after clipping
- after first purchase, `q_nonhardcore` is 1
- for a never-purchasing customer, `q_nonhardcore` decreases or stays the same after nonpurchase visits
- JAX likelihood matches a NumPy reference implementation

### 14.4 Pipeline smoke test

Create a synthetic dataset with 20 machines and known timestamps. Run:

```bash
python -m src.evcm_jax_pipeline --data <synthetic> --output <tmp> --n-sim 10 --max-customers 20
```

Verify output files exist and metrics JSON contains EV LL, CM LL, and cumulative purchase MAPE.

---

## 15. Reporting outputs

### 15.1 `metrics.json`

Required keys:

```json
{
  "data": {
    "n_raw_rows": 0,
    "n_sessions": 0,
    "n_visits": 0,
    "n_machines_total": 0,
    "n_machines_calibration": 0,
    "n_machines_holdout": 0,
    "n_cold_start_holdout_machines": 0,
    "split_date": "YYYY-MM-DD",
    "holdout_end": "YYYY-MM-DD",
    "aggregation": "daily"
  },
  "ev": {
    "log_likelihood": 0.0,
    "aic": 0.0,
    "bic_observation": 0.0,
    "bic_customer": 0.0,
    "params": {}
  },
  "cm": {
    "log_likelihood": 0.0,
    "aic": 0.0,
    "bic": 0.0,
    "params": {},
    "calibration_purchase_rate": 0.0,
    "mean_predicted_purchase_probability": 0.0
  },
  "forecast": {
    "mape_cumulative_purchases": 0.0,
    "smape_cumulative_purchases": 0.0,
    "mae_cumulative_purchases": 0.0,
    "rmse_cumulative_purchases": 0.0,
    "final_cumulative_purchase_error": 0.0,
    "final_cumulative_purchase_pct_error": 0.0,
    "coverage_90pct_cumulative_purchases": 0.0,
    "n_sim": 0
  }
}
```

### 15.2 Parameter CSV files

`params_ev.csv`:

```text
param,value
r,...
alpha,...
s,...
beta,...
```

`params_cm.csv`:

```text
param,value
r_v,...
mu0,...
k,...
r_tau,...
psi,...
pi,...
```

### 15.3 Figures

Use matplotlib. Save PNG files under `reports/evcm_jax/figures/`.

Required plots:

1. Actual vs predicted cumulative purchases over time with 90% simulation interval.
2. Actual vs predicted period purchases.
3. Actual vs predicted cumulative visits over time.
4. Calibration conversion calibration plot: binned predicted purchase probability vs observed purchase rate.
5. Optional segmentation plot of EV posterior expected visit rates at calibration end.

---

## 16. Practical performance guidance for 385K+ rows

Expected shape after daily aggregation may be much smaller than raw rows. Print diagnostics:

```text
raw rows
unique sessions
unique machine-day visits
calibration visits
holdout visits
max visits per machine
p50/p90/p99 visits per machine
purchase rate
buyer rate
```

Performance checklist:

- Convert Pandas to contiguous NumPy arrays before JAX.
- Keep arrays on device during repeated likelihood calls.
- JIT objective once; avoid changing array shapes between optimizer calls.
- Use padded arrays for first implementation; bucket if padding waste is large.
- Avoid Python loops over customers.
- Use `block_until_ready()` when timing JAX code.
- Use `jax.debug.print` only while debugging; remove from final fit loops.
- For MC simulation, process simulations in chunks if `n_sim * N * max_sim_visits` is too large.

---

## 17. Acceptance criteria

The task is complete only when:

1. `pip install -r requirements.txt` succeeds in a clean environment.
2. `pytest` passes.
3. The CLI can run on a synthetic dataset in under a few minutes.
4. The CLI can run on the real Amazon dataset and produce the report files.
5. `metrics.json` includes EV LL, CM LL, AIC/BIC, and cumulative purchase MAPE.
6. `forecast_by_period.csv` includes actual and predicted period/cumulative visits and purchases.
7. The code logs JAX devices and can use GPU if a CUDA-enabled JAX wheel is installed.
8. The README includes exact commands to run preprocessing, fitting, forecasting, and tests.

---

## 18. Suggested implementation order for Codex

1. Create `requirements.txt` and install dependencies.
2. Implement `data_prep.py` and tests.
3. Implement parameter transforms and small utilities in `jax_utils.py`.
4. Implement CM likelihood first because it is conditional on observed visits and easier to test.
5. Implement EV likelihood and calibration states.
6. Implement fitting utilities with Adam + L-BFGS.
7. Implement full pipeline CLI.
8. Implement holdout actual aggregation and CM-conditional diagnostic.
9. Implement EV/CM Monte Carlo forecasting.
10. Add plots and final reporting.
11. Run tests and a small synthetic smoke run.
12. Run on the full dataset and inspect estimated parameters for boundary issues.

---

## 19. Notes on parameter estimates and debugging

If CM estimates are extremely small for `r_v`, `mu0`, or `r_tau`, or if `k` collapses close to zero, do not assume the implementation is wrong immediately. This can happen with sparse purchase data and hard-core never-buyer mixtures. Debug in this order:

1. Check purchase flag definition and purchase rate.
2. Check whether daily aggregation collapses multiple purchase sessions into one purchase day.
3. Check whether the likelihood improves over a beta-binomial benchmark.
4. Check whether `pi` is close to observed buyer share.
5. Try session-level aggregation as a robustness check.
6. Try fixing `k = 1` and `psi = 0` to compare with static variants.
7. Try CM without hard-core never-buyers by fixing `pi = 1`.
8. Inspect calibration predicted purchase probabilities by visit number and by previous purchase history.

Add benchmark model variants if possible:

```text
CM full:        r_v, mu0, k, r_tau, psi, pi
CM no hardcore: pi fixed at 1
CM static:      k fixed at 1, psi fixed at 0
Beta-binomial:  mu0 fixed at 0, k irrelevant, psi fixed at 0, pi fixed at 1
```

Report likelihood improvements over these baselines.
