# Codex Task: Add EV-Beta-Bernoulli Choice-CM Model for Amazon/eBay Competition in JAX

## Goal

Add a new model and a separate fitting pipeline for a two-platform Amazon/eBay competition extension of the existing EV/CM project.

The new behavioral sequence is:

1. A customer follows the existing **EV visit-intention process**.
2. Conditional on having a visit intention, the customer chooses a platform through a **Beta-Bernoulli choice model**:
   - Amazon vs eBay.
   - Only the chosen platform appears in the observed platform-specific session data.
3. Conditional on visiting the chosen platform, the customer follows a **CM purchase process**.

The existing Amazon CM parameters should remain fixed at the previously estimated values. The eBay CM process should share the same main CM parameters as Amazon, but allow eBay-specific values for `pi`, `mu0`, and `k`. These eBay-specific parameters should initialize from the Amazon estimates.

This should be implemented in **JAX**, with a pipeline structure similar to the current project.

---

## Existing Amazon estimates to use as defaults

Use these as initial/default values unless the config overrides them.

### Amazon CM parameters, fixed

```python
amazon_cm_fixed = {
    "r": 0.457522,
    "alpha": 7.372648,
    "s": 16.440086,
    "beta": 16.369219,
}
```

### Amazon EV / auxiliary parameters

```python
amazon_ev_init = {
    "r_v": 1.939664,
    "mu0": 0.751528,
    "k": 0.730525,
    "r_tau": 22.712252,
    "psi": -0.080496,
    "pi": 0.208507,
}
```

For the eBay branch, initialize:

```python
ebay_init = {
    "pi": amazon_ev_init["pi"],
    "mu0": amazon_ev_init["mu0"],
    "k": amazon_ev_init["k"],
}
```

The initial values should be the same, but the eBay values should be trainable unless the config freezes them.

---

## Data assumptions

Input files should have the same session-level format as the Amazon data, including at least:

```text
machine_id
site_session_id
user_session_id
event_date
event_time
tran_flg
basket_tot
domain_name
```

The eBay file is in the same date range as Amazon, for example Jan-Aug 2024.

The fitting target for eBay is **monthly total purchase counts only**.

Important: do not count product-line rows as separate purchases if the raw data has multiple rows per transaction/session. Aggregate to session level first.

Recommended aggregation:

```python
session_purchase = max(tran_flg filled with 0) per unique session
monthly_purchases = sum(session_purchase) by calendar month
```

Use one of these as the unique session key, in order of preference:

1. `user_session_id`
2. `site_session_id`
3. `(machine_id, event_date, event_time)` only as a fallback

---

## New model

Add a new model file, for example:

```text
src/models/ev_beta_choice_cm.py
```

The model should expose a small, composable API:

```python
class EVBetaChoiceCMParams(NamedTuple):
    # Fixed Amazon CM params
    r: float
    alpha: float
    s: float
    beta: float

    # Shared / Amazon EV params
    r_v: float
    r_tau: float
    psi: float

    # Amazon-specific values
    pi_amazon: float
    mu0_amazon: float
    k_amazon: float

    # eBay-specific trainable values
    pi_ebay: float
    mu0_ebay: float
    k_ebay: float

    # Beta-Bernoulli choice parameters
    choice_a: float
    choice_b: float

    # Observation noise for aggregate monthly counts
    obs_scale: float
```

If the existing project already has a parameter-container convention, follow that convention instead of introducing a new style.

---

## Behavioral interpretation

For customer `i` in month `m`:

### 1. EV visit intention

Use the existing EV logic to produce an expected number of visit intentions:

```text
E[V_im] = EV(customer_i, month_m; r_v, mu0, k, r_tau, psi)
```

For Amazon use:

```text
mu0 = mu0_amazon
k = k_amazon
pi = pi_amazon
```

For eBay use:

```text
mu0 = mu0_ebay
k = k_ebay
pi = pi_ebay
```

If the existing EV implementation does not expose a customer-level expectation, add a helper that computes expected visits over a time interval.

### 2. Beta-Bernoulli platform choice

Each customer has a latent eBay preference:

```text
theta_i ~ Beta(choice_a, choice_b)
```

For each visit intention:

```text
C_ij ~ Bernoulli(theta_i)
```

where:

```text
C_ij = 1 means eBay
C_ij = 0 means Amazon
```

The population-average eBay choice probability is:

```text
E[theta_i] = choice_a / (choice_a + choice_b)
```

Because the target data only contains monthly eBay purchase totals, `choice_a` and `choice_b` will be weakly identified unless we add prior information or regularization. Therefore, the model should support either:

1. fixing the concentration `choice_a + choice_b`; or
2. fitting both with an explicit regularization/prior penalty.

Recommended parameterization:

```python
choice_mean = sigmoid(raw_choice_mean)
choice_concentration = softplus(raw_choice_concentration) + min_concentration

choice_a = choice_mean * choice_concentration
choice_b = (1.0 - choice_mean) * choice_concentration
```

Default:

```python
min_concentration = 2.0
initial_choice_mean = 0.5
initial_choice_concentration = 20.0
```

### 3. CM purchase process

Use the existing CM logic for purchase conversion after a platform visit.

Amazon CM parameters must be fixed:

```text
r, alpha, s, beta
```

eBay should share these same CM parameters by default.

The only channel-specific eBay differences should be:

```text
pi_ebay
mu0_ebay
k_ebay
```

If the current implementation treats `pi`, `mu0`, and `k` as EV-side parameters rather than CM-side parameters, keep that convention. The important modeling requirement is that eBay gets its own trainable `pi`, `mu0`, and `k`, initialized from Amazon's estimated values.

---

## Aggregate monthly eBay likelihood

The eBay fitting target is monthly purchase totals:

```text
y_m = observed eBay purchases in month m
```

The model should produce:

```text
lambda_m = expected eBay purchases in month m
```

Then fit using one of these likelihoods.

### Preferred: Negative Binomial aggregate count likelihood

Use this if feasible:

```text
y_m ~ NegativeBinomial(mean=lambda_m, dispersion=obs_scale)
```

This is preferred because monthly counts are likely overdispersed.

Implement a numerically stable JAX version:

```python
def negbinom_logpmf_from_mean_dispersion(y, mean, dispersion):
    # dispersion > 0
    # Var[y] = mean + mean^2 / dispersion
```

### Simpler fallback: Poisson likelihood

Use this if the project currently lacks count-distribution utilities:

```text
y_m ~ Poisson(lambda_m)
```

Implement:

```python
loglik = y * log(lambda_m + eps) - lambda_m - gammaln(y + 1)
```

Use `jax.scipy.special.gammaln`.

### Optional Gaussian approximation

For quick debugging only:

```text
y_m ~ Normal(lambda_m, sigma_m)
```

Do not make this the default unless the count likelihood causes numerical issues.

---

## Expected eBay purchase count

A reasonable first implementation can compute monthly expected purchases as:

```text
lambda_m =
    total_expected_visit_intentions_m
    * E[theta]
    * expected_purchase_probability_ebay_m
```

where:

```text
E[theta] = choice_a / (choice_a + choice_b)
```

and `expected_purchase_probability_ebay_m` comes from the CM process using the shared CM parameters plus eBay-specific `pi_ebay`.

If the existing code has a simulation-based forecasting function, implement both:

1. deterministic expectation mode for fast optimization;
2. simulation mode for posterior/predictive diagnostics.

For deterministic fitting, start with the simplest stable approximation that matches the existing EV/CM code conventions.

---

## Parameter constraints

All positive parameters should be optimized in unconstrained raw space and transformed:

```python
positive = softplus(raw) + eps
probability = sigmoid(raw)
real = raw
```

Recommended constraints:

```text
r, alpha, s, beta: fixed positive
r_v, r_tau: positive
mu0_amazon, mu0_ebay: positive
k_amazon, k_ebay: positive
pi_amazon, pi_ebay: probability in (0, 1)
choice_mean: probability in (0, 1)
choice_concentration: positive
obs_scale: positive
psi: real
```

Since Amazon CM is fixed, do not optimize `r`, `alpha`, `s`, or `beta` in the new eBay aggregate fitting pipeline unless the config explicitly requests joint fitting.

---

## Loss function

The default loss should be:

```text
loss = - monthly_count_log_likelihood
       + regularization
       + optional prior_penalty
```

Recommended regularization/prior penalties:

```text
pi_ebay close to pi_amazon initially, but not forced
mu0_ebay close to mu0_amazon initially, but not forced
k_ebay close to k_amazon initially, but not forced
choice_mean weakly regularized around 0.5 unless user supplies prior
choice_concentration fixed or weakly regularized
```

Example:

```python
prior_penalty =
    lambda_pi * (logit(pi_ebay) - logit(pi_amazon)) ** 2
  + lambda_mu * (log(mu0_ebay) - log(mu0_amazon)) ** 2
  + lambda_k  * (log(k_ebay) - log(k_amazon)) ** 2
```

The prior weights should be configurable.

Default prior weights should be mild, e.g.:

```python
lambda_pi = 0.1
lambda_mu = 0.1
lambda_k = 0.1
```

---

## New fitting pipeline

Add a separate pipeline, for example:

```text
src/pipelines/fit_ebay_choice_cm_jax.py
```

or, if the project uses module entry points:

```text
src/evcm_ebay_choice_pipeline.py
```

The pipeline should not replace the existing Amazon pipeline.

The pipeline should do the following:

1. Load Amazon session data if needed for EV baseline / customer universe.
2. Load eBay session data.
3. Filter both to the configured date range.
4. Split calibration and holdout periods:
   - default calibration: Jan-June
   - default holdout: July-Aug
5. Aggregate eBay purchases by month.
6. Initialize parameters from Amazon estimates.
7. Freeze Amazon CM parameters.
8. Fit eBay-specific `pi_ebay`, `mu0_ebay`, `k_ebay`, and choice parameters.
9. Save fitted parameters.
10. Save monthly fitted-vs-actual tables.
11. Save holdout forecasts.
12. Save diagnostic plots.

---

## Suggested config

Add a config file:

```text
configs/ebay_choice_cm.py
```

Example config fields:

```python
data = {
    "amazon_path": "...",
    "ebay_path": "...",
    "date_col": "event_date",
    "session_id_col": "user_session_id",
    "purchase_col": "tran_flg",
    "domain_col": "domain_name",
}

date_range = {
    "start": "2024-01-01",
    "calibration_end": "2024-06-30",
    "holdout_end": "2024-08-31",
    "freq": "M",
}

amazon_fixed = {
    "r": 0.457522,
    "alpha": 7.372648,
    "s": 16.440086,
    "beta": 16.369219,
    "r_v": 1.939664,
    "r_tau": 22.712252,
    "psi": -0.080496,
    "pi": 0.208507,
    "mu0": 0.751528,
    "k": 0.730525,
}

ebay_init = {
    "pi": 0.208507,
    "mu0": 0.751528,
    "k": 0.730525,
}

choice = {
    "initial_mean": 0.5,
    "initial_concentration": 20.0,
    "fix_concentration": True,
    "min_concentration": 2.0,
}

fit = {
    "seed": 123,
    "optimizer": "adam",
    "learning_rate": 1e-2,
    "num_steps": 5000,
    "print_every": 250,
    "likelihood": "negative_binomial",
    "use_jit": True,
}

priors = {
    "lambda_pi": 0.1,
    "lambda_mu": 0.1,
    "lambda_k": 0.1,
}

outputs = {
    "workdir": "runs/ebay_choice_cm",
    "save_plots": True,
    "save_forecasts": True,
}
```

---

## CLI

The new pipeline should be runnable in a way similar to the current project.

Example:

```bash
python -m src.pipelines.fit_ebay_choice_cm_jax \
  --config configs/ebay_choice_cm.py \
  --workdir runs/ebay_choice_cm_2024
```

If the existing project uses `absl`, `ml_collections`, or another pattern, follow the existing pattern.

---

## Required outputs

Save these files to `workdir`.

### Parameter files

```text
params_initial.csv
params_fitted.csv
params_fitted.json
```

Include both constrained and unconstrained/raw parameter values if possible.

### Monthly fit table

```text
monthly_fit.csv
```

Columns:

```text
month
split
actual_ebay_purchases
pred_mean_ebay_purchases
pred_p05_ebay_purchases
pred_p50_ebay_purchases
pred_p95_ebay_purchases
ape
```

For deterministic mode, p05/p50/p95 can be omitted or set to null. For simulation mode, fill them.

### Diagnostics

```text
loss_curve.csv
monthly_forecast_plot.png
cumulative_forecast_plot.png
parameter_trace_or_history.csv
```

### Console summary

At the end of the run, print:

```text
Fitted eBay pi
Fitted eBay mu0
Fitted eBay k
Fitted choice mean
Fitted choice concentration
Calibration MAPE
Holdout MAPE
Final cumulative error
```

---

## Diagnostics to plot

Add a plotting helper, for example:

```text
src/plots/plot_ebay_choice_cm.py
```

Required plots:

1. Monthly eBay actual vs predicted purchases.
2. Cumulative eBay actual vs predicted purchases.
3. Loss curve.
4. Bar chart comparing Amazon vs eBay:
   - `pi`
   - `mu0`
   - `k`
5. If simulation mode is implemented:
   - predictive interval plot.
   - distribution of customer-level eBay choice probabilities `theta_i`.

---

## Testing requirements

Add tests under:

```text
tests/test_ev_beta_choice_cm.py
tests/test_ebay_choice_cm_pipeline.py
```

Minimum tests:

### Data aggregation

Given synthetic session rows with multiple rows per session, verify that monthly purchase counts deduplicate sessions correctly.

### Parameter transforms

Verify:

```text
positive params > 0
probability params in (0, 1)
```

### Likelihood sanity check

For fixed observed counts, the loss should be lower when predicted means are close to observed counts.

### Initialization

Verify that:

```python
pi_ebay_init == pi_amazon
mu0_ebay_init == mu0_amazon
k_ebay_init == k_amazon
```

### Freezing Amazon CM

Verify that `r`, `alpha`, `s`, and `beta` do not change during eBay-only fitting.

### JAX compilation

Run a tiny synthetic dataset through:

```python
jax.jit(loss_fn)
jax.grad(loss_fn)
```

and verify no NaNs.

---

## Numerical stability requirements

Use:

```python
eps = 1e-8
```

or a project-standard value for:

```text
log
division
probability clipping
count likelihood mean
```

Clip or transform predicted monthly means:

```python
lambda_m = jnp.maximum(lambda_m, eps)
```

Use `jax.debug.print` only during debugging, not in final default training loop.

---

## Identifiability warning

Because this pipeline only uses monthly eBay purchase totals, several latent mechanisms can explain the same aggregate count:

1. more visit intentions;
2. higher eBay choice probability;
3. higher eBay purchase probability;
4. slower eBay decay / different `k`;
5. observation noise.

Therefore, the code should log this warning at the start of fitting:

```text
Warning: fitting EV-Beta-Bernoulli-Choice-CM from aggregate monthly eBay purchase counts only.
Choice heterogeneity and eBay-specific EV/CM parameters may be weakly identified.
Use priors, fixed concentration, or additional session-level moments if available.
```

The config should allow adding extra moments later, such as:

```text
monthly eBay visits
monthly Amazon visits
monthly Amazon purchases
customer-level overlap
repeat purchase counts
```

Do not block the current implementation on those extensions.

---

## Implementation notes

### Keep the new pipeline separate

Do not break existing Amazon-only scripts. The new code should be additive.

### Prefer deterministic fitting first

Start with deterministic expected monthly counts. Add simulation only after the deterministic pipeline works.

### Use vectorized JAX operations

Use:

```python
jax.vmap
jax.jit
jax.grad
```

where natural.

Avoid Python loops inside the loss over customers or months unless the dataset is tiny.

### Reproducibility

Every run should save:

```text
config used
random seed
git commit hash if available
initial parameters
final parameters
loss history
```

---

## Suggested implementation sequence for Codex

1. Inspect the existing project structure and identify:
   - existing EV model code;
   - existing CM model code;
   - existing JAX parameter transform utilities;
   - existing pipeline/config pattern.
2. Add `EVBetaChoiceCMParams` or adapt the existing parameter container.
3. Add constrained/unconstrained parameter transform functions.
4. Add eBay monthly aggregation utility.
5. Add aggregate monthly likelihood.
6. Add deterministic expected-count prediction function.
7. Add training loop with Optax.
8. Add config.
9. Add CLI entry point.
10. Add output saving.
11. Add diagnostic plots.
12. Add tests.
13. Run tests and a tiny synthetic smoke test.
14. Run the real eBay pipeline.

---

## Acceptance criteria

The task is complete when:

1. The new eBay pipeline runs from the CLI.
2. It loads Amazon and eBay session-format data.
3. It aggregates eBay monthly purchases correctly.
4. It initializes eBay `pi`, `mu0`, and `k` from Amazon estimates.
5. It freezes Amazon CM parameters by default.
6. It fits eBay-specific `pi`, `mu0`, `k`, and choice parameters.
7. It outputs fitted parameters, monthly fit tables, forecast tables, and plots.
8. Unit tests pass.
9. Existing Amazon-only pipeline still works.
