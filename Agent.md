# AGENTS.md

## Project goal

Implement the Moe-Fader EV/CM model for Amazon browsing and purchase sessions.

## Data

The main dataset is `data/amazon_sessions.csv`.

Important columns:
- machine_id: customer/machine identifier
- event_date, event_time: session timestamp
- pages_viewed, duration: browsing intensity variables
- tran_flg: purchase indicator; missing means no transaction
- basket_tot, prod_totprice, prod_qty: purchase amount/quantity fields
- demographic columns: census_region, household_size, household_income, racial_background, country_of_origin

## Modeling instructions

Follow `EV_CM_IMPLEMENTATION_SPEC.md`.

Implement:
1. Data preprocessing and chronological train/test split.
2. EV model for evolving visits / browsing arrivals.
3. CM model for conversion conditional on visit behavior.
4. Joint or staged likelihood as specified.
5. MLE fitting with `scipy.optimize`.
6. Log-likelihood, AIC, BIC, parameter table, standard errors if feasible.
7. Forecast cumulative purchases over time.
8. Report MAPE of cumulative purchases on the holdout period.

## Expected commands

Create runnable scripts such as:

```bash
python -m src.prepare_data --input data/amazon_sessions.csv --out data/processed/
python -m src.fit_ev_cm --config configs/default.yaml
python -m src.forecast --config configs/default.yaml
python -m src.report --config configs/default.yaml
pytest
