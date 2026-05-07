# Amazon ML EV/CM pipeline

This additive pipeline fits an Amazon-only EV/CM model with a linear covariate head. It uses machine-level demographic covariates to produce heterogeneous EV/CM parameters:

```text
theta_i = constraints(base + X_i @ W)
```

The default covariates are intentionally time-invariant and exogenous-looking:

- `census_region`
- `household_size`
- `household_income`
- `racial_background`
- `country_of_origin`

Behavioral/session outcome columns such as `tran_flg`, `basket_tot`, `prod_totprice`, `prod_qty`, `pages_viewed`, and `duration` are not used by the default covariate head.

## Train

```bash
python scripts/train_amazon_ml_evcm.py --config configs/amazon_ml_evcm.yaml
```

Key outputs under `outputs.workdir`:

- `machine_parameter_predictions.csv`
- `segment_parameter_summary.csv`
- `covariate_coefficients.csv`
- `covariate_metadata.json`
- `fitted_params.json`
- `training_loss.csv`
- `model_comparison.csv`
- `plots/training_loss.png`
- `plots/param_by_income.png`
- `plots/param_by_region.png`
- `plots/covariate_coefficients_heatmap.png`

## Evaluate / regenerate plots

```bash
python scripts/evaluate_amazon_ml_evcm.py --fit-path outputs/amazon_ml_evcm
```

## Notes

- The linear head starts close to the homogeneous Amazon EV/CM model when `w_init_scale: 0.0`.
- Increase `fit.lambda_beta` if covariate effects look too noisy.
- Use `covariates.rare_min_count` to group rare categorical levels into `Other`.
