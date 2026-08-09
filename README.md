# Freight Rate Prediction

This repository builds a freight rate model from raw data, generates validation predictions, and fills December chart inputs.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python predict_rates.py
```

This reads inputs from `data/` and writes outputs to `outputs/`.

This creates:

- `outputs/validation_predictions.csv`
- `outputs/december_chart_inputs.csv`

## Notes

- Training uses `data/train-test.csv`.
- Validation output is populated from `data/validation-predictions-template.csv`.
- December predictions are read from `data/december-chart-inputs.csv`.

## Final Results

- **Holdout MAE:** 81.31
- **Holdout MAPE:** 0.0386
- **Training set:** 48,000 rows (`data/train-test.csv`)
- **Validation set:** 12,000 rows (`data/validation.csv`)
- **December predictions:** 31 rows (`outputs/december_chart_inputs.csv`)
- **Model / pipeline:** `HistGradientBoostingRegressor` wrapped with `TransformedTargetRegressor` (log1p/expm1); numeric imputation + `StandardScaler`; small-category `OneHotEncoder`; large-category `OrdinalEncoder`.
- **Generated outputs:** `outputs/validation_predictions.csv`, `outputs/december_chart_inputs.csv`, `outputs/december_predictions_chart.png`
- **Repository:** https://github.com/ASHUTOSH-06-SONI/submission (main branch)

If you want this section expanded into a one-page summary for the PDF report, tell me and I will generate it.
