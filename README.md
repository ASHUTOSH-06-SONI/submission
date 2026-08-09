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
