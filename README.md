# Freight Rate Prediction

This repository builds a model from `train-test.csv`, predicts `validation.csv`, and fills out December chart inputs.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python predict_rates.py
```

This creates:

- `validation_predictions.csv`
- `data/december_chart_inputs.csv`

## Notes

- Training uses `train-test.csv`.
- Validation output is populated from `validation-predictions-template.csv`.
- December predictions are written to `data/december_chart_inputs.csv`.
