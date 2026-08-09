import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TARGET_COLUMN = 'posted_rate'
DROP_COLUMNS = [
    'load_id',
    'pickup_lat',
    'pickup_lon',
    'delivery_lat',
    'delivery_lon',
    'date',
    TARGET_COLUMN,
]
NUMERIC_FEATURES = [
    'distance_log',
    'weight_log',
    'market_index',
    'quote_signal',
    'month',
    'day_of_week',
    'day',
    'weight_missing',
    'market_missing',
    'distance_weight_ratio',
]
SMALL_CATEGORICAL_FEATURES = ['equipment', 'distance_bucket', 'season']
LARGE_CATEGORICAL_FEATURES = ['pickup', 'delivery', 'pickup_delivery']


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in ['market_index', 'quote_signal']:
        if column not in df.columns:
            df[column] = np.nan

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day'] = df['date'].dt.day

    df['pickup'] = df['pickup'].fillna('__missing__').astype(str)
    df['delivery'] = df['delivery'].fillna('__missing__').astype(str)
    df['equipment'] = df['equipment'].fillna('__missing__').astype(str)
    df['pickup_delivery'] = df['pickup'] + '|' + df['delivery']

    df['weight_missing'] = df['weight'].isna().astype(int)
    df['market_missing'] = df['market_index'].isna().astype(int)

    positive_weight = df['weight'].where(df['weight'] > 0, np.nan)
    df['distance_log'] = np.log1p(df['distance'])
    df['weight_log'] = np.log1p(positive_weight.fillna(0.0))
    df['distance_weight_ratio'] = (
        df['distance'] / positive_weight
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df['distance_bucket'] = pd.cut(
        df['distance'],
        bins=[0, 500, 1000, 1500, 2000, 2500, 3500, 10000],
        labels=[
            '0-500',
            '500-1000',
            '1000-1500',
            '1500-2000',
            '2000-2500',
            '2500-3500',
            '3500+',
        ],
    ).astype(str)

    df['season'] = df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall',
    }).fillna('Unknown')

    return df


def make_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    small_cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False)),
    ])

    large_cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('small_cat', small_cat_transformer, SMALL_CATEGORICAL_FEATURES),
            ('large_cat', large_cat_transformer, LARGE_CATEGORICAL_FEATURES),
        ],
        remainder='drop',
        sparse_threshold=0,
    )


def make_model() -> Pipeline:
    pipeline = Pipeline([
        ('preprocessor', make_preprocessor()),
        ('regressor', HistGradientBoostingRegressor(
            random_state=42,
            max_iter=300,
            max_depth=12,
            learning_rate=0.05,
            loss='absolute_error',
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )),
    ])

    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_validation_predictions(template_path: Path, predictions: np.ndarray, output_path: Path) -> None:
    template = pd.read_csv(template_path)
    if list(template.columns) != ['load_id', 'predicted_rate']:
        raise ValueError('Validation template must contain exactly load_id and predicted_rate columns.')
    template['predicted_rate'] = np.round(predictions, 2)
    template.to_csv(output_path, index=False)


def save_december_predictions(source_path: Path, predictions: np.ndarray, output_path: Path) -> None:
    df = pd.read_csv(source_path)
    df['predicted_rate'] = np.round(predictions, 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=DROP_COLUMNS, errors='ignore')
    y = df[TARGET_COLUMN]
    return X, y


def evaluate_holdout(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_hold: pd.DataFrame, y_hold: pd.Series) -> dict:
    model.fit(X_train, y_train)
    predictions = np.clip(model.predict(X_hold), 0.0, None)
    return {
        'mae': float(mean_absolute_error(y_hold, predictions)),
        'mape': float(mean_absolute_percentage_error(y_hold, predictions)),
    }


def main(args: argparse.Namespace) -> None:
    train_df = build_features(load_csv(Path(args.train_file)))
    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(f'Training file must contain {TARGET_COLUMN}.')

    X_train, y_train = prepare_training_data(train_df)
    model = make_model()

    if args.holdout_fraction > 0.0:
        from sklearn.model_selection import train_test_split

        X_fit, X_hold, y_fit, y_hold = train_test_split(
            X_train,
            y_train,
            test_size=args.holdout_fraction,
            random_state=args.random_state,
        )
        metrics = evaluate_holdout(model, X_fit, y_fit, X_hold, y_hold)
        print(f'Holdout MAE: {metrics["mae"]:.2f}')
        print(f'Holdout MAPE: {metrics["mape"]:.4f}')
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)

    validation_df = build_features(load_csv(Path(args.validation_file)))
    X_validation = validation_df.drop(columns=['load_id', 'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon', 'date'], errors='ignore')
    validation_predictions = np.clip(model.predict(X_validation), 0.0, None)
    save_validation_predictions(Path(args.validation_template), validation_predictions, Path(args.validation_output))
    print(f'Saved validation predictions to {args.validation_output}')

    december_df = build_features(load_csv(Path(args.december_file)))
    X_december = december_df.drop(columns=['date'], errors='ignore')
    december_predictions = np.clip(model.predict(X_december), 0.0, None)
    save_december_predictions(Path(args.december_file), december_predictions, Path(args.december_output))
    print(f'Saved December predictions to {args.december_output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a freight rate model and generate predictions.')
    parser.add_argument('--train-file', type=str, default='data/train-test.csv', help='Training data file')
    parser.add_argument('--validation-file', type=str, default='data/validation.csv', help='Validation data file')
    parser.add_argument('--validation-template', type=str, default='data/validation-predictions-template.csv', help='Validation output template file')
    parser.add_argument('--december-file', type=str, default='data/december-chart-inputs.csv', help='December chart input file')
    parser.add_argument('--validation-output', type=str, default='outputs/validation_predictions.csv', help='Validation output file')
    parser.add_argument('--december-output', type=str, default='outputs/december_chart_inputs.csv', help='December output file')
    parser.add_argument('--holdout-fraction', type=float, default=0.2, help='Holdout fraction for evaluation')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    main(args)
