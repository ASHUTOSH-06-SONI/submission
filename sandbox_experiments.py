import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler
import joblib

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['market_index', 'quote_signal']:
        if col not in df.columns:
            df[col] = np.nan

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

    weight_positive = df['weight'].where(df['weight'] > 0, np.nan)
    df['distance_log'] = np.log1p(df['distance'])
    df['weight_log'] = np.log1p(weight_positive.fillna(0.0))
    df['distance_weight_ratio'] = (df['distance'] / weight_positive).replace([np.inf, -np.inf], np.nan).fillna(0.0)

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

SMALL_CATEGORICAL_FEATURES = [
    'equipment',
    'distance_bucket',
    'season',
]

LARGE_CATEGORICAL_FEATURES = [
    'pickup',
    'delivery',
    'pickup_delivery',
]

CATEGORICAL_FEATURES = SMALL_CATEGORICAL_FEATURES + LARGE_CATEGORICAL_FEATURES

SCALER_FACTORY = {
    'none': None,
    'standard': StandardScaler,
    'minmax': MinMaxScaler,
    'robust': RobustScaler,
}


def _missing_package(name: str):
    raise ImportError(f'{name} is not installed. Install it to use this model.')


def make_xgboost():
    if XGBRegressor is None:
        _missing_package('xgboost')
    return XGBRegressor(
        random_state=42,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        tree_method='hist',
        objective='reg:squarederror',
        n_jobs=-1,
    )


def make_lightgbm():
    if LGBMRegressor is None:
        _missing_package('lightgbm')
    return LGBMRegressor(
        random_state=42,
        n_estimators=100,
        max_depth=12,
        learning_rate=0.05,
        n_jobs=-1,
    )


def make_catboost():
    if CatBoostRegressor is None:
        _missing_package('catboost')
    return CatBoostRegressor(
        random_seed=42,
        iterations=200,
        depth=6,
        learning_rate=0.05,
        verbose=0,
    )

MODEL_FACTORY = {
    'histgb': lambda: HistGradientBoostingRegressor(
        random_state=42,
        max_iter=200,
        max_depth=12,
        learning_rate=0.05,
        loss='absolute_error',
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    ),
    'random_forest': lambda: RandomForestRegressor(
        random_state=42,
        n_estimators=100,
        max_depth=12,
        n_jobs=-1,
    ),
    'gradient_boosting': lambda: GradientBoostingRegressor(
        random_state=42,
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
    ),
    'ridge': lambda: Ridge(alpha=1.0),
    'xgboost': make_xgboost,
    'lightgbm': make_lightgbm,
    'catboost': make_catboost,
}


def make_pipeline(scaler_name: str, encoder_name: str, model_name: str, use_log_target: bool) -> Pipeline:
    numeric_steps = [('imputer', SimpleImputer(strategy='median'))]
    scaler_cls = SCALER_FACTORY[scaler_name]
    if scaler_cls is not None:
        numeric_steps.append((scaler_name, scaler_cls()))
    numeric_transformer = Pipeline(numeric_steps)

    if encoder_name == 'ordinal':
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ])
        transformers = [
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
        ]
    elif encoder_name == 'onehot_small':
        small_cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ])
        large_cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ])
        transformers = [
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('small_cat', small_cat_transformer, SMALL_CATEGORICAL_FEATURES),
            ('large_cat', large_cat_transformer, LARGE_CATEGORICAL_FEATURES),
        ]
    else:
        raise ValueError(f'Unknown encoder: {encoder_name}')

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        sparse_threshold=0,
    )

    regressor = MODEL_FACTORY[model_name]()
    pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', regressor)])

    if use_log_target:
        pipeline = TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
        )
    return pipeline


def evaluate_model_holdout(model, X_train, y_train, X_hold, y_hold):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_hold)
    y_pred = np.clip(y_pred, 0.0, None)
    return {
        'mae': float(mean_absolute_error(y_hold, y_pred)),
        'mape': float(mean_absolute_percentage_error(y_hold, y_pred)),
    }


def evaluate_model_cv(model, X, y, cv_folds, random_state):
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    maes = []
    mapes = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_hold = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_hold = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_hold)
        y_pred = np.clip(y_pred, 0.0, None)
        maes.append(mean_absolute_error(y_hold, y_pred))
        mapes.append(mean_absolute_percentage_error(y_hold, y_pred))
    return {
        'mae': float(np.mean(maes)),
        'mape': float(np.mean(mapes)),
    }


def load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def run_experiments(args: argparse.Namespace) -> None:
    train_df = load_df(Path(args.train_file))
    train_df = build_features(train_df)
    target_column = 'posted_rate'
    if target_column not in train_df.columns:
        raise ValueError(f'Training file must contain {target_column}.')

    X = train_df.drop(columns=['load_id', 'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon', 'date', target_column], errors='ignore')
    y = train_df[target_column]

    use_cv = args.cv_folds > 1
    use_holdout = args.holdout_fraction > 0.0 and not use_cv
    if not use_cv and not use_holdout:
        raise ValueError('Enable either cross-validation by setting --cv-folds > 1 or holdout by setting --holdout-fraction > 0.')

    combos = []
    for scaler_name in args.scalers.split(','):
        for encoder_name in args.encoders.split(','):
            for model_name in args.models.split(','):
                for log_target in args.log_target.split(','):
                    combos.append((scaler_name, encoder_name, model_name, log_target == 'true'))

    results = []
    run_count = 0
    max_runs = int(args.max_runs) if args.max_runs is not None and args.max_runs > 0 else None
    for scaler_name, encoder_name, model_name, use_log_target in combos:
        if max_runs is not None and run_count >= max_runs:
            print(f"Reached max runs ({max_runs}); stopping further experiments.")
            break
        print(f'Running: scaler={scaler_name}, encoder={encoder_name}, model={model_name}, log_target={use_log_target}')
        try:
            pipeline = make_pipeline(scaler_name, encoder_name, model_name, use_log_target)
        except (ValueError, ImportError) as exc:
            print(f'  Skipping {model_name} because: {exc}')
            continue

        if use_cv:
            metrics = evaluate_model_cv(pipeline, X, y, args.cv_folds, args.random_state)
        else:
            X_fit, X_hold, y_fit, y_hold = train_test_split(
                X, y, test_size=args.holdout_fraction, random_state=args.random_state
            )
            metrics = evaluate_model_holdout(pipeline, X_fit, y_fit, X_hold, y_hold)

        print(f"  MAE: {metrics['mae']:.2f}, MAPE: {metrics['mape']:.4f}\n")
        results.append({
            'scaler': scaler_name,
            'encoder': encoder_name,
            'model': model_name,
            'log_target': use_log_target,
            **metrics,
        })
        run_count += 1

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['mae', 'mape']).reset_index(drop=True)
    print('=== SUMMARY ===')
    print(results_df.to_string(index=False))
    # Print top-k concise summary
    topk = int(args.top_k) if args.top_k is not None else 5
    print('\n=== TOP {} CONFIGS ==='.format(topk))
    print(results_df.head(topk).to_string(index=False))
    # Optionally save and/or export best model trained on full data
    if args.save_best_model:
        if results_df.shape[0] == 0:
            print('No results to build best model from.')
        else:
            best = results_df.iloc[0]
            print(f"Training best model on full data: {best['model']} with scaler={best['scaler']} encoder={best['encoder']} log_target={best['log_target']}")
            best_pipeline = make_pipeline(best['scaler'], best['encoder'], best['model'], bool(best['log_target']))
            # train on all available data
            try:
                if use_cv:
                    best_pipeline.fit(X, y)
                else:
                    best_pipeline.fit(X, y)
                joblib.dump(best_pipeline, args.save_best_model)
                print(f'Wrote best model pipeline to {args.save_best_model}')
            except Exception as e:
                print('Failed to train/save best model:', e)
    if args.output_summary:
        results_df.to_csv(args.output_summary, index=False)
        print(f'Wrote summary to {args.output_summary}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run preprocessing plus model sandbox experiments.')
    parser.add_argument('--train-file', type=str, default='train-test.csv', help='Training CSV file path')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--scalers', type=str, default='none,standard,minmax,robust',
                        help='Comma-separated scaler names to evaluate: none,standard,minmax,robust')
    parser.add_argument('--encoders', type=str, default='ordinal,onehot_small',
                        help='Comma-separated encoder options to evaluate: ordinal,onehot_small')
    parser.add_argument('--models', type=str, default='histgb,random_forest,gradient_boosting,ridge,xgboost,lightgbm',
                        help='Comma-separated model names to evaluate: histgb,random_forest,gradient_boosting,ridge,xgboost,lightgbm,catboost')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of cross-validation folds; if >1, cross-validation is used instead of holdout')
    parser.add_argument('--holdout-fraction', type=float, default=0.0, help='Holdout split fraction if cv-folds is 1 or less')
    parser.add_argument('--log-target', type=str, default='false,true',
                        help='Comma-separated booleans to evaluate target log transform: false,true')
    parser.add_argument('--max-runs', type=int, default=0,
                        help='Optional maximum number of experiment runs to execute (0 = no limit)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='How many top configurations to print in summary')
    parser.add_argument('--save-best-model', type=str, default='',
                        help='Path to save the best trained pipeline (joblib)')
    parser.add_argument('--output-summary', type=str, default='sandbox_results.csv', help='Optional CSV file to save results')
    args = parser.parse_args()
    run_experiments(args)
