from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

FEATURE_COLUMNS = [
    "luas_bangunan",
    "luas_tanah",
    "jumlah_kamar",
    "jumlah_kamar_mandi",
    "usia_rumah",
    "jarak_ke_pusat_kota",
]
TARGET_COLUMN = "harga_rumah"


def rmse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred, squared=False))


def main() -> None:
    data_dir = Path("artifacts/data")
    model_dir = Path("artifacts/model")
    model_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Data belum ada. Jalankan dulu: python src/generate_data.py"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train_full = train_df[FEATURE_COLUMNS]
    y_train_full = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    # Validation split for early stopping.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    # Hyperparameters focused on reducing overfitting.
    param_dist = {
        "n_estimators": [200, 300, 500, 700],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [3, 4, 5, 6],
        "min_child_weight": [1, 3, 5, 7],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "gamma": [0, 0.1, 0.2, 0.4],
        "reg_alpha": [0, 0.01, 0.1, 0.5],
        "reg_lambda": [1, 1.5, 2, 3],
    }

    tuner = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=30,
        scoring="neg_root_mean_squared_error",
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )

    tuner.fit(X_train, y_train)
    best_params = tuner.best_params_

    best_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **best_params,
    )

    best_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)

    metrics = {
        "train_rmse": rmse(y_train, train_pred),
        "test_rmse": rmse(y_test, test_pred),
        "train_mae": float(mean_absolute_error(y_train, train_pred)),
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
        "train_r2": float(r2_score(y_train, train_pred)),
        "test_r2": float(r2_score(y_test, test_pred)),
        "overfit_gap_rmse": rmse(y_test, test_pred) - rmse(y_train, train_pred),
        "best_params": best_params,
    }

    model_path = model_dir / "xgboost_harga_rumah.joblib"
    metadata_path = model_dir / "metadata.json"

    joblib.dump(best_model, model_path)

    metadata = {
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "metrics": metrics,
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Best Parameters:")
    print(json.dumps(best_params, indent=2))
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nModel saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
