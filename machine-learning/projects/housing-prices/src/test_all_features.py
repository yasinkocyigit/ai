from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_preprocessing import load_data, fill_missing_values, remove_outliers, encode_features, add_features

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

def test_full_pipeline(random_state: int = 42) -> None:
    # 1. Load raw training data
    df = load_data(RAW_DIR / "train.csv")
    
    # 2. Preprocess (fill missing, outliers, encode, engineer)
    df = fill_missing_values(df)
    df = remove_outliers(df)
    df = encode_features(df)
    df = add_features(df)
    
    # 3. Use ALL columns except 'Id' and 'SalePrice' as features
    X = df.drop(columns=["Id", "SalePrice"], errors="ignore")
    y = df["SalePrice"]
    
    # Ensure there are no remaining object types
    object_cols = X.select_dtypes(include="object").columns
    if len(object_cols) > 0:
        print(f"Warning: object columns remaining: {list(object_cols)}")
        X = X.drop(columns=object_cols)
        
    print(f"Number of features after full preprocessing: {X.shape[1]}")
    
    # 4. Train/validation split (80/20)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # 5. Train Ridge Model
    pipeline = make_pipeline(StandardScaler(), Ridge(alpha=10, random_state=random_state))
    y_train_log = np.log1p(y_train)
    pipeline.fit(X_train, y_train_log)
    
    # 6. Evaluate
    y_valid_pred = np.expm1(pipeline.predict(X_valid))
    y_valid_true = y_valid
    
    # Calculate R2 and RMSE (in dollars)
    rmse = mean_squared_error(y_valid_true, y_valid_pred, squared=False)
    r2 = r2_score(y_valid_true, y_valid_pred)
    
    # Also calculate RMSLE (which is RMSE on log scale - Kaggle's metric)
    rmsle = mean_squared_error(np.log1p(y_valid_true), np.log1p(y_valid_pred), squared=False)
    
    print("\n--- Validation Results with All Features ---")
    print(f" - R² Score: {r2:.4f}")
    print(f" - RMSE (SalePrice scale): {rmse:.2f}")
    print(f" - RMSLE (Log scale - Kaggle metric): {rmsle:.5f}")

if __name__ == "__main__":
    test_full_pipeline()
