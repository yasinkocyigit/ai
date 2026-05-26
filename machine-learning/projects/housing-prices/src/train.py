from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_preprocessing import load_data, fill_missing_values, remove_outliers, encode_features, add_features

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SUBMISSION_DIR = ROOT / "submissions"

def train_and_evaluate() -> None:
    """Train regression models, evaluate them, and create the final Kaggle submission."""
    print("Starting Advanced Housing Prices Model Training Pipeline...")
    
    # Load datasets
    train_df = load_data(RAW_DIR / "train.csv")
    test_df = load_data(RAW_DIR / "test.csv")
    
    # Remove outliers
    train_df = remove_outliers(train_df)
    
    # Keep targets and IDs
    y = train_df["SalePrice"]
    y_log = np.log1p(y)
    
    test_ids = test_df["Id"]
    
    train_features = train_df.drop(columns=["Id", "SalePrice"], errors="ignore")
    test_features = test_df.drop(columns=["Id", "SalePrice"], errors="ignore")
    
    # Combine datasets temporarily so that one-hot encoded columns align perfectly
    combined = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)
    
    # Preprocess the datasets
    combined = fill_missing_values(combined)
    combined = encode_features(combined)
    combined = add_features(combined)
    
    # Apply log-transform to skewed features (skewness > 0.75)
    numeric_cols = combined.select_dtypes(include=["number"]).columns
    skewed_feats = combined[numeric_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    high_skew = skewed_feats[abs(skewed_feats) > 0.75]
    
    print(f"Log-transforming {len(high_skew)} highly skewed features...")
    for feat in high_skew.index:
        # Skip binary flags and year columns
        if combined[feat].nunique() > 2 and feat not in ["YearBuilt", "YearRemodAdd", "YrSold"]:
            combined[feat] = np.log1p(combined[feat])
            
    # Split back to train and test sets
    X = combined.iloc[:len(train_df)].copy()
    X_test = combined.iloc[len(train_df):].copy()
    
    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])
    
    print(f"Total training features aligned: {X.shape[1]}")
    
    # Define models
    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=15, random_state=42)),
        "ElasticNet": make_pipeline(StandardScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=42, max_iter=10000)),
        "HistGB": HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, random_state=42)
    }
    
    # 5-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = {name: np.zeros(len(X)) for name in models}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
        X_tr, y_tr = X.iloc[train_idx], y_log.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y_log.iloc[val_idx]
        
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            oof_predictions[name][val_idx] = model.predict(X_va)
            
    # Print validation scores for each model
    print("\n--- Out-of-Fold 5-Fold CV Validation Scores (RMSLE) ---")
    for name, oof_preds in oof_predictions.items():
        rmsle = mean_squared_error(y_log, oof_preds, squared=False)
        print(f" - {name:12s} RMSLE: {rmsle:.5f}")
        
    # Weighted Ensemble of predictions
    w_ridge, w_enet, w_hgb = 0.30, 0.21, 0.49
    ensemble_oof = (
        w_ridge * oof_predictions["Ridge"] +
        w_enet * oof_predictions["ElasticNet"] +
        w_hgb * oof_predictions["HistGB"]
    )
    ensemble_rmsle = mean_squared_error(y_log, ensemble_oof, squared=False)
    print(f" - Ensemble (Ridge+ElasticNet+HistGB) RMSLE: {ensemble_rmsle:.5f}")
    
    # Train models on all training data and predict test set
    print("\nFitting final estimators on the full training dataset...")
    test_preds = np.zeros(len(X_test))
    
    # Train Ridge
    models["Ridge"].fit(X, y_log)
    test_preds += w_ridge * models["Ridge"].predict(X_test)
    
    # Train ElasticNet
    models["ElasticNet"].fit(X, y_log)
    test_preds += w_enet * models["ElasticNet"].predict(X_test)
    
    # Train HistGB
    models["HistGB"].fit(X, y_log)
    test_preds += w_hgb * models["HistGB"].predict(X_test)
    
    # Convert predictions back from log scale
    final_prices = np.expm1(test_preds)
    
    # Save the final ensembled predictions
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUBMISSION_DIR / "final_submission.csv"
    submission = pd.DataFrame({"Id": test_ids, "SalePrice": final_prices})
    submission.to_csv(submission_path, index=False)
    
    print(f"\nSuccessfully generated and saved ensembled predictions to: {submission_path}")
    print("Process completed! Ready to upload to Kaggle.")

if __name__ == "__main__":
    train_and_evaluate()
