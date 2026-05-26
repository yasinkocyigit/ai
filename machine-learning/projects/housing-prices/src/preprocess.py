from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew

from data_preprocessing import load_data, fill_missing_values, remove_outliers, encode_features, add_features

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

def save_processed_data() -> None:
    """Preprocess raw datasets and save them to the data/processed folder."""
    print("Preprocessing datasets and saving processed features...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw datasets
    train_df = load_data(RAW_DIR / "train.csv")
    test_df = load_data(RAW_DIR / "test.csv")

    # Remove outliers
    train_df = remove_outliers(train_df)

    # Keep target and IDs
    y = train_df["SalePrice"]
    train_ids = train_df["Id"]
    test_ids = test_df["Id"]

    # Drop columns we don't train on
    train_features = train_df.drop(columns=["Id", "SalePrice"], errors="ignore")
    test_features = test_df.drop(columns=["Id", "SalePrice"], errors="ignore")

    # Combine datasets so that one-hot encoding columns match perfectly
    combined = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)

    # Preprocess
    combined = fill_missing_values(combined)
    combined = encode_features(combined)
    combined = add_features(combined)

    # Apply log-transform to highly skewed numeric columns (skewness > 0.75)
    numeric_cols = combined.select_dtypes(include=["number"]).columns
    skewed_feats = combined[numeric_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    high_skew = skewed_feats[abs(skewed_feats) > 0.75]
    
    print(f"Applying log-transform to {len(high_skew)} skewed features...")
    for feat in high_skew.index:
        # Skip binary flags and year columns
        if combined[feat].nunique() > 2 and feat not in ["YearBuilt", "YearRemodAdd", "YrSold"]:
            combined[feat] = np.log1p(combined[feat])

    # Split combined features back to train and test sets
    X_train = combined.iloc[:len(train_df)].copy()
    X_test = combined.iloc[len(train_df):].copy()

    # Keep only numeric columns
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    # Save processed datasets
    train_features_path = PROCESSED_DIR / "train_features.csv"
    train_target_path = PROCESSED_DIR / "train_target.csv"
    train_processed_path = PROCESSED_DIR / "train_processed.csv"
    test_features_path = PROCESSED_DIR / "test_features.csv"
    test_id_path = PROCESSED_DIR / "test_id.csv"

    X_train.to_csv(train_features_path, index=False)
    y.to_frame("SalePrice").to_csv(train_target_path, index=False)
    pd.concat([X_train.reset_index(drop=True), y.reset_index(drop=True)], axis=1).to_csv(
        train_processed_path,
        index=False
    )
    X_test.to_csv(test_features_path, index=False)
    test_ids.to_frame("Id").to_csv(test_id_path, index=False)

    print(f"Successfully preprocessed and saved all outputs to: {PROCESSED_DIR}")
    print(f" - {train_features_path.name} (shape: {X_train.shape})")
    print(f" - {train_target_path.name}")
    print(f" - {train_processed_path.name}")
    print(f" - {test_features_path.name} (shape: {X_test.shape})")
    print(f" - {test_id_path.name}")

if __name__ == "__main__":
    save_processed_data()
