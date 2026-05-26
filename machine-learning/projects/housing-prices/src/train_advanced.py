from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SUBMISSION_DIR = ROOT / "submissions"

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Impute LotFrontage by Neighborhood median
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )
    df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    # Categorical columns where NaN means "None" (absence of feature)
    none_cols = [
        "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
        "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "PoolQC", "Fence", "MiscFeature", "MasVnrType"
    ]
    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # Numeric columns where NaN means 0 (absence of feature)
    zero_cols = [
        "GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2",
        "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"
    ]
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # For other missing categoricals, fill with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # For other missing numerics, fill with median
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ordinal mapping
    qual_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    qual_cols = [
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
        "HeatingQC", "KitchenQual", "FireplaceQu",
        "GarageQual", "GarageCond", "PoolQC"
    ]
    for col in qual_cols:
        if col in df.columns:
            df[col] = df[col].map(qual_map)

    if "BsmtExposure" in df.columns:
        df["BsmtExposure"] = df["BsmtExposure"].map({"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4})

    bsmt_fin_map = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
    for col in ["BsmtFinType1", "BsmtFinType2"]:
        if col in df.columns:
            df[col] = df[col].map(bsmt_fin_map)

    if "GarageFinish" in df.columns:
        df["GarageFinish"] = df["GarageFinish"].map({"None": 0, "Unf": 1, "RFn": 2, "Fin": 3})

    if "Fence" in df.columns:
        df["Fence"] = df["Fence"].map({"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4})

    if "CentralAir" in df.columns:
        df["CentralAir"] = df["CentralAir"].map({"N": 0, "Y": 1})

    if "PavedDrive" in df.columns:
        df["PavedDrive"] = df["PavedDrive"].map({"N": 0, "P": 1, "Y": 2})

    # One-hot encode nominal columns with dtype=int to preserve them in numeric selectors
    one_hot_cols = [
        "MSZoning", "Street", "Alley", "LotShape", "LandContour",
        "Utilities", "LotConfig", "LandSlope", "Neighborhood",
        "Condition1", "Condition2", "BldgType", "HouseStyle",
        "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
        "MasVnrType", "Foundation", "Heating", "Electrical",
        "Functional", "GarageType", "MiscFeature", "SaleType",
        "SaleCondition"
    ]
    one_hot_cols = [col for col in one_hot_cols if col in df.columns]
    df = pd.get_dummies(df, columns=one_hot_cols, dtype=int)

    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Total square footage of the house
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    
    # Total bathroom count (Full bath + 0.5 * Half bath)
    df["TotalBath"] = (
        df["FullBath"] + 0.5 * df["HalfBath"] +
        df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    )
    
    # Total porch area
    df["TotalPorchSF"] = (
        df["OpenPorchSF"] + df["EnclosedPorch"] +
        df["3SsnPorch"] + df["ScreenPorch"] + df["WoodDeckSF"]
    )
    
    # Age features
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    
    # Garage Age: 0 if no garage, else YrSold - GarageYrBlt
    df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
    df.loc[df["GarageYrBlt"] == 0, "GarageAge"] = 0

    # Binary flags
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)

    # Highly predictive interactions
    df["OverallQual_GrLivArea"] = df["OverallQual"] * df["GrLivArea"]
    df["OverallQual_TotalSF"] = df["OverallQual"] * df["TotalSF"]
    df["OverallQual_OverallCond"] = df["OverallQual"] * df["OverallCond"]

    return df

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Recommended outlier deletion from the dataset author
    df = df.drop(df[(df["GrLivArea"] > 4000) & (df["SalePrice"] < 200000)].index)
    return df

def main() -> None:
    print("Starting Advanced Housing Prices Model Training...")
    
    # 1. Load datasets
    train_df = pd.read_csv(RAW_DIR / "train.csv")
    test_df = pd.read_csv(RAW_DIR / "test.csv")
    
    # Remove outliers
    train_df = remove_outliers(train_df)
    
    # Extract target and IDs
    y = train_df["SalePrice"]
    y_log = np.log1p(y)
    
    test_ids = test_df["Id"]
    
    train_features = train_df.drop(columns=["Id", "SalePrice"], errors="ignore")
    test_features = test_df.drop(columns=["Id", "SalePrice"], errors="ignore")
    
    # Combine train and test features for uniform preprocessing
    combined = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)
    
    # Preprocess
    combined = fill_missing_values(combined)
    combined = encode_features(combined)
    combined = add_features(combined)
    
    # Fix skewness on independent numeric features
    numeric_cols = combined.select_dtypes(include=["number"]).columns
    skewed_feats = combined[numeric_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    high_skew = skewed_feats[abs(skewed_feats) > 0.75]
    
    print(f"Applying log-transform to {len(high_skew)} skewed features...")
    for feat in high_skew.index:
        if combined[feat].nunique() > 2 and feat not in ["YearBuilt", "YearRemodAdd", "YrSold"]:
            combined[feat] = np.log1p(combined[feat])
            
    # Split back into train and test features
    X = combined.iloc[:len(train_df)].copy()
    X_test = combined.iloc[len(train_df):].copy()
    
    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])
    
    print(f"Total training features: {X.shape[1]}")
    
    # 2. Define our optimized models
    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=15, random_state=42)),
        "ElasticNet": make_pipeline(StandardScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=42, max_iter=10000)),
        "HistGB": HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, random_state=42)
    }
    
    # 3. Validation stage (5-Fold CV) to double-check performance
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = {name: np.zeros(len(X)) for name in models}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
        X_tr, y_tr = X.iloc[train_idx], y_log.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y_log.iloc[val_idx]
        
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            oof_predictions[name][val_idx] = model.predict(X_va)
            
    # Calculate OOF scores
    print("\n--- Out-of-Fold Validation Results ---")
    for name, oof_preds in oof_predictions.items():
        rmsle = mean_squared_error(y_log, oof_preds, squared=False)
        print(f" - {name:12s} RMSLE: {rmsle:.5f}")
        
    # Ensemble OOF score using optimized weights
    w_ridge, w_enet, w_hgb = 0.30, 0.21, 0.49
    ensemble_oof = (
        w_ridge * oof_predictions["Ridge"] +
        w_enet * oof_predictions["ElasticNet"] +
        w_hgb * oof_predictions["HistGB"]
    )
    ensemble_rmsle = mean_squared_error(y_log, ensemble_oof, squared=False)
    print(f" - Ensemble (Combined) RMSLE: {ensemble_rmsle:.5f}")
    
    # 4. Train final models on FULL training data and predict
    print("\nTraining models on full training data...")
    test_preds = np.zeros(len(X_test))
    
    # Model 1: Ridge
    models["Ridge"].fit(X, y_log)
    test_preds += w_ridge * models["Ridge"].predict(X_test)
    
    # Model 2: ElasticNet
    models["ElasticNet"].fit(X, y_log)
    test_preds += w_enet * models["ElasticNet"].predict(X_test)
    
    # Model 3: HistGB
    models["HistGB"].fit(X, y_log)
    test_preds += w_hgb * models["HistGB"].predict(X_test)
    
    # Convert log predictions back to standard price scale using expm1
    final_prices = np.expm1(test_preds)
    
    # 5. Save submission
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUBMISSION_DIR / "final_submission.csv"
    submission = pd.DataFrame({"Id": test_ids, "SalePrice": final_prices})
    submission.to_csv(submission_path, index=False)
    
    print(f"\nSaved final ensemble predictions to: {submission_path}")
    print("Process complete! Ready for Kaggle upload.")

if __name__ == "__main__":
    main()
