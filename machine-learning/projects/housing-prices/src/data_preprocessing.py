from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def load_data(path: str | Path) -> pd.DataFrame:
    """Load a CSV file from a path."""
    return pd.read_csv(path)

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in the dataset using simple rules."""
    df = df.copy()

    # Fill missing LotFrontage using the median LotFrontage of that neighborhood
    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )
    if "LotFrontage" in df.columns:
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    # Categorical features where missing value means the house doesn't have it
    none_cols = [
        "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
        "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "PoolQC", "Fence", "MiscFeature", "MasVnrType"
    ]
    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # Numeric features where missing value means the house doesn't have it (so 0)
    zero_cols = [
        "GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2",
        "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"
    ]
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Fill any other missing categoricals with the most common value (mode)
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Fill any other missing numerics with the median value
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers (houses with area > 4000 sq ft but very low price)."""
    df = df.copy()
    if "GrLivArea" in df.columns and "SalePrice" in df.columns:
        df = df.drop(df[(df["GrLivArea"] > 4000) & (df["SalePrice"] < 200000)].index)
    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical text columns into numbers."""
    df = df.copy()

    # Map quality ratings from text to numbers
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

    # One-hot encode nominal columns (use dtype=int so columns are numeric, not boolean)
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
    """Create new custom features from the existing columns."""
    df = df.copy()

    # Total indoor area of the house
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    
    # Total bathroom count (Full bath counts as 1.0, half bath as 0.5)
    df["TotalBath"] = (
        df["FullBath"] + 0.5 * df["HalfBath"] +
        df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    )
    
    # Total porch and deck area
    df["TotalPorchSF"] = (
        df["OpenPorchSF"] + df["EnclosedPorch"] +
        df["3SsnPorch"] + df["ScreenPorch"] + df["WoodDeckSF"]
    )
    
    # Age features based on Year Sold
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    
    # Garage Age (set to 0 if there is no garage)
    df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
    df.loc[df["GarageYrBlt"] == 0, "GarageAge"] = 0

    # Binary flags (1 if present or modified, 0 otherwise)
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)

    # Interactions between overall quality and area
    df["OverallQual_GrLivArea"] = df["OverallQual"] * df["GrLivArea"]
    df["OverallQual_TotalSF"] = df["OverallQual"] * df["TotalSF"]
    df["OverallQual_OverallCond"] = df["OverallQual"] * df["OverallCond"]

    return df
