import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew

from data_preprocessing import load_data, fill_missing_values, remove_outliers, encode_features, add_features

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PLOTS_DIR = ROOT / "plots"

def main() -> None:
    print("Generating evaluation plots...")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Load raw data
    train_df = pd.read_csv(RAW_DIR / "train.csv")

    # Plot 1: Scatter plot to show outliers
    plt.figure(figsize=(10, 5))
    plt.scatter(train_df["GrLivArea"], train_df["SalePrice"], alpha=0.6, color="steelblue", label="Normal Houses")
    
    # Highlight the 2 anomalies
    anomalies = train_df[(train_df["GrLivArea"] > 4000) & (train_df["SalePrice"] < 200000)]
    plt.scatter(anomalies["GrLivArea"], anomalies["SalePrice"], color="crimson", s=100, edgecolor="black", linewidth=1.5, zorder=5, label="Anomalies (Removed)")
    
    plt.title("Outlier Identification: GrLivArea vs SalePrice", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Above Ground Living Area (GrLivArea) [sq ft]", fontsize=11)
    plt.ylabel("Sale Price ($)", fontsize=11)
    plt.legend(frameon=True, facecolor="white")
    outlier_path = PLOTS_DIR / "outlier_analysis.png"
    plt.tight_layout()
    plt.savefig(outlier_path, dpi=300)
    plt.close()
    print(f" - Saved: {outlier_path.name}")

    # Remove outliers for the next plots
    train_df = remove_outliers(train_df)

    # Plot 2: Bar chart showing correlations with SalePrice
    # Get correlations of numeric columns
    numeric_df = train_df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()["SalePrice"].sort_values(ascending=False)
    # Drop SalePrice itself and take top 15
    top_corr = correlations.drop("SalePrice").head(15)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_corr.values, y=top_corr.index, palette="Blues_r")
    plt.title("Top 15 Numerical Features Correlated with SalePrice", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Pearson Correlation Coefficient", fontsize=11)
    plt.ylabel("Features", fontsize=11)
    corr_path = PLOTS_DIR / "correlation_with_saleprice.png"
    plt.tight_layout()
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f" - Saved: {corr_path.name}")

    # Preprocess the data
    y = train_df["SalePrice"]
    y_log = np.log1p(y)
    train_features = train_df.drop(columns=["Id", "SalePrice"], errors="ignore")
    
    # Preprocess
    combined = fill_missing_values(train_features)
    combined = encode_features(combined)
    combined = add_features(combined)

    # Apply log-transform to skewed features
    numeric_cols = combined.select_dtypes(include=["number"]).columns
    skewed_feats = combined[numeric_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    high_skew = skewed_feats[abs(skewed_feats) > 0.75]
    for feat in high_skew.index:
        if combined[feat].nunique() > 2 and feat not in ["YearBuilt", "YearRemodAdd", "YrSold"]:
            combined[feat] = np.log1p(combined[feat])

    X = combined.select_dtypes(include=[np.number])

    # Run 5-Fold Cross-Validation
    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=15, random_state=42)),
        "ElasticNet": make_pipeline(StandardScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=42, max_iter=10000)),
        "HistGB": HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, random_state=42)
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = {name: np.zeros(len(X)) for name in models}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
        X_tr, y_tr = X.iloc[train_idx], y_log.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y_log.iloc[val_idx]
        
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            oof_predictions[name][val_idx] = model.predict(X_va)

    # Calculate weighted ensemble predictions
    w_ridge, w_enet, w_hgb = 0.30, 0.21, 0.49
    ensemble_oof = (
        w_ridge * oof_predictions["Ridge"] +
        w_enet * oof_predictions["ElasticNet"] +
        w_hgb * oof_predictions["HistGB"]
    )

    # Calculate RMSLE scores
    scores = {}
    for name, oof in oof_predictions.items():
        scores[name] = mean_squared_error(y_log, oof, squared=False)
    scores["Combined Ensemble"] = mean_squared_error(y_log, ensemble_oof, squared=False)

    # Plot 3: Bar chart comparing model validation scores
    plt.figure(figsize=(10, 5.5))
    colors = ["lightsteelblue", "lightsteelblue", "lightsteelblue", "royalblue"]
    bars = plt.bar(scores.keys(), scores.values(), color=colors, width=0.55, edgecolor="black", linewidth=0.7)
    
    # Add score values on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.002, f"{yval:.5f}", ha="center", va="bottom", fontweight="bold", color="black")

    plt.title("5-Fold Cross-Validation Scores (RMSLE) Comparison", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Validation RMSLE (Lower is Better)", fontsize=11)
    plt.ylim(0, max(scores.values()) * 1.25)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    cv_path = PLOTS_DIR / "model_cv_comparison.png"
    plt.tight_layout()
    plt.savefig(cv_path, dpi=300)
    plt.close()
    print(f" - Saved: {cv_path.name}")

    # Plot 4: Residual plot for the final ensemble
    # Predicted price vs prediction error
    predicted_prices = np.expm1(ensemble_oof)
    residuals = y - predicted_prices

    plt.figure(figsize=(11, 5.5))
    plt.scatter(predicted_prices, residuals, alpha=0.5, color="royalblue", edgecolor="none", s=30)
    plt.axhline(y=0, color="crimson", linestyle="--", linewidth=1.5, label="Perfect Predictions")
    plt.title("Ensemble Residual Analysis: Predicted vs Residual Price Error", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted SalePrice ($)", fontsize=11)
    plt.ylabel("Residual Error ($) [Actual - Predicted]", fontsize=11)
    plt.legend(frameon=True, facecolor="white")
    res_path = PLOTS_DIR / "ensemble_residuals.png"
    plt.tight_layout()
    plt.savefig(res_path, dpi=300)
    plt.close()
    print(f" - Saved: {res_path.name}")
    print("All plots generated successfully!")

if __name__ == "__main__":
    main()
