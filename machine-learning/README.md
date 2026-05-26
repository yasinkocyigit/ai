# Machine Learning

This folder contains structured study notes on core machine learning algorithms and end-to-end Kaggle competition projects.

---

## Folder Structure

```
machine-learning/
├── training-models/       # Algorithm notes, theory and implementation examples
│   ├── linearRegression/
│   ├── polynomialRegression/
│   ├── gradient-descent/
│   ├── learning-curve/
│   ├── logisticRegression/
│   └── regularizedLinearModels/
│       ├── ridge-regression.md
│       ├── lasso-regression.md
│       ├── elastic-net.md
│       └── early-stopping.md
└── projects/           
    ├── housing-prices/ 
    └── titanic/       
```

> Other folders under `projects/` are excluded from version control via `.gitignore`.

---

## Training Models

Study notes and implementation examples organized by topic:

| Topic | Description |
|---|---|
| `linearRegression/` | Normal Equation, iterative approach, model fundamentals |
| `polynomialRegression/` | Polynomial feature expansion and curve fitting |
| `gradient-descent/` | Batch, mini-batch and stochastic gradient descent with optimization techniques |
| `learning-curve/` | Diagnosing bias/variance tradeoff using learning curves |
| `logisticRegression/` | Binary classification and decision boundaries |
| `regularizedLinearModels/` | Ridge, Lasso, Elastic Net regularization and Early Stopping |

---

## Kaggle Projects

### 🏠 Housing Prices — Advanced Regression Techniques
> Predict house sale prices using the Ames Housing dataset.

**Techniques used:** Missing value imputation, feature engineering, skewness correction, Ridge / ElasticNet / HistGradientBoosting ensemble modeling, 5-fold cross-validation (RMSLE).

 [`projects/housing-prices/`](projects/housing-prices/)

---

### 🚢 Titanic — Machine Learning from Disaster
> Binary survival classification using passenger data.

**Techniques used:** Exploratory data analysis, feature encoding, Random Forest with GridSearchCV hyperparameter tuning.

 [`projects/titanic/`](projects/titanic/)
