# Housing Prices

This document describes the purpose and basic workflow of the housing prices prediction project in `machine-learning/projects/housing-prices/`.

## Project Workflow and Steps

1. Data Preparation
   - Place the raw data in `data/raw/`.
   - Data cleaning and feature engineering steps are organized in modules under `src/`.
   ```python
   import pandas as pd
   train = pd.read_csv('data/raw/train.csv')
   ```

2. Exploratory Data Analysis (EDA)
   - Analyze the dataset structure, missing values, distributions, and correlations in `notebooks/`.
   - EDA findings guide modeling decisions.

3. Preprocessing
   - Cleaning, missing value handling, and feature engineering are automated with functions under `src/`.
   ```python
   from src.preprocessing import clean_data
   df_cleaned = clean_data(pd.read_csv('data/raw/train.csv'))
   ```

4. Modeling
   - Train and evaluate different regression models in notebooks.

5. Prediction and Submission
   - Save the best model predictions to the `submissions/` folder.

## Folder Guidelines

- `data/raw/`: Original raw data files.
- `data/processed/`: Cleaned data ready for modeling.
- `notebooks/`: Project analysis, EDA, and modeling notebooks.
- `src/`: Reusable data processing and modeling helper code.
- `submissions/`: Model prediction outputs and submission files.

## Environment Setup

Recommended steps for this project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` file contains the Python packages required for this project.

---

