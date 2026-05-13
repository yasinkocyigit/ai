import pandas as pd
import numpy as np

def preprocess_titanic(df):
    """
    Applies all cleaning and feature engineering steps for the Titanic dataset.
    Can be used for both training and testing data.
    """
    
    # Dropping unnecessary columns
    # Name and PassengerId can be kept for now as they might be needed later
    drop_cols = ['Cabin', 'Ticket'] 
    df.drop(drop_cols, axis=1, inplace=True)
    
    # Filling missing values for Age
    mean_age = df['Age'].mean()
    std_age = df['Age'].std()
    null_count = df['Age'].isnull().sum()
    if null_count > 0:
        random_ages = np.random.randint(mean_age - std_age, mean_age + std_age, size=null_count)
        df.loc[df['Age'].isnull(), 'Age'] = random_ages
    df['Age'] = df['Age'].astype(int)
    
    # Filling missing values for Embarked (using Mode)
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        
    # Filling missing values for Fare (specifically for potential missing values in the test set)
    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Feature Engineering (FamilySize)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    
    # Extracting and Grouping Titles
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
        'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare',
        'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare',
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'
    }
    df['Title'] = df['Title'].replace(title_mapping)
    
    # Dropping remaining unnecessary text columns
    df.drop(['Name'], axis=1, inplace=True)
    
    return df

if __name__ == "__main__":
    import os
    
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    
    # Read raw data
    train_path = os.path.join(raw_dir, 'train.csv')
    test_path = os.path.join(raw_dir, 'test.csv')
    
    print("Loading raw data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Process data
    print("Preprocessing data...")
    train_processed = preprocess_titanic(train_df)
    test_processed = preprocess_titanic(test_df)
    
    # Save processed data
    train_out_path = os.path.join(processed_dir, 'train_processed.csv')
    test_out_path = os.path.join(processed_dir, 'test_processed.csv')
    
    train_processed.to_csv(train_out_path, index=False)
    test_processed.to_csv(test_out_path, index=False)
    
    print(f"Preprocessing completed! Processed files saved to {processed_dir}")