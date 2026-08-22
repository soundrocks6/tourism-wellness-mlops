# -----------------------------------------------------------------------------
# prep.py
# Data preparation step of the pipeline.
# - Loads the dataset directly from the repository data folder
# - Cleans the data and removes unnecessary columns
# - Splits the data into train and test sets and saves them locally as CSVs
#   (the workflow passes these files to the next job as an artifact)
# -----------------------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

def main():
    # ---- Load the dataset from the repository data folder ----
    df = pd.read_csv(DATA_PATH, index_col=0)
    print(f"Loaded dataset with shape {df.shape}")

    # ---- Data cleaning ----
    # 1. Drop unnecessary columns: CustomerID is a unique identifier and
    #    carries no predictive information.
    df = df.drop(columns=["CustomerID"])
    print("Dropped unnecessary column: CustomerID")

    # 2. Fix the known data-entry inconsistency in Gender ('Fe Male' -> 'Female')
    df["Gender"] = df["Gender"].replace("Fe Male", "Female")
    print("Standardised Gender categories:", df["Gender"].unique().tolist())

    # 3. Remove duplicate rows, if any
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows")

    # 4. Handle missing values, if any (median for numeric, mode for categorical)
    if df.isnull().sum().sum() > 0:
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype == "object":
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna(df[col].median())
        print("Missing values imputed.")
    else:
        print("No missing values found.")

    # ---- Split into features (X) and target (y) ----
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # ---- Train/test split (stratified to preserve class balance) ----
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- Save the splits locally so the workflow can upload them as artifacts ----
    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print(f"\nTrain set: {Xtrain.shape[0]} rows | Test set: {Xtest.shape[0]} rows")
    print("Saved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")

if __name__ == "__main__":
    main()
