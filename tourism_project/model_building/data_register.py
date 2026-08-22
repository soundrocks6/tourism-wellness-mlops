# -----------------------------------------------------------------------------
# data_register.py
# Registers the dataset stored inside the GitHub repository.
# - Reads tourism.csv from the repository data folder
# - Validates that all expected columns are present
# - Prints a short summary of the dataset
# -----------------------------------------------------------------------------

import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

# Columns the dataset is expected to contain (as per the data dictionary)
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

def main():
    # Load the dataset from the repository (first column is a serial index)
    df = pd.read_csv(DATA_PATH, index_col=0)
    print(f"Dataset loaded successfully from {DATA_PATH}")

    # ---- Validation: check that every expected column is present ----
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Validation failed. Missing columns: {missing}")
    print("Validation passed: all expected columns are present.")

    # ---- Print a short summary of the registered dataset ----
    print("\n----- Dataset Summary -----")
    print(f"Shape                : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Duplicate rows       : {df.duplicated().sum()}")
    print(f"Missing values       : {int(df.isnull().sum().sum())}")
    print("\nTarget distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts(normalize=True).round(4))
    print("\nColumn data types:")
    print(df.dtypes)

if __name__ == "__main__":
    main()
