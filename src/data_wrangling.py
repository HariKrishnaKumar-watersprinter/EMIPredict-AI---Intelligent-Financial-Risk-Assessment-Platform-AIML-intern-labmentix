import pandas as pd
import numpy as np
from src.data_loader import load_data

def data_wrang(df):
    #df=load_data()
    """Perform data wrangling operations"""
    print("\n--- Missing Values Check ---")
    print(f"\nTotal Missing Values: {df.isnull().sum().sum()}")
    
    # Handle missing values
    print("\n--- Handling Missing Values ---")
    
    # Numeric columns - median imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  - {col}: Filled {df[col].isnull().sum()} missing values with median ({median_val:.2f})")
    
    # Categorical columns - mode imputation
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  - {col}: Filled missing values with mode ({mode_val})")
    int_cols = ['age','monthly_salary','years_of_employment', 'monthly_rent','family_size',
                'school_fees','college_fees','travel_expenses','groceries_utilities',
                'other_monthly_expenses','current_emi_amount','credit_score','dependents',
      
              'bank_balance','emergency_fund', 'requested_amount','requested_tenure','max_monthly_emi']
    int_cols = [col for col in int_cols if col in df.columns]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(0)
        df[col] = df[col].astype(int)
        print(f"  - {col}: Integer columns converted successfully")
    # Check and remove duplicates
    print(f"\n--- Duplicate Check ---")
    duplicates = df.duplicated().sum()
    print(f"Duplicate Rows: {duplicates}")
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"Duplicates removed. New shape: {df.shape}")
    
    # Check for negative values
    print("\n--- Negative Values Check ---")
    for col in numeric_cols:
        if (df[col] < 0).any():
            print(f"  Warning: {col} has negative values")
            df[col] = df[col].abs()
    print("Negative values handled")
    
    print(f"\nFinal Dataset Shape: {df.shape}")
    return df