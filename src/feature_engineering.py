import pandas as pd
from src.data_loader import load_data
from src.data_wrangling import data_wrang
import numpy as np

def create_features(df):
    
    """Create new features and preprocess data"""
    
    print("\n--- Creating New Features ---")
    
    # FIX: Ensure critical numeric columns are actually numeric and fill NaNs
    # coerce turns strings/unparseable data into NaN, fillna(0) replaces NaN with 0
    numeric_cols_to_clean = [
        'monthly_salary', 'dependents', 'monthly_rent', 'school_fees', 
        'college_fees', 'travel_expenses', 'groceries_utilities', 
        'other_monthly_expenses', 'current_emi_amount', 'credit_score', 
        'max_monthly_emi', 'requested_amount', 'requested_tenure', 'age'
    ]
    
    for col in numeric_cols_to_clean:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
           
    # Income-based features
    # dependents is now guaranteed to be numeric and not NaN, so +1 will work
    df['income_per_dependent'] = (df['monthly_salary'] / (df['dependents'] + 1)).astype(int)
    print("  - income_per_dependent: Monthly income adjusted for dependents")
    
    df['monthly_expenses'] = df['monthly_rent'] + df['school_fees'] + df['college_fees'] + df['travel_expenses'] + df['groceries_utilities'] + df['other_monthly_expenses']
    print("  - monthly_expenses: Sum of monthly expenses")
    
    df['savings_potential'] = df['monthly_salary'] - df['monthly_expenses'] - df['current_emi_amount']
    df['savings_potential'] = df['savings_potential'].clip(lower=0)
    print("  - savings_potential: Remaining income after expenses and EMIs")
    
    # Credit-based features
    df['credit_score_category'] = pd.cut(df['credit_score'], 
                                          bins=[0, 550, 650, 750, 850],
                                          labels=['Poor', 'Fair', 'Good', 'Excellent'],
                                          include_lowest=True) # Added include_lowest to catch credit scores of exactly 0
    print("  - credit_score_category: Categorized credit score")
    df['debt_to_income']=(df['max_monthly_emi'] / df['monthly_salary']*100).round(2)
    # Financial health score
    # Replace inf with NaN, then NaN with 0, to avoid infinite values in debt_to_income
    df['financial_health'] = (
        (df['credit_score'] / 850) * 30 +
        (df['savings_potential'] / df['monthly_salary']).clip(0, 1) * 20 
    ).round(2)
    # Clean up any potential infinite values from the financial_health calculation
    df['financial_health'] = df['financial_health'].replace([np.inf, -np.inf], np.nan).fillna(0)
    print("  - financial_health: Composite financial health score (0-100)")
    
    # Loan affordability
    df['emi_to_income_ratio'] = ((df['requested_amount'] / df['requested_tenure']) / df['monthly_salary']).round(2)
    df['emi_to_income_ratio'] = df['emi_to_income_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
    print("  - emi_to_income_ratio: Calculated EMI as ratio of income")
    
    # Age groups
    df['age_group'] = pd.cut(df['age'], bins=[20, 30, 40, 50, 65], 
                              labels=['Young', 'Middle-Age', 'Senior', 'Late-Career'])
    print("  - age_group: Categorized age groups")
    
    print(f"\nNew features created. Total columns: {df.shape[1]}")
    
    # FIX: Removed 'debt_to_income' from the print list because it is not defined in this function.
    print(f"\nNew columns: {['income_per_dependent', 'savings_potential', 'credit_score_category', 'financial_health', 'emi_to_income_ratio', 'age_group']}")
    
    return df
