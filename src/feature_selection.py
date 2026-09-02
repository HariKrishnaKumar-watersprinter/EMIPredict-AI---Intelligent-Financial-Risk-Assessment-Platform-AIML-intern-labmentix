import numpy as np
import pandas as pd
from src.feature_engineering import create_features

def feature_select(df):
    
    """Select important features and manipulate as needed"""
    
    print("\n--- Dropping Redundant/Low-Value Features ---")
    
    # Features to drop
    drop_features = ['monthly_salary', 'loan_to_income', 'emi_to_income_ratio']
    df = df.drop(columns=drop_features, errors='ignore')
    print(f"  Dropped features: {drop_features}")
    
    # Calculate VIF for multicollinearity check
    print("\n--- Variance Inflation Factor (VIF) Analysis ---")
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_features if col not in ['emi_eligibility', 'max_emi_amount']]
    
    # Ensure no NaN or infinite values
    vif_data = df[numeric_features].replace([np.inf, -np.inf], np.nan).dropna()
    
    vif_results = []
    for i, col in enumerate(numeric_features):
        if col in vif_data.columns:
            try:
                vif = variance_inflation_factor(vif_data.values, i)
                vif_results.append({'Feature': col, 'VIF': vif})
            except:
                pass
    
    vif_df = pd.DataFrame(vif_results).sort_values('VIF', ascending=False)
    print(vif_df.to_string(index=False))
    
    # Drop features with VIF > 10
    high_vif = vif_df[vif_df['VIF'] > 10]['Feature'].tolist()
    if high_vif:
        # Keep some important features even if high VIF
        keep_features = ['monthly_income', 'credit_score', 'debt_to_income']
        drop_vif = [f for f in high_vif if f not in keep_features]
        if drop_vif:
            df = df.drop(columns=drop_vif, errors='ignore')
            print(f"\n  Dropped high VIF features: {drop_vif}")
    
    print(f"\nFinal features: {df.shape[1]} columns")
    print(f"Feature list: {df.columns.tolist()}")
    df=df.drop(columns='emi_eligibility',errors='ignore')
    return df