"""
EMIPredict AI - Configuration for Your Dataset Schema
"""

import os
from dotenv import load_dotenv
import streamlit as st
load_dotenv()


class MongoDBConfig:
    """MongoDB Configuration"""
    MONGO_URI = st.secrets["MONGO_URI"]
    DATABASE_NAME = st.secrets.get("MONGO_DB_NAME", "emipredict_ai")
    
    # Collection Names
    APPLICANTS_COLLECTION = "applicants"
    APPLICANTS_FOLDER_1 = "applicants.pred Emi Eligibility"
    APPLICANTS_FOLDER_2 = "applicants.Pred Emi Amount "
    APPLICANTS_FOLDER_3 = "applicants.Actual data"
    AUDIT_LOG_COLLECTION = "audit_logs"
    PREDICTION_HISTORY_COLLECTION = "prediction_history"
    
    # Connection Pool
    MAX_POOL_SIZE = int(st.secrets("MONGO_MAX_POOL_SIZE", 100))
    MIN_POOL_SIZE = int(st.secrets("MONGO_MIN_POOL_SIZE", 10))
    SERVER_SELECTION_TIMEOUT_MS = int(st.secrets("MONGO_TIMEOUT_MS", 5000))
    
    # TTL for audit logs (days)
    TTL_AUDIT_DAYS = int(st.secrets("AUDIT_TTL_DAYS", 90))


class AppSettings:
    """Application Settings"""
    APP_NAME = "EMIPredict AI"
    APP_VERSION = "2.0.0"
    ENVIRONMENT = st.secrets("ENVIRONMENT", "development")
    DEBUG = st.secrets("DEBUG", "True").lower() == "true"


# ============================================================================
# SCHEMA VALIDATION FOR YOUR DATASET COLUMNS
# ============================================================================

# Your exact 27 columns from CSV
DATASET_COLUMNS = [
    "age", "gender", "marital_status", "education", "monthly_salary",
    "employment_type", "years_of_employment", "company_type", "house_type",
    "monthly_rent", "family_size", "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses",
    "existing_loans", "current_emi_amount", "credit_score", "bank_balance",
    "emergency_fund", "emi_scenario", "requested_amount", "requested_tenure",
    "emi_eligibility", "max_monthly_emi",'Predicted_EMI_Eligibility',"Confidence",'Predicted_EMI_Amount'
]

# Target columns
TARGET_COLUMNS = ["emi_eligibility", "max_monthly_emi"]

# Feature columns (all except targets)
FEATURE_COLUMNS = [col for col in DATASET_COLUMNS if col not in TARGET_COLUMNS]

# Demographic columns
DEMOGRAPHIC_COLUMNS = ["age", "gender", "marital_status", "education", "family_size", "dependents"]

# Employment columns
EMPLOYMENT_COLUMNS = ["monthly_salary", "employment_type", "years_of_employment", "company_type"]

# Housing columns
HOUSING_COLUMNS = ["house_type", "monthly_rent"]

# Expense columns
EXPENSE_COLUMNS = [
    "school_fees", "college_fees", "travel_expenses", 
    "groceries_utilities", "other_monthly_expenses"
]

# Financial columns
FINANCIAL_COLUMNS = [
    "existing_loans", "current_emi_amount", "credit_score", 
    "bank_balance", "emergency_fund"
]

# Loan request columns
LOAN_COLUMNS = ["emi_scenario", "requested_amount", "requested_tenure"]

# Categorical columns with their allowed values
CATEGORICAL_FIELDS = {
    "gender": ["Male", "Female"],
    "marital_status": ["Single", "Married"],
    "education": ["High School", "Graduate","Post Graduate","Professional",'0'],
    "employment_type": ["Private", "Self-employed", "Government"],
    "company_type": ["MNC", "Mid-size", "Startup", "Large Indian",'Small'],
    "house_type": ["Own", "Rented", "Family"],
    "emi_scenario": ["Personal Loan EMI", "E-commerce Shopping EMI", "Education EMI",'Vehicle EMI','Home Appliances EMI'],
    "existing_loans": ["Yes", "No"],
    'emi_eligibility':["Not_Eligible","Eligible","High_Risk"],
    'Predicted_EMI_Eligibility':['❌ NOT ELIGIBLE FOR EMI','✅ EMI ELIGIBLE','🚨High_Risk']
    #"bank_balance": ["Yes", "No"],
    #"emergency_fund": ["Yes", "No"]
}

# Status values for application workflow
STATUS_VALUES = ["Pending", "Approved", "Rejected", "Under Review", "Cancelled"]

# Field validation rules
FIELD_VALIDATION = {
    "age": {"type": int, "min": 18, "max": 100, "required": True},
    "gender": {"type": str, "allowed": CATEGORICAL_FIELDS["gender"], "required": True},
    "marital_status": {"type": str, "allowed": CATEGORICAL_FIELDS["marital_status"], "required": True},
    "education": {"type": str, "allowed": CATEGORICAL_FIELDS["education"], "required": True},
    "monthly_salary": {"type": (int, float), "min": 0, "required": True},
    "employment_type": {"type": str, "allowed": CATEGORICAL_FIELDS["employment_type"], "required": True},
    "years_of_employment": {"type": (int, float), "min": 0, "max": 50, "required": False},
    "company_type": {"type": str, "allowed": CATEGORICAL_FIELDS["company_type"], "required": False},
    "house_type": {"type": str, "allowed": CATEGORICAL_FIELDS["house_type"], "required": False},
    "monthly_rent": {"type": (int, float), "min": 0, "required": False},
    "family_size": {"type": int, "min": 0, "max": 20, "required": False},
    "dependents": {"type": int, "min": 0, "max": 20, "required": False},
    "school_fees": {"type": (int, float), "min": 0, "required": False},
    "college_fees": {"type": (int, float), "min": 0, "required": False},
    "travel_expenses": {"type": (int, float), "min": 0, "required": False},
    "groceries_utilities": {"type": (int, float), "min": 0, "required": False},
    "other_monthly_expenses": {"type": (int, float), "min": 0, "required": False},
    "existing_loans": {"type": str, "allowed": CATEGORICAL_FIELDS["existing_loans"], "required": False},
    "current_emi_amount": {"type": (int, float), "min": 0, "required": False},
    "credit_score": {"type": int, "min": 0, "max": 1400, "required": True},
    "bank_balance": {"type": (int, float), "min": 0, "required": False},
    "emergency_fund": {"type": (int, float), "min": 0, "required": False},
    "emi_scenario": {"type": str, "allowed": CATEGORICAL_FIELDS["emi_scenario"], "required": False},
    "requested_amount": {"type": (int, float), "min": 0, "required": True},
    "requested_tenure": {"type": int, "min": 1, "max": 360, "required": True},
    "emi_eligibility": {"type": str, "allowed": CATEGORICAL_FIELDS["emi_eligibility"], "required": False},
    "max_monthly_emi": {"type": (int, float), "min": 0, "required": False},
    "Predicted_EMI_Eligibility": {"type": str, "allowed": CATEGORICAL_FIELDS["Predicted_EMI_Eligibility"], "required": False},
    "Confidence": {"type": (int, float), "min": 0, "max": 1, "required": False},
    'Predicted_EMI_Amount': {"type": (int, float), "min": 0, "required": False}}

# Required fields for creating a new applicant
REQUIRED_FIELDS = [
    "age", "gender", "marital_status", "education", "monthly_salary",
    "employment_type", "credit_score", "requested_amount", "requested_tenure"]
