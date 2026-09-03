import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
import datetime
from src.data_wrangling import data_wrang
from src.feature_engineering import create_features 
from src.feature_selection import feature_select
from src.Data_split import datascale
from datetime import datetime
from curd.mongo_curd1 import EMIMongoDBManager, OperationError
from curd.config import DATASET_COLUMNS, CATEGORICAL_FIELDS, STATUS_VALUES
import json
import mlflow
import logging # Added logger since you used logger.warning

# Setup logger if not already configured
logger = logging.getLogger(__name__)

def get_db():
    db_root = EMIMongoDBManager()
    db_f1 = EMIMongoDBManager(folder=1)
    return db_f1

db = get_db()

def pred():  
    st.header("🔍 EMI Eligibility Prediction")
    
    uploaded_file = st.file_uploader("Upload your csv file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        numeric_cols_to_clean = [
            'monthly_salary', 'dependents', 'monthly_rent', 'school_fees', 
            'college_fees', 'travel_expenses', 'groceries_utilities', 
            'other_monthly_expenses', 'current_emi_amount', 'credit_score','bank_balance',
            'max_monthly_emi', 'requested_amount', 'requested_tenure', 'age'
        ]
        
        for col in numeric_cols_to_clean:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(np.int32)
        
        if 'age' in data.columns:
            data['age'] = pd.to_numeric(data['age'], errors='coerce').fillna(0).astype(np.int32)
            
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(0)
        
        st.markdown("### 📈 Uploaded Data")
        st.write(data.head())
        
        # --- PREDICTION BUTTON ---
        if st.button("🔮 Predict EMI Eligibility", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                data1 = data.copy()
                data1 = data_wrang(data1)
                data1 = create_features(data1)

                if 'emi_eligibility' in data1.columns:
                    data1 = data1.drop('emi_eligibility', axis=1)   
         
                # Handle Categoricals, replace infinities, and fill NaNs
                categorical_cols = data1.select_dtypes(['category']).columns
                data1[categorical_cols] = data1[categorical_cols].astype(str)
                
                data1 = data1.replace([np.inf, -np.inf], np.nan)
                data1 = data1.fillna(0)
                data1 = pd.get_dummies(data1, drop_first=True, dtype=int)

                # Load Scaler and Model
                scaler_path = os.path.join(os.getcwd(), "scaler_cl.pkl")
                scaler = joblib.load(scaler_path)
                mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
       
                artifact_uri = os.getenv("ARTIFACTS1")
                local_path = mlflow.artifacts.download_artifacts(artifact_uri)
                model = joblib.load(local_path)
                
                required_cols = list(model.feature_names_in_)
                data1 = data1.reindex(columns=required_cols, fill_value=0)
                data1 = data1.fillna(0)
                
                single = scaler.transform(data1)
                predict = model.predict(single)
                probability = model.predict_proba(single)
                label_map = {0: '❌ NOT ELIGIBLE FOR EMI', 1: '✅ EMI ELIGIBLE', 2: '🚨High_Risk'}
                
                # Add predictions to the original data
                data['Predicted_EMI_Eligibility'] = pd.Series(predict).map(label_map).values
                data['Confidence'] = probability.max(axis=1)

                # Ensure all types are Arrow-compatible before converting
                data_to_save = data.copy()
                for col in data_to_save.columns:
                    if data_to_save[col].dtype == 'object':
                        data_to_save[col] = data_to_save[col].astype(str)
                
                # ✅ SAVE TO SESSION STATE instead of local variables
                st.session_state.prediction_data = data
                st.session_state.data_to_save = data_to_save

        # --- DISPLAY RESULTS & BUTTONS (Outside the Predict button block) ---
        # Check if predictions exist in session state
        if "prediction_data" in st.session_state and "data_to_save" in st.session_state:
            
            st.markdown("### 🔮 EMI Eligibility Prediction Results")
            st.dataframe(st.session_state.prediction_data)
            
            # Download Button
            csv = st.session_state.data_to_save.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Prediction History (CSV)",
                data=csv,
                file_name="efficiency_prediction_history.csv",
                mime="text/csv",
                help="Click to download all saved prediction records as a CSV file."
            )
            
            # Save to DB Button
            if st.button("💾 Save to Database", type="primary", use_container_width=True):
                with st.spinner("Saving to Database..."):
                    try:
                        records = json.loads(st.session_state.data_to_save.to_json(orient='records', date_format='iso'))
                        saved_ids = []
                        failed_count = 0
                
                        for record in records:
                            try:
                                app_id = db.create_applicant(record)
                                saved_ids.append(app_id)
                            except Exception as row_e:
                                failed_count += 1
                                logger.warning(f"Failed to save row to MongoDB: {row_e}")
                        
                        if saved_ids:
                            st.success(f"💾 Saved {len(saved_ids)} records to MongoDB!")
                        if failed_count > 0:
                            st.warning(f"⚠️ Failed to save {failed_count} records due to validation errors.")
                    
                    except Exception as e:
                        st.error(f"DB save error: {e}")
                        
    else:
        st.info("☝️ Please upload a CSV file to proceed.")
