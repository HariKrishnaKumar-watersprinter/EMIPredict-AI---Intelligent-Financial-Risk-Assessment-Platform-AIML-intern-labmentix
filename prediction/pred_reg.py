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
def get_db():   
    db_root = EMIMongoDBManager()
    db_f2 = EMIMongoDBManager(folder=2)
    return db_f2
db = get_db()

def pred_r():  
    st.header("🔍 EMI Amount Prediction ")
    
    uploaded_file = st.file_uploader("Upload your csv file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        numeric_cols_to_clean = [
        'monthly_salary', 'dependents', 'monthly_rent', 'school_fees', 
        'college_fees', 'travel_expenses', 'groceries_utilities', 
        'other_monthly_expenses', 'current_emi_amount', 'credit_score','bank_balance',
        'max_monthly_emi', 'requested_amount', 'requested_tenure', 'age']
        
        for col in numeric_cols_to_clean:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(np.int32)
        # FIX: Clean the 'age' column in the original dataframe immediately 
        # so Streamlit and MongoDB don't crash on mixed object types (e.g., '58' str vs 58 int)
        if 'age' in data.columns:
            data['age'] = pd.to_numeric(data['age'], errors='coerce').fillna(0).astype(np.int32)
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(0)
         # Replace infinities with NaN
        st.markdown("### 📈 Uploaded Data")
        st.write(data.head())
        

        
        data1=data.copy()
        data1 = data_wrang(data1)
        data1 = create_features(data1)

        if 'emi_eligibility' in data1.columns:
            data1 = data1.drop('emi_eligibility', axis=1)
        
       

         # FIX 1: Handle Categoricals, replace infinities, and fill NaNs
        # 1. Convert categorical columns to string/object so fillna(0) works
        categorical_cols = data1.select_dtypes(['category']).columns
        data1[categorical_cols] = data1[categorical_cols].astype(str)
        
        # 2. Replace infinities with NaN
        data1 = data1.replace([np.inf, -np.inf], np.nan)
        
        # 3. Fill all remaining NaNs with 0
        data1 = data1.fillna(0)
        # FIX 2: CRITICAL! Keep the original 'data' dataframe in sync with 'data1'
        # by selecting only the indices that survived the dropna() above
        #data = data.loc[data1.index]

        # Apply get_dummies
        data1 = pd.get_dummies(data1, drop_first=True, dtype=int)

        # FIX 3: Reindex BEFORE transforming with the pipeline
        model_path = os.path.join(os.getcwd(), 'model', 'rf_emi_pred.pkl')
        if os.path.exists(model_path):
             model = joblib.load(model_path)

        possible_cols = list(model.feature_names_in_)
        data1 = data1.reindex(columns=possible_cols)
        data1 = data1.fillna(0) # Fill missing dummy columns with 0
        scaler_path = os.path.join( os.getcwd(), "scaler_reg.pkl")
        scaler   = joblib.load(scaler_path)
        # FIX 4: Load pipeline and use .transform() ONLY. 
        # Never use .fit_transform() on inference data!
        
        single = scaler.transform(data1)

        if st.button("🔮 Predict EMI Amount", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                st.markdown("## 📈 EMI Amount Prediction")
                predict = model.predict(single)
                # Because we synced 'data' with 'data1' above, the lengths now match perfectly!
                data['Predicted_EMI_Amount'] = pd.Series(predict)
            data_to_save = data.copy()
            for col in data_to_save.columns:
                if data_to_save[col].dtype == 'object':
                    data_to_save[col] = data_to_save[col].astype(str)

            st.dataframe(data)
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Prediction History (CSV)",
                data=csv,
                file_name="efficiency_prediction_history.csv",
                mime="text/csv",
                help="Click to download all saved prediction records as a CSV file."
            )
            try:
                records = json.loads(data_to_save.to_json(orient='records', date_format='iso'))
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
