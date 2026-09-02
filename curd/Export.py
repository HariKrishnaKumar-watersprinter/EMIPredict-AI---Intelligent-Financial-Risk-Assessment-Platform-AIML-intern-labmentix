import streamlit as st
import pandas as pd
import numpy as np

from curd.mongo_curd1 import EMIMongoDBManager, OperationError
from curd.config import DATASET_COLUMNS, CATEGORICAL_FIELDS, STATUS_VALUES

def IMP_exp(db):
    st.header("Import/Export")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Import from CSV")
        csv_file = st.file_uploader("Upload CSV", type=['csv'])
        if csv_file and st.button("📥 Import"):
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as f:
                f.write(csv_file.getvalue())
                temp_path = f.name
            
            result = db.import_from_csv(temp_path)
            st.success(f"✅ Imported {result['successful_count']} records")
            if result['failed_count'] > 0:
                st.warning(f"Failed: {result['failed_count']}")
            os.unlink(temp_path)
    
    with col2:
        st.subheader("Export Data")
        export_format = st.radio("Format", ["CSV", "JSON"])
        status_filter = st.selectbox("Filter", ["All"] + STATUS_VALUES)
        
        if st.button("📤 Export"):
            status = None if status_filter == "All" else status_filter
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if export_format == "CSV":
                path = f'data/export_{timestamp}.csv'
                count = db.export_to_csv(path, status_filter=status)
                mime = 'text/csv'
            else:
                path = f'data/export_{timestamp}.json'
                count = db.export_to_json(path, status_filter=status)
                mime = 'application/json'
            
            st.success(f"✅ Exported {count} records")
            
            with open(path, 'rb') as f:
                st.download_button(
                    label=f"⬇️ Download {export_format}",
                    data=f,
                    file_name=os.path.basename(path),
                    mime=mime
                )
    
    # Dataset info
    st.markdown("---")
    st.subheader("Dataset Schema (27 Columns)")
    
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        st.markdown("**Demographic**")
        for c in ["age", "gender", "marital_status", "education", "family_size", "dependents"]:
            st.markdown(f"- `{c}`")
    
    with col_b:
        st.markdown("**Employment & Housing**")
        for c in ["monthly_salary", "employment_type", "years_of_employment", "company_type", "house_type", "monthly_rent"]:
            st.markdown(f"- `{c}`")
    
    with col_c:
        st.markdown("**Expenses & Financial**")
        for c in ["school_fees", "college_fees", "travel_expenses", "groceries_utilities", "other_monthly_expenses", "existing_loans", "current_emi_amount", "credit_score", "bank_balance", "emergency_fund"]:
            st.markdown(f"- `{c}`")
    
    with col_d:
        st.markdown("**Loan & Targets**")
        for c in ["emi_scenario", "requested_amount", "requested_tenure", "emi_eligibility", "max_monthly_emi"]:
            st.markdown(f"- `{c}`")
        st.markdown("**Derived (Auto)**")
        for c in ["foir_percentage", "debt_to_income_ratio", "net_monthly_income", "total_monthly_expenses"]:
            st.markdown(f"- `{c}`")