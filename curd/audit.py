import streamlit as st
import pandas as pd
import numpy as np

from curd.mongo_curd1 import EMIMongoDBManager, OperationError
from curd.config import DATASET_COLUMNS, CATEGORICAL_FIELDS, STATUS_VALUES

def aud(db):
    st.header("📋 Audit Trail")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        filter_id = st.number_input("Applicant ID", 0, 999999, 0)
        op_filter = st.selectbox("Operation", ["All", "CREATE", "UPDATE", "DELETE", "BULK_STATUS_UPDATE"])
        limit = st.selectbox("Entries", [50, 100, 200])
    
    with col2:
        logs = db.get_audit_log(limit=limit, **({"applicant_id": filter_id} if filter_id > 0 else {}))
        if op_filter != "All":
            logs = [l for l in logs if l.get('operation') == op_filter]
        
        if logs:
            df = pd.DataFrame(logs)[['timestamp', 'applicant_id', 'operation', 'performed_by']]
            st.dataframe(df, use_container_width=True, height=500)
        else:
            st.info("No logs")