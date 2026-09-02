import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dashboard.univariate import uni
from dashboard.bivariate import biv
from dashboard.multivariate import multi

#from dashboard.corr_heat import corr_heat
#from dashboard.anomallydetection import fraud_det
from src.data_loader import load_data as few_data
from src.feature_engineering import create_features
from src.data_wrangling import data_wrang
from curd.curd_st import curd_op
from curd.audit import aud
from curd.Export import IMP_exp
from prediction.predict import pred
from prediction.pred_reg import pred_r
import os
import json
from datetime import datetime
from curd.mongo_curd1 import EMIMongoDBManager, OperationError
from curd.config import DATASET_COLUMNS, CATEGORICAL_FIELDS, STATUS_VALUES
# ==============================================================================
# 3. SIDEBAR NAVIGATION
# ==============================================================================

st.set_page_config(
    page_title="EMIPredict AI - Financial Risk Assessment",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded")
# Industry-Grade CSS Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)
def get_db():
    return EMIMongoDBManager()
db = get_db()
with st.sidebar:
    st.markdown("### ⚙️ Configuration Panel")
    uploaded_file = st.file_uploader("Import Market Data (CSV)", type=['csv'], label_visibility="collapsed")
    options = ["📊 EDA Dashboard", '🔍 EMI Eligibility Prediction', '💵 EMI Amount Prediction',"📝 CRUD", 
               "📋 Audit", "📤 Import/Export"]
    selection = st.sidebar.radio("Select Business Case:", options)
#selection = st.sidebar.radio
st.sidebar.title("📋 Navigation")
st.sidebar.markdown("---")

try:
    health = db.health_check()
    st.sidebar.markdown(f"### 🟢 Database: {health['status']}")
    st.sidebar.metric("Records", health['applicant_count'])
    st.sidebar.metric("Size", f"{health['data_size_mb']} MB")
except:
    st.sidebar.error("❌ DB Connection Failed")

st.sidebar.markdown("---")

try:
    stats = db.get_statistics()
    st.sidebar.metric("Eligibility Rate", f"{stats['eligibility_rate']}%")
    if stats.get('averages'):
        st.sidebar.metric("Avg Salary", f"₹{stats['averages']['avg_salary']:,.0f}")
        st.sidebar.metric("Avg FOIR", f"{stats['averages']['avg_foir']}%")
except:
    pass

@st.cache_resource

def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = create_features(df)
    else:
        df=few_data()
        df = create_features(df)
    return df
df = load_data(uploaded_file)
st.markdown("""
<div class="main-header">
    <h1>💰 EMIPredict AI</h1>
    <h3>Intelligent Financial Risk Assessment Platform</h3>
</div>
""", unsafe_allow_html=True)

# Key Metrics


st.sidebar.markdown("---")
if selection == "📊 EDA Dashboard" and not df.empty:
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        eligible_count = (df['emi_eligibility'] == 'Eligible').sum()
        eligible_percentage = (eligible_count / len(df['emi_eligibility'])) * 100
        st.metric("Eligible %", f"{eligible_percentage:.1f}%")
    with col3:
        st.metric("Avg Income", f"₹{df['monthly_salary'].mean()/1000:.0f}K")
    with col4:
        st.metric("Avg Credit", f"{df['credit_score'].mean():.0f}")
    #st.title("📈 Tesla Stock Price Analytics and Deep Learning Predictor ")
    #st.write("""
    #This dashboard presents findings from the PhonePe Pulse GitHub data analysis. 
    #It focuses on 5 key business cases: Transaction Growth, Device Market Share, 
    #Insurance Opportunities, Geographical Expansion, and User Registration patterns.
    #""")
    #st.markdown("### End-to-End Analysis: ETL -> SQL -> ML -> Strategy")
    # Key Metrics Row
    st.header("📊 Exploratory Data Analysis Dashboard")
    
    # Row for chart selection
    chart_type = st.selectbox("Select Analysis Type", 
                               ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])
    if chart_type == "Univariate Analysis":
        uni(df)
    elif chart_type == "Bivariate Analysis":
        biv(df)
    else:
        chart_type == "Multivariate Analysis"
        multi(df)
    
elif selection  == '🔍 EMI Eligibility Prediction':
    pred()
elif selection == "💵 EMI Amount Prediction":
    pred_r()
elif selection == "📝 CRUD":
    curd_op(db)
elif selection == "📋 Audit":
    aud(db)
elif selection == "📤 Import/Export":
    IMP_exp(db)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>EMIPredict AI | 🗄️ MongoDB | 27 Features | CRUD + ML</p>
</div>
""", unsafe_allow_html=True)