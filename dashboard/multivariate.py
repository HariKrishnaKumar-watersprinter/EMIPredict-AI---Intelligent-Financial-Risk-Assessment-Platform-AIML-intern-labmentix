import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.feature_engineering import create_features
import numpy as np

def multi(df):
   
        
    # FIX: Removed st.columns() since you only need one full-width column.
    # If you wanted columns, it would be st.columns(2), but here it's unnecessary.
        
    st.subheader("Correlation Heatmap")
    
    numeric_df = df.select_dtypes(include=[np.number]).drop('loan_amount_bin', axis=1, errors='ignore')
    corr_matrix = numeric_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    corr_matrix_masked = corr_matrix.copy()
    corr_matrix_masked[mask] = None
    fig15 = px.imshow(corr_matrix_masked,text_auto=True,aspect="auto", title='Correlation Heatmap', color_continuous_scale='RdBu_r')
    fig15.update_layout(template='plotly_white', height=600, width=600)
    
    # FIX: Changed width='stretch' to use_container_width=True
    st.plotly_chart(fig15, width='stretch')
    
    st.subheader("3D Scatter: Income vs Credit Score vs Max EMI")
    eligible_df = df[df['emi_eligibility'] == 'Eligible']
    
    # FIX: Prevent crash if there are less than 1000 eligible rows
    sample_size = min(1000, len(eligible_df))
    if sample_size > 0:
        eligible_df_sample = eligible_df.sample(n=sample_size, random_state=42)
        fig = px.scatter_3d(eligible_df_sample, x='monthly_salary', y='credit_score', 
                           z='max_monthly_emi', color='employment_type', opacity=0.6)
        fig.update_layout(template='plotly_white', height=600)
        st.plotly_chart(fig, width='stretch')
    else:
        st.warning("No eligible data to display for 3D scatter.")
        
    st.subheader("Radar Chart: Customer Profile")
    
    # FIX: Removed 'Assets' from categories to match the 4 features you are calculating
    categories = ['Income', 'Credit Score', 'Emp Years', 'Loan Tenure']
    
    eligible_profile = df[df['emi_eligibility']=='Eligible'][['monthly_salary', 'credit_score', 
                                                              'years_of_employment', 
                                                              'requested_tenure']].mean()
    not_eligible_profile = df[df['emi_eligibility']=='Not_Eligible'][['monthly_salary', 'credit_score', 
                                                                   'years_of_employment', 
                                                                   'requested_tenure']].mean()
    max_vals = df[['monthly_salary', 'credit_score', 'years_of_employment', 
                   'requested_tenure']].max()
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=(eligible_profile/max_vals*100).tolist() + [(eligible_profile/max_vals*100).tolist()[0]],
        theta=categories + [categories[0]],
        fill='toself', name='Eligible', line_color='#00CC96'
    ))
    fig.add_trace(go.Scatterpolar(
        r=(not_eligible_profile/max_vals*100).tolist() + [(not_eligible_profile/max_vals*100).tolist()[0]],
        theta=categories + [categories[0]],
        fill='toself', name='Not Eligible', line_color='#EF553B'
    ))
    fig.update_layout(template='plotly_white', polar=dict(radialaxis=dict(range=[0, 100])))
    st.plotly_chart(fig, width='stretch')
