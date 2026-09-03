import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.data_wrangling import data_wrang
from src.feature_engineering import create_features

def biv(df):
    #df = data_wrang()
    
    col1, col2 = st.columns(2)
        
    with col1:
            st.subheader("Income vs EMI Eligibility")
            fig = px.box(df, x='emi_eligibility', y='monthly_salary',
                        color='emi_eligibility',
                        color_discrete_map={0: '#EF553B', 1: '#00CC96'})
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, width='stretch')
            
            st.subheader("Income vs Max EMI Amount")
            eligible_df = df[df['emi_eligibility'] == 'Eligible']
            fig = px.scatter(eligible_df, x='monthly_salary', y='max_monthly_emi',
                            color='credit_score', color_continuous_scale='RdYlGn',
                            opacity=0.6, trendline='ols')
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, width='stretch')
            
            st.subheader("DTI Ratio vs Eligibility")
            df1=df.copy()
            df1['debt_to_income']=(df['max_monthly_emi'] / df['monthly_salary']*100).round(2)
            fig = px.histogram(df1, x='debt_to_income', color='emi_eligibility',
                              color_discrete_map={0: '#EF553B', 1: '#00CC96'},
                              nbins=50, barmode='overlay', opacity=0.7)
            fig.add_vline(x=0.4, line_dash='dash', line_color='black')
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, width='stretch')
        
    with col2:
            st.subheader("Credit Score vs EMI Eligibility")
            fig = px.box(df, x='emi_eligibility', y='credit_score',
                        color='emi_eligibility',
                        color_discrete_map={0: '#EF553B', 1: '#00CC96'})
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, width='stretch')
            st.subheader("Age vs Credit Score by Eligibility")
            fig = px.scatter(df, x='age', y='credit_score', color='emi_eligibility',
                            color_discrete_map={0: '#EF553B', 1: '#00CC96'}, opacity=0.5)
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, width='stretch')
            
            st.subheader("Employment Years vs Income")
            fig = px.scatter(df, x='years_of_employment', y='monthly_salary',
                            color='employment_type', opacity=0.5)
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, width='stretch')
