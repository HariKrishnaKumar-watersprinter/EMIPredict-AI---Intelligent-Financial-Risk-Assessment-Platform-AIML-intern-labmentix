import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.data_wrangling import data_wrang

def uni(df):
    
    
    col1, col2 = st.columns(2)
        
    with col1:
            st.subheader("Age Distribution")
            fig = px.histogram(df, x='age', nbins=30, marginal='box',
                              color_discrete_sequence=['#636EFA'])
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, width='stretch')
            
            st.subheader("Credit Score Distribution")
            fig = px.histogram(df, x='credit_score', nbins=40, marginal='box',
                              color_discrete_sequence=['#00CC96'])
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, width='stretch')
            
            st.subheader("Loan Amount Distribution")
            fig = px.box(df, y='requested_amount', color_discrete_sequence=['#FFA15A'])
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, width='stretch')
        
    with col2:
            st.subheader("Monthly Income Distribution")
            fig = px.histogram(df, x='monthly_salary', nbins=50, marginal='violin',
                              color_discrete_sequence=['#EF553B'])
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, width='stretch')
            
            st.subheader("Employment Type Distribution")
            emp_counts = df['employment_type'].value_counts()
            fig = px.pie(values=emp_counts.values, names=emp_counts.index,
                        hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, width='stretch')
            
            st.subheader("Education Level Distribution")
            edu_counts = df['education'].value_counts().sort_values(ascending=True)
            fig = px.bar(x=edu_counts.values, y=edu_counts.index, orientation='h',
                        color=edu_counts.values, color_continuous_scale='Viridis')
            fig.update_layout(template='plotly_white', height=400, showlegend=False)
            st.plotly_chart(fig, width='stretch')