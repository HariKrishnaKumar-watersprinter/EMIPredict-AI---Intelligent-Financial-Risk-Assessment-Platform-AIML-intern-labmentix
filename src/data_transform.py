from sklearn.preprocessing import  LabelEncoder
from src.feature_engineering import create_features
from src.feature_selection import feature_selection
import numpy as np
import pandas as pd
def data_trans(df):
    df=create_features(df)
    df_selected=feature_selection(df)
    le = preprocessing.LabelEncoder()
    df['emi_eligibility']=le.fit_transform(df['emi_eligibility'])
 #0-Not eligible
    #1-eligible
    #2-high risk
    df_selected=pd.get_dummies(df_selected,drop_first=True,dtype=int)
    return df,df_selected