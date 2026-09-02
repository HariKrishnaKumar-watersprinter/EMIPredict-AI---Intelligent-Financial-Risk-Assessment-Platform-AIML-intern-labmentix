import pandas as pd
import os
import numpy as np
def load_data():
    url='https://dagshub.com/harikrishnakumar368/EMIPredict-AI---Intelligent-Financial-Risk-Assessment-Platform-AIML-intern-labmentix/raw/main/s3:/EMIPredict-AI---Intelligent-Financial-Risk-Assessment-Platform-AIML-intern-labmentix/emi_prediction_dataset.csv'
    df = pd.read_csv(url)
    print(df)
    #df = df.to_pandas()
    return df 
