import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import numpy as np
from sklearn.pipeline import Pipeline

def datascale():
    
    pipeline_steps = [
               ('imputer', SimpleImputer(strategy='mean')), # Safety for any NaNs
               ('scaler', RobustScaler())]
    preprocess_pipeline = Pipeline(pipeline_steps)
    preprocess_pipeline1= Pipeline(pipeline_steps)
    
    
    return preprocess_pipeline,preprocess_pipeline1