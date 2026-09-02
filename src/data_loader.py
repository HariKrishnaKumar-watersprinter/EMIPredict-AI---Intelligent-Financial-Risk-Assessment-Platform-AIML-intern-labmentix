import pandas as pd
import os
import numpy as np
def load_data():
    path = os.path.join(os.getcwd(), "data", "emi_prediction_dataset.csv")
    
    df = pd.read_csv(path)
    
    return df   