
#read dataset

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(filepath, nrows=None):
    dataset = pd.read_csv(filepath, engine='python', nrows=nrows)
    dataset = dataset.drop(["Data", "Time"], axis=1)
    
    target_column = "Radiation"
    scalar_dim = dataset["Temperature"].values.reshape(-1, 1)

    scaler_all = MinMaxScaler(feature_range=(0, 1))
    scaler_dim = MinMaxScaler(feature_range=(0, 1))

    dataset_scaled = scaler_all.fit_transform(dataset.astype('float32'))
    scalar_dim_scaled = scaler_dim.fit_transform(scalar_dim)

    return dataset_scaled, scalar_dim_scaled, scaler_dim, scaler_all
