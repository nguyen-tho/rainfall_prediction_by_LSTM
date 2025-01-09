import pandas as pd
import numpy as np

def read_data(data_path, drop_col=None, encoding='utf-8', na_value='-'):
    # Read data from file and define NaN value
    data = pd.read_csv(data_path,na_values=na_value, encoding=encoding)
    # Drop columns if specified
    if drop_col:
        data.drop(columns=drop_col, inplace=True)
    return data

def set_date_as_index(data, format='%d/%M/%Y'): #dd/MM/YYYY is default format
    # Convert date column to datetime and set as index
    data['date'] = pd.to_datetime(data['date'], format=format)
    data.set_index('date', inplace=True)
    return data
    