#import read_data as r
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np
import pandas as pd

def data_mapping(data):
    manual_mapping = {
        'Không mưa': 'Không mưa',
        'Mưa không đáng kể': 'Mưa không đáng kể',
        'Mưa nhỏ': 'Mưa nhỏ',
        'Mưa': 'Mưa',
        'Mưa vừa': 'Mưa vừa',
        'Mưa to': 'Mưa to',
        'Mưa rất to': 'Mưa rất to',
    }
    data['rain_status'] = data['rain_status'].replace(manual_mapping)
    return data

def feature_and_target(data):
    features = data.drop(['rain_status'], axis=1)
    no_nan_data=data.replace(np.nan, 0)
    target = no_nan_data.rain_status
    features = pd.DataFrame(features)
    target = pd.DataFrame(target)
    return features, target

def data_normalization(X):
    scaler = MinMaxScaler(feature_range=(0,1))
    X = scaler.fit_transform(X)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    return X

def ordinal_encoding(y, categories):
    # Modified to handle unknown categories
    encoder = OrdinalEncoder(
        categories=categories
    )
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))
    return y_encoded

def target_encoding(y):
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))
    #y_encoded = y_encoded.toarray()
    return y_encoded


