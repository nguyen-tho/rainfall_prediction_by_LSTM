import read_data as r
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np
"""
data_path = '../data/vd.csv'
test_path = '../data/new_test.csv'

#read train and test set
data = r.read_data(data_path)
test_data = r.read_data(test_path)

#set datetime as index
data = r.set_date_as_index(data)
test_data = r.set_date_as_index(test_data)
"""
def data_mapping(data):
    classes = {
        'Không mưa':0,
        'Mưa không đáng kể':1,
        'Mưa':2,
        'Mưa nhỏ':3,
        'Mưa vừa':4,
        'Mưa to':5,
        'Mưa rất to':6
        }
    manual_mapping = {
        'Mưa không đáng kể': 'Mưa không đáng kể',
        'Mưa nhỏ': 'Mưa nhỏ',
        'Mưa vừa': 'Mưa vừa',
        'Mưa rất to': 'Mưa rất to',
    # Add other mappings if necessary
    }
    data['rain_status'] = data['rain_status'].replace(manual_mapping)
    data['rain_status'] = data['rain_status'].replace(classes)
    
    return data

#data = data_mapping(data)
#test_data = data_mapping(test_data)

def feature_and_target(data):
    #features
    features = data.drop(['rain_status'], axis=1)
    #target
    target = data.rain_status
    return features, target

#feature, target = feature_and_target(data)
#test_feature, test_target = feature_and_target(test_data)
    
def data_normalization(X):
    #normalize data with min-max scaling for features X
    scaler = MinMaxScaler(feature_range=(0,1))
    X = scaler.fit_transform(X)
    X = X.reshape(X.shape[0], 1, X.shape[1]) #reshape to 3D array for LSTM 
    
    return X

def target_encoding(y):
    #one-hot encoding for target y
    encoder = OneHotEncoder()
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))
    y_encoded = y_encoded.toarray()
    
    return y

def over_sampling(X, y):
    #oversampling the minority class in y
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def over_sampling_for_valid_set(X_val, Y_val):
    # Calculate k_neighbors based on the minimum number of samples in each class in Y_val
    minority_class_counts = [count for _, count in Counter(np.argmax(Y_val, axis=1)).items() if count <= 5]
    k_neighbors_val = min(5, min(minority_class_counts) - 1) if minority_class_counts else 1  # Ensure k_neighbors_val is at least 1

    # Adjust k_neighbors for validation set based on the count of the rarest class
    smote_val = SMOTE(random_state=42, k_neighbors=k_neighbors_val)
    X_val, Y_val = smote_val.fit_resample(X_val, Y_val)
    
    return X_val, Y_val
    

