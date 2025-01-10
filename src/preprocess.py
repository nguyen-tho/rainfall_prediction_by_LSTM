import read_data as r

data_path = '../data/vd.csv'
test_path = '../data/new_test_data.csv'

#read train and test set
data = r.read_data(data_path)
test_data = r.read_data(test_path)

#set datetime as index
data = r.set_date_as_index(data)
test_data = r.set_date_as_index(test_data)

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

data = data_mapping(data)
test_data = data_mapping(test_data)

def feature_and_target(data):
    #features
    features = data.drop(['rain_status'], axis=1)
    #target
    target = data.rain_status
    return features, target

feature, target = feature_and_target(data)
test_feature, test_target = feature_and_target(test_data)
    

    

