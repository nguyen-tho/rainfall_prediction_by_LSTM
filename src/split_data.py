from sklearn.model_selection import train_test_split

def split_data(X, Y, split_ratio=0.2, shuffle=False, random_state=42):
    # split ratio is a float to seperate data into 2 parts train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=split_ratio,
                                                        shuffle=shuffle,
                                                        random_state=random_state)
    
    return X_train, X_test, Y_train, Y_test


def split_data_3_parts(X, Y, train_test_ratio=(0.6, 0.2), shuffle=False, random_state=42):
    # train_val_ratio is a tuple to seperate data into 3 parts train, val and test
    train_ratio, test_ratio = train_test_ratio
    X_train_val, X_test, Y_train_val, Y_test = split_data(X, Y, test_ratio)
    
    #The remaining part is validation set
    remaining = test_ratio/(train_ratio + test_ratio)
    X_train, X_val, Y_train, Y_val = split_data(X_train_val, Y_train_val, split_ratio=remaining)
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test
    

