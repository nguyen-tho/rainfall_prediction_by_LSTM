from sklearn.model_selection import train_test_split

def split_data(X, Y, split_ratio, shuffle=True, random_state=42):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=split_ratio,
                                                        shuffle=shuffle,
                                                        random_state=random_state)