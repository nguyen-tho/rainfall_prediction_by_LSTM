from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SVMSMOTE
from collections import Counter
import numpy as np
#smote is not use for this data because it makes overlapping and overfitting
def smote(X, Y, random_state): 
    minority_class_counts = [count for _, count in Counter(np.argmax(Y, axis=1)).items() if count <= 5]
    k_neighbors_val = min(5, min(minority_class_counts) - 1) if minority_class_counts else 1
    sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors_val)
    X_res, Y_res = sm.fit_resample(X, Y)
    
    return X_res, Y_res

def adasyn(X, Y, random_state):
    minority_class_counts = [count for _, count in Counter(np.argmax(Y, axis=1)).items() if count <= 5]
    k_neighbors_val = min(5, min(minority_class_counts) - 1) if minority_class_counts else 1
    ada = ADASYN(random_state=random_state, k_neighbors=k_neighbors_val)
    X_res, Y_res = smote.fit_resample(X, Y)
    
    return X_res, Y_res

def svm_smote(X, Y, random_state):
    minority_class_counts = [count for _, count in Counter(np.argmax(Y, axis=1)).items() if count <= 5]
    k_neighbors_val = min(5, min(minority_class_counts) - 1) if minority_class_counts else 1
    svmsm = SMOTE(random_state=random_state, k_neighbors=k_neighbors_val)
    X_res, Y_res = svmsm.fit_resample(X, Y)
    
    return X_res, Y_res


    