# rainfall_prediction_by_LSTM
This is a program to classify and predict 7 classes based on 7 ranges of rainfall in An Giang, Vietnam

In this project we seperate original data based on daily period into 7 ranges to classify them:

Table of label key

|Label Key  | Range of value |
| ------------- | ------------- |
|0          | 0              |
|1          |(0,0.3]         |
|2          |(0.3,3]         |
|3          |(3,8]           |
|4          |(8,25]          |
|5          |(25,50]         |
|6          |(50, +inf)      |


This project has some problems need to solve:
  - Imbalanced data
    - To improve imbalanced data we need to use oversampling techniques such as SMOTE, ADASYN, SVM SMOTE,...
    - In this poject we use SVM SMOTE to oversample data
      ```sh
      from imblearn.oversampling import SVMSMOTE
      from collections import Counter
      import numpy as np

      def svm_smote(X, Y, random_state):
        #minority class count is optional but we need class count to ensure dataset is not missing any classes
        minority_class_counts = [count for _, count in Counter(np.argmax(Y, axis=1)).items() if count <= 5]
        k_neighbors_val = min(5, min(minority_class_counts) - 1) if minority_class_counts else 1
        #SVM SMOTE processing
        svmsm = SVMSMOTE(random_state=random_state, k_neighbors=k_neighbors_val)
        X_res, Y_res = svmsm.fit_resample(X, Y)
    
        return X_res, Y_res
      ```
  - Overlapping data
  - Overfitting classification model

  - See my experiments [!click here](https://github.com/nguyen-tho/rainfall_prediction_by_LSTM/blob/main/Ordinal_Classification_with_SVMSMOTE_and_LSTM.ipynb)



