# rainfall_prediction_by_LSTM
This is a program to classify and predict 7 classes based on 7 ranges of rainfall in An Giang, Vietnam

In this project we seperate original data based on daily period into 7 ranges to classify them:

Table of label key

|Label Key  | Range of value |
|----------------------------|
|0          | 0              |
|1          |(0,0.3]         |
|2          |(0.3,3]         |
|3          |(3,8]           |
|4          |(8,25]          |
|5          |(25,50]         |
|6          |(50, +inf)      |


This project has some problems need to solve:
  - Imbalanced data
  - Overlapping data
  - Overfitting classification model



