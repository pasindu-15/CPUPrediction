# svr_regression.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('/Users/pasindu/Documents/Msc/ResearchProject/project/CPUPrediction/data-set/metrics.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y),1)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

# Training the SVR model on the Training set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)).reshape(-1,1))
np.set_printoptions(precision=5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
from sklearn.metrics import r2_score
print("performance :", r2_score(y_test, y_pred))

# Predict a new value
def get_predicted_cpu(parameters):
    y1_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform([parameters])).reshape(-1, 1))
    print("predicted cpu : ",y1_pred)
    return y1_pred

# get_predicted_cpu([3230.0, 0.010447052321981423, 137.0, 129.0])
