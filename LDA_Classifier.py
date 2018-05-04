# -*- coding: utf-8 -*-
"""
Created on Thu May  3 20:00:06 2018

@author: Febrian Adhi Pratama
"""

from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.linear_model import LinearRegression

# Step 1 - Load Iris dataset
iris = datasets.load_iris()
X = np.array(iris.data)
y = np.array(iris.target)

encoder = LabelEncoder()
label_encoder = encoder.fit(y)
y = label_encoder.transform(y) + 1

label_dict = {1: 'Setosa', 2: 'Versicolor', 3:'Virginica'}

# Step 2 - Use multiple linear regression to predict the class
regressor = LinearRegression()
regressor.fit(X, y)
y_hat = np.array(regressor.predict(X))

# Step 3 - Find the mean of y_hat values of every class
mean_setosa = np.mean(y_hat[:50])
mean_versicolor = np.mean(y_hat[50:100])
mean_virginica = np.mean(y_hat[100:])

# Step 4 - Find the covariance of every two classes
covariance_12 = (np.mean(y_hat[:50])*50+np.mean(y_hat[50:100])*50)/100
covariance_23 = (np.mean(y_hat[50:100])*50+np.mean(y_hat[100:])*50)/100

# Step 5 - Make LDA prediction
y_prediction = []
for i in range(len(y_hat)):
    if y_hat[i] >= covariance_23:
        y_prediction.append(3)
    elif y_hat[i] >= covariance_12:
        y_prediction.append(2)
    else:
        y_prediction.append(1)
        
# Step 6 - Calculate the accuracy and error
correct_prediction = 0
for i in range(len(y)):
    if y[i] == y_prediction[i]:
        correct_prediction += 1

accuracy = correct_prediction/len(y)
error = (len(y)-correct_prediction)/len(y)