# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 12:07:50 2018

@author: 727853
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dateutil.parser import parse

"""
date = parse("9/1/2018  05:07:00 PM")
print("Day: ", date.day)
print("Month: ", date.month)
print("Year: ", date.year)
print("Hour: ", date.hour)
print("Minute: ", date.minute)
print("Seconds: ", date.second)
"""

# Importing the dataset
dataset = pd.read_csv('SepFootFallODC1.csv')
dataset['day'] =  dataset.apply(lambda row: (parse(row['date'])).day, axis=1)
dataset['month'] =  dataset.apply(lambda row: (parse(row['date'])).month, axis=1)
dataset['year'] =  dataset.apply(lambda row: (parse(row['date'])).year, axis=1)
dataset['hour'] =  dataset.apply(lambda row: (parse(row['date'])).hour, axis=1)
dataset['minute'] =  dataset.apply(lambda row: (parse(row['date'])).minute, axis=1)
dataset['seconds'] =  dataset.apply(lambda row: (parse(row['date'])).second, axis=1)

X = dataset.iloc[:, 5:11].values
y = dataset.iloc[:, 2:3].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

sy = StandardScaler()
y_train = sy.fit_transform(y_train)
y_test = sy.transform(y_test)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'sigmoid', input_dim = 6))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'sigmoid'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'linear'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

xx = np.matrix([13,11,2018,15,45,0])

pred = classifier.predict(xx)