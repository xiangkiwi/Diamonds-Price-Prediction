# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:56:29 2022

@author: xiangkiwi
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

#Read the csv file
data = pd.read_csv(r"C:\Users\USER\Desktop\Python\Deep Learning\Datasets\diamonds.csv")
#Check the data type of each columns
print(data.info())
#To show whether there is any missing value in dataset
print(data.isna().sum())
#Perform one hot encoding on column with string object
data = pd.get_dummies(data)

#%%
#Drop the unnecessary columns
data = data.drop('Unnamed: 0', axis = 1)

#Set the labels and features
label = data['price']
features = data.drop('price', axis=1)

#%%
#Run train test split
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing

SEED = 12345

x_train, x_test, y_train, y_test = train_test_split(features, label, test_size = 0.2, random_state = SEED)
x_train = np.array(x_train)
x_test = np.array(x_test)

#%%
#Standardize data
standardizer = preprocessing.StandardScaler()
standardizer.fit(x_train)
x_train = standardizer.transform(x_train)
x_test = standardizer.transform(x_test)

#Data preparation is done
#%%
inputs = tf.keras.Input(shape=(x_train.shape[-1],))
dense = tf.keras.layers.Dense(128,activation = 'relu')
x = dense(inputs)
dense = tf.keras.layers.Dense(64,activation = 'relu')
x = dense(x)
dense = tf.keras.layers.Dense(32,activation = 'relu')
x = dense(x)
dense = tf.keras.layers.Dense(16,activation = 'relu')
x = dense(x)
dense = tf.keras.layers.Dense(16,activation = 'relu')
x = dense(x)
outputs = tf.keras.layers.Dense(1,activation = 'relu')(x)


model = tf.keras.Model(inputs = inputs,outputs = outputs,name='diamonds_model')
model.summary()

#%%
model.compile(optimizer = 'adam',loss = 'mse',metrics = ['mse','mae'])
history = model.fit(x_train, y_train, validation_data = (x_test, y_test),batch_size = 32,epochs = 30)

#%%
#Show the values for MSE and MAE
print(f"MSE = {np.mean(history.history['mse'])}")
print(f"MAE = {np.mean(history.history['mae'])}")


#%%
#Plot the plot for Actual vs Prediction
y_pred = model.predict(x_test)
y_pred = np.array(y_pred)
y_test = np.array(y_test)

#Pre-setting for the plot
fig, axis = plt.subplots(figsize = (10, 10))
axis.scatter(y_pred, y_test, facecolor = 'blue', s = 10)
axis.set_title('Actual vs Prediction')
axis.set_xlabel('Prediction')
axis.set_ylabel('Actual')

plt.show()