#!/usr/bin/env python3
# coding: utf-8

"""
Author:
    Wenhan Tan
Date:
    2021/9

Description:
    For both decision flow results and multi-class results, this script first
    calculates a 5-dimensional vector for each slide from 3 csv files (train,
    test, and validation). Then train a simple neural network for decision flow
    and another one for multi-class. Then the two trained neural networks are 
    applied on testing sets. Finally, the predicted labels and true labels are
    used to calcualte quadratic weighted kappa. The 3 csv files are generated
    from another python script (Save_slideFolders_classification_result.csv).

Input:
    Provided by this work:
        1) train_slideFolders_classification.csv (from script *Save_slideFolders_classification_result.py*)
        2) test_slideFolders_classification.csv (from script *Save_slideFolders_classification_result.py*)
        3) validation_slideFolders_classification.csv (from script *Save_slideFolders_classification_result.py*)

Output:
    To terminal:
        1) Quadratic weighted kappa value (decision flow)
        2) Quadratic weighted kappa value (multi-class)

Usage:
    To reproduce results, simply run this script in terminal. Make sure you
    have all the python packages and input files ready.

    To use it on a different dataset, go through all the input files and make
    sure you have them in the same format for your dataset. Change input
    filepath based on your file locations.
"""

import numpy as np
from tqdm import trange
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def quadratic_kappa(actuals, preds, N=4):
    """
    This function calculates the Quadratic Kappa Metric used for Evaluation in
    the PetFinder competition at Kaggle. It returns the Quadratic Weighted Kappa
    metric score between the actual and the predicted values of adoption rating.
    """

    w = np.zeros((N,N))
    O = confusion_matrix(actuals, preds)
    for i in range(len(w)): 
        for j in range(len(w)):
            w[i][j] = float(((i-j)**2)/(N-1)**2)
    
    act_hist=np.zeros([N])
    for item in actuals: 
        act_hist[item]+=1
    
    pred_hist=np.zeros([N])
    for item in preds: 
        pred_hist[item]+=1
                         
    E = np.outer(act_hist, pred_hist)
    E = E/E.sum()
    O = O/O.sum()
    
    num=0
    den=0
    for i in range(len(w)):
        for j in range(len(w)):
            num+=w[i][j]*O[i][j]
            den+=w[i][j]*E[i][j]
    return (1 - (num/den))

"""
Filepath
"""
train_slideFolders_classification_path = "./prostate-cancer-grade-assessment/train_slideFolders_classification.csv"
test_slideFolders_classification_path = "./prostate-cancer-grade-assessment/test_slideFolders_classification.csv"
validation_slideFolders_classification_path = "./prostate-cancer-grade-assessment/validation_slideFolders_classification.csv"

"""
Find row length
"""
train_row_length = len(np.genfromtxt(train_slideFolders_classification_path, delimiter=",", usecols=1))
test_row_length = len(np.genfromtxt(test_slideFolders_classification_path, delimiter=",", usecols=1))
validation_row_length = len(np.genfromtxt(validation_slideFolders_classification_path, delimiter=",", usecols=1))

"""
Read in train data
"""
print("Reaing in train data")
train_DF_data = []
train_DF_magnitude = []
train_MC_data = []
train_MC_magnitude = []
train_y = []

for i in trange(train_row_length):
    train_slideFolders_classification = np.genfromtxt(train_slideFolders_classification_path, delimiter=",", skip_header=i, max_rows=1, dtype=np.str)
    train_slideFolders_classification = train_slideFolders_classification[train_slideFolders_classification != '']
    
    # split data of each row into decision flow' result and probabilities and
    # multi-class's results and probabilities
    train_DF_data.append(train_slideFolders_classification[:int((len(train_slideFolders_classification) - 1)/2/2)].astype(float))
    train_DF_magnitude.append(train_slideFolders_classification[int((len(train_slideFolders_classification) - 1)/2/2):int((len(train_slideFolders_classification) - 1)/2)].astype(float))
    train_MC_data.append(train_slideFolders_classification[int((len(train_slideFolders_classification) - 1)/2):int((len(train_slideFolders_classification) - 1)/2 + (len(train_slideFolders_classification) - 1)/2/2)].astype(float))
    train_MC_magnitude.append(train_slideFolders_classification[int((len(train_slideFolders_classification) - 1)/2 + (len(train_slideFolders_classification) - 1)/2/2):len(train_slideFolders_classification)-1].astype(float))
    train_y.append(train_slideFolders_classification[-1].tolist())

"""
Read in test data
"""
print("Reading in test data")
test_DF_data = []
test_DF_magnitude = []
test_MC_data = []
test_MC_magnitude = []
test_y = []

for i in trange(test_row_length):
    test_slideFolders_classification = np.genfromtxt(test_slideFolders_classification_path, delimiter=",", skip_header=i, max_rows=1, dtype=np.str)
    test_slideFolders_classification = test_slideFolders_classification[test_slideFolders_classification != '']
    
    test_DF_data.append(test_slideFolders_classification[:int((len(test_slideFolders_classification) - 1)/2/2)].astype(float))
    test_DF_magnitude.append(test_slideFolders_classification[int((len(test_slideFolders_classification) - 1)/2/2):int((len(test_slideFolders_classification) - 1)/2)].astype(float))
    test_MC_data.append(test_slideFolders_classification[int((len(test_slideFolders_classification) - 1)/2):int((len(test_slideFolders_classification) - 1)/2 + (len(test_slideFolders_classification) - 1)/2/2)].astype(float))
    test_MC_magnitude.append(test_slideFolders_classification[int((len(test_slideFolders_classification) - 1)/2 + (len(test_slideFolders_classification) - 1)/2/2):len(test_slideFolders_classification)-1].astype(float))
    test_y.append(test_slideFolders_classification[-1].tolist())

"""
Read in validation data
"""
print("Reading in validation data")
validation_DF_data = []
validation_DF_magnitude = []
validation_MC_data = []
validation_MC_magnitude = []
validation_y = []

for i in trange(validation_row_length):
    validation_slideFolders_classification = np.genfromtxt(validation_slideFolders_classification_path, delimiter=",", skip_header=i, max_rows=1, dtype=np.str)
    validation_slideFolders_classification = validation_slideFolders_classification[validation_slideFolders_classification != '']
    
    validation_DF_data.append(validation_slideFolders_classification[:int((len(validation_slideFolders_classification) - 1)/2/2)].astype(float))
    validation_DF_magnitude.append(validation_slideFolders_classification[int((len(validation_slideFolders_classification) - 1)/2/2):int((len(validation_slideFolders_classification) - 1)/2)].astype(float))
    validation_MC_data.append(validation_slideFolders_classification[int((len(validation_slideFolders_classification) - 1)/2):int((len(validation_slideFolders_classification) - 1)/2 + (len(validation_slideFolders_classification) - 1)/2/2)].astype(float))
    validation_MC_magnitude.append(validation_slideFolders_classification[int((len(validation_slideFolders_classification) - 1)/2 + (len(validation_slideFolders_classification) - 1)/2/2):len(validation_slideFolders_classification)-1].astype(float))
    validation_y.append(validation_slideFolders_classification[-1].tolist())

"""
Construct decision flow data
"""
print("Constructing decision flow data")
train_DF_x = []
for tmp_idx in trange(len(train_y)):
    tmp_1 = train_DF_magnitude[tmp_idx][train_DF_data[tmp_idx] == 1].sum()
    tmp_2 = train_DF_magnitude[tmp_idx][train_DF_data[tmp_idx] == 2].sum()
    tmp_3 = train_DF_magnitude[tmp_idx][train_DF_data[tmp_idx] == 3].sum()
    tmp_4 = train_DF_magnitude[tmp_idx][train_DF_data[tmp_idx] == 4].sum()
    tmp_5 = train_DF_magnitude[tmp_idx][train_DF_data[tmp_idx] == 5].sum()
    sumOfMagnitude = tmp_1 + tmp_2 + tmp_3 + tmp_4 + tmp_5
    train_DF_x.append([tmp_1, tmp_2, tmp_3, tmp_4, tmp_5] / sumOfMagnitude)

test_DF_x = []
for tmp_idx in trange(len(test_y)):
    tmp_1 = test_DF_magnitude[tmp_idx][test_DF_data[tmp_idx] == 1].sum()
    tmp_2 = test_DF_magnitude[tmp_idx][test_DF_data[tmp_idx] == 2].sum()
    tmp_3 = test_DF_magnitude[tmp_idx][test_DF_data[tmp_idx] == 3].sum()
    tmp_4 = test_DF_magnitude[tmp_idx][test_DF_data[tmp_idx] == 4].sum()
    tmp_5 = test_DF_magnitude[tmp_idx][test_DF_data[tmp_idx] == 5].sum()
    sumOfMagnitude = tmp_1 + tmp_2 + tmp_3 + tmp_4 + tmp_5
    test_DF_x.append([tmp_1, tmp_2, tmp_3, tmp_4, tmp_5] / sumOfMagnitude)

validation_DF_x = []
for tmp_idx in trange(len(validation_y)):
    tmp_1 = validation_DF_magnitude[tmp_idx][validation_DF_data[tmp_idx] == 1].sum()
    tmp_2 = validation_DF_magnitude[tmp_idx][validation_DF_data[tmp_idx] == 2].sum()
    tmp_3 = validation_DF_magnitude[tmp_idx][validation_DF_data[tmp_idx] == 3].sum()
    tmp_4 = validation_DF_magnitude[tmp_idx][validation_DF_data[tmp_idx] == 4].sum()
    tmp_5 = validation_DF_magnitude[tmp_idx][validation_DF_data[tmp_idx] == 5].sum()
    sumOfMagnitude = tmp_1 + tmp_2 + tmp_3 + tmp_4 + tmp_5
    validation_DF_x.append([tmp_1, tmp_2, tmp_3, tmp_4, tmp_5] / sumOfMagnitude)

"""
Construct multi-class data
"""
print("Constructing multi-class data")
train_MC_x = []
for tmp_idx in trange(len(train_y)):
    tmp_1 = train_MC_magnitude[tmp_idx][train_MC_data[tmp_idx] == 1].sum()
    tmp_2 = train_MC_magnitude[tmp_idx][train_MC_data[tmp_idx] == 2].sum()
    tmp_3 = train_MC_magnitude[tmp_idx][train_MC_data[tmp_idx] == 3].sum()
    tmp_4 = train_MC_magnitude[tmp_idx][train_MC_data[tmp_idx] == 4].sum()
    tmp_5 = train_MC_magnitude[tmp_idx][train_MC_data[tmp_idx] == 5].sum()
    sumOfMagnitude = tmp_1 + tmp_2 + tmp_3 + tmp_4 + tmp_5
    train_MC_x.append([tmp_1, tmp_2, tmp_3, tmp_4, tmp_5] / sumOfMagnitude)

test_MC_x = []
for tmp_idx in trange(len(test_y)):
    tmp_1 = test_MC_magnitude[tmp_idx][test_MC_data[tmp_idx] == 1].sum()
    tmp_2 = test_MC_magnitude[tmp_idx][test_MC_data[tmp_idx] == 2].sum()
    tmp_3 = test_MC_magnitude[tmp_idx][test_MC_data[tmp_idx] == 3].sum()
    tmp_4 = test_MC_magnitude[tmp_idx][test_MC_data[tmp_idx] == 4].sum()
    tmp_5 = test_MC_magnitude[tmp_idx][test_MC_data[tmp_idx] == 5].sum()
    sumOfMagnitude = tmp_1 + tmp_2 + tmp_3 + tmp_4 + tmp_5
    test_MC_x.append([tmp_1, tmp_2, tmp_3, tmp_4, tmp_5] / sumOfMagnitude)

validation_MC_x = []
for tmp_idx in trange(len(validation_y)):
    tmp_1 = validation_MC_magnitude[tmp_idx][validation_MC_data[tmp_idx] == 1].sum()
    tmp_2 = validation_MC_magnitude[tmp_idx][validation_MC_data[tmp_idx] == 2].sum()
    tmp_3 = validation_MC_magnitude[tmp_idx][validation_MC_data[tmp_idx] == 3].sum()
    tmp_4 = validation_MC_magnitude[tmp_idx][validation_MC_data[tmp_idx] == 4].sum()
    tmp_5 = validation_MC_magnitude[tmp_idx][validation_MC_data[tmp_idx] == 5].sum()
    sumOfMagnitude = tmp_1 + tmp_2 + tmp_3 + tmp_4 + tmp_5
    validation_MC_x.append([tmp_1, tmp_2, tmp_3, tmp_4, tmp_5] / sumOfMagnitude)

"""
Normalize decision flow data
"""
print("Normalizing decision flow data")
scaler = StandardScaler()
scaler.fit(train_DF_x)
train_DF_x = scaler.transform(train_DF_x)
test_DF_x = scaler.transform(test_DF_x)
validation_DF_x = scaler.transform(validation_DF_x)

"""
Normalize multi-class data
"""
print("Normalizing multi-class data")
scaler = StandardScaler()
scaler.fit(train_MC_x)
train_MC_x = scaler.transform(train_MC_x)
test_MC_x = scaler.transform(test_MC_x)
validation_MC_x = scaler.transform(validation_MC_x)

"""
Change y to correct data type
"""
print("Changing y to correct data type")
train_y = np.array(train_y)
train_y = train_y.reshape(len(train_y), 1)

test_y = np.array(test_y)
test_y = test_y.reshape(len(test_y), 1)

validation_y = np.array(validation_y)
validation_y = validation_y.reshape(len(validation_y), 1)

"""
Manually label Gleason grade groups on y
"""
print("Labelling Gleason grade groups on y")
train_y[train_y == "negative"] = 0
train_y[train_y == "3+3"] = 1
train_y[train_y == "3+4"] = 2
train_y[train_y == "3+5"] = 4
train_y[train_y == "4+3"] = 3
train_y[train_y == "4+4"] = 4
train_y[train_y == "4+5"] = 5
train_y[train_y == "5+3"] = 4
train_y[train_y == "5+4"] = 5
train_y[train_y == "5+5"] = 5
train_y = train_y.astype(np.float)

test_y[test_y == "negative"] = 0
test_y[test_y == "3+3"] = 1
test_y[test_y == "3+4"] = 2
test_y[test_y == "3+5"] = 4
test_y[test_y == "4+3"] = 3
test_y[test_y == "4+4"] = 4
test_y[test_y == "4+5"] = 5
test_y[test_y == "5+3"] = 4
test_y[test_y == "5+4"] = 5
test_y[test_y == "5+5"] = 5
test_y = test_y.astype(np.float)

validation_y[validation_y == "negative"] = 0
validation_y[validation_y == "3+3"] = 1
validation_y[validation_y == "3+4"] = 2
validation_y[validation_y == "3+5"] = 4
validation_y[validation_y == "4+3"] = 3
validation_y[validation_y == "4+4"] = 4
validation_y[validation_y == "4+5"] = 5
validation_y[validation_y == "5+3"] = 4
validation_y[validation_y == "5+4"] = 5
validation_y[validation_y == "5+5"] = 5
validation_y = validation_y.astype(np.float)

"""
Change y to categorical data
"""
print("Changing y to categorical data")
train_y = tf.keras.utils.to_categorical(train_y, 6)
test_y = tf.keras.utils.to_categorical(test_y, 6)
validation_y = tf.keras.utils.to_categorical(validation_y, 6)

"""
Create and compile a new model
"""
print("Creating and compiling a new model")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, input_shape=(5,)),
    tf.keras.layers.Dense(5),
    tf.keras.layers.Dense(6, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
    loss="categorical_crossentropy"
)

"""
Train with decision flow data
"""
print("Training with decision flow data")
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=25, verbose=1)
history = model.fit(x=train_DF_x, y=train_y, epochs=200, validation_data=(validation_DF_x, validation_y), shuffle=True, callbacks=[es])

"""
Calcualte quadratic kappa for decision flow
"""
print("Calculating quadratic weighted kappa for decision flow")
yhat = model.predict(test_DF_x)
yhat = np.argmax(yhat, axis=1)
DF_quadratic_kappa = quadratic_kappa(np.argmax(test_y, axis=1), yhat, 6)

"""
Create and compile a new model
"""
print("Creating and compiling a new model")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, input_shape=(5,)),
    tf.keras.layers.Dense(5),
    tf.keras.layers.Dense(6, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
    loss="categorical_crossentropy"
)

"""
Train with multi-class data
"""
print("Training with multi-class data")
history = model.fit(x=train_MC_x, y=train_y, epochs=200, validation_data=(validation_MC_x, validation_y), shuffle=True, callbacks=[es])

"""
Calcualte quadratic kappa for multi-class
"""
print("Calculating quadratic weighted kappa for multi-class")
yhat = model.predict(test_MC_x)
yhat = np.argmax(yhat, axis=1)
MC_quadratic_kappa = quadratic_kappa(np.argmax(test_y, axis=1), yhat, 6)

"""
Print quadratic weighted kappa to stdout
"""
print("Quadratic weighted kappa (decision flow):", DF_quadratic_kappa)
print("Quadratic weighted kappa (multi-class):", MC_quadratic_kappa)