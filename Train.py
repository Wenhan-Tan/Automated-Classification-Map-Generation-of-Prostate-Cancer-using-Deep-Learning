#!/usr/bin/env python3
# coding: utf-8

"""
Author:
    Wenhan Tan
Date:
    2021/9

Description:
    This script trains 5 different models (4 for decision flow & 1 for multi-class).
    To choose which model to train, there is a variable called "trainModelID"
    can be changed to control. Train, test, and validation sets are all used and
    the model is DenseNet 201. DenseNet 201 doesnot have any regularizations, so
    a DropOut layer is added and L2 regularization is applied on all layers. At
    the end of training process, a few things are saved to computer (see Output below).
    This script is very long (about 970 lines), but the first 800 lines is only
    for loading correct amount of train, test, and valiation data for 5 different
    model. The rest 170 lines is for model setup and saving results.

Input:
    Provided by this work:
        1) patches_level1_128.csv (from script "Extract_patches.py")
        2) Extracted patches ("patches_path", from script "Extract_patches.py")
        3) patches_level1_128_slideFolders.csv (from script "Extract_patches.py")
        4) train_slideFolders.csv (from script "Train_test_validation_split.py")
        5) test_slideFolders.csv (from script "Train_test_validation_split.py")
        6) validation_slideFolders.csv (from script "Train_test_validation_split.py")

Output:
    1) Trained weights (example filepath: ./model_result/#dateTime/checkpoint/)
    2) Train and validation loss over time graph (example filename: ./model_result/#dateTime/Loss.png)
    3) Train and validation accuracy over time graph (example filename: ./model_result/#dateTime/Accuracy.png)
    4) Train and validation AUC over time graph (example filename: ./model_result/#dateTime/AUC.png)
    5) Testing result (example filename: ./model_result/#dateTime/testing_result.txt)
        
        Example:
            # Train, test, and validation data size and number of classes
            y_train distribution: (array([0, 1]), array([10000, 10000], dtype=int64))
            y_test distribution: (array([0, 1]), array([3944, 3944], dtype=int64))
            y_validation distribution: (array([0, 1]), array([2092, 2092], dtype=int64))

            # only 3 numbers for augmentation because only benign, pattern 3, and
            # pattern 5 do not have enough patches
            NbrOfAugmentationPatches: 0 0 771

            # loss, accuracy, AUC
            [0.8319622278213501, 0.9768002033233643, 0.9954116940498352]

            # Confusion matrix
            Confusion Matrix
            [3839  105]
            [  78 3866]

    6) Misclassified patches (exampel filename: ./model_result/#dateTime/a_1_9918.tiff)

        Filename explanation:
            True label:
            a is stroma, b is benign, c is pattern 3, d is pattern 4, e is pattern 5
            
            Predicted label:
            0 is stroma, 1 is benign, 2 is pattern 3, 3 is pattern 4, 4 is pattern 5

            a_1_9918.tiff: 
            The patch is stroma, but classified as benign

Usage:
    To reproduce results, simply run this script in terminal. Note that training
    takes a long time. Make sure you have all the python packages and input files ready.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tempfile
from tensorflow.keras import layers, models, applications
from sklearn.metrics import confusion_matrix
from PIL import Image
import os
from datetime import datetime
from tqdm.notebook import trange, tqdm

"""
Set GPU memory limit (Change the limit based on your GPU)
"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            # Memory limit is 5G (RTX 2060 max is 6G)
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5102)]
        )
    except RuntimeError as e:
        print(e)

"""
Learning rate scheduler function
"""
def step_decay(epoch, lr):
    """
    This function updates learning rate during training. It slowly decreases
    learning rate based on number of steps.
    """
    initial_lrate = 0.0002
    decay = 0.02

    return initial_lrate * (1 / (1 + decay * epoch))

"""
Filepath
"""
patches_info_path = "./prostate-cancer-grade-assessment/patches_level1_128.csv"
patches_path = "./prostate-cancer-grade-assessment/patches_level1_128/"
slideFolders_path = "./prostate-cancer-grade-assessment/patches_level1_128_slideFolders.csv"
train_slideFolders_path = "./prostate-cancer-grade-assessment/train_slideFolders.csv"
test_slideFolders_path = "./prostate-cancer-grade-assessment/test_slideFolders.csv"
validation_slideFolders_path = "./prostate-cancer-grade-assessment/validation_slideFolders.csv"

# Output result path
result_path = "./model_result/"

"""
Read in data
"""
print("Reading in data")
patches_id = np.genfromtxt(patches_info_path, delimiter=",", dtype='str', usecols=0)
slideFolders = np.genfromtxt(slideFolders_path, delimiter=",", dtype='str', usecols=0)
train_slideFolders = np.genfromtxt(train_slideFolders_path, delimiter=",", dtype='str', usecols=0)
test_slideFolders = np.genfromtxt(test_slideFolders_path, delimiter=",", dtype='str', usecols=0)
validation_slideFolders = np.genfromtxt(validation_slideFolders_path, delimiter=",", dtype='str', usecols=0)

"""
Control which model to train
"""
# Choose a number between 1 to 5 for training a model
# 1: Stroma v. Benign v. 3 v. 4 v. 5 (multi-class)
# 2: Stroma v. Benign + 3 + 4 + 5 (decision flow)
# 3: Benign v. 3 + 4 + 5 (decision flow)
# 4: 3 v. 4 + 5 (decision flow)
# 5: 4 v. 5 (decision flow)
trainModelID = 1 # IMPORTANT: Remember to change this value

"""
Check number of patches for train, test, and validation sets to make sure they
follow 64%-20%-16% ratio
"""
print("Checking 64-20-16 ratio")
test_error = 1
validation_error = 1

while test_error > 0.05 or validation_error > 0.05:
    train_count1 = 0
    train_count2 = 0
    train_count3 = 0
    train_count4 = 0
    train_count5 = 0
    test_count1 = 0
    test_count2 = 0
    test_count3 = 0
    test_count4 = 0
    test_count5 = 0
    validation_count1 = 0
    validation_count2 = 0
    validation_count3 = 0
    validation_count4 = 0
    validation_count5 = 0

    # count number of patches for each class in train set
    for f in tqdm(train_slideFolders):
        for c in os.listdir(patches_path + f):
            for p in os.listdir(patches_path + f + "/" + c):
                if int(c) == 1:
                    train_count1 += 1
                if int(c) == 2:
                    train_count2 += 1
                if int(c) == 3:
                    train_count3 += 1
                if int(c) == 4:
                    train_count4 += 1
                if int(c) == 5:
                    train_count5 += 1

    # count number of patches for each class in test set
    for f in tqdm(test_slideFolders):
        for c in os.listdir(patches_path + f):
            for p in os.listdir(patches_path + f + "/" + c):
                if int(c) == 1:
                    test_count1 += 1
                if int(c) == 2:
                    test_count2 += 1
                if int(c) == 3:
                    test_count3 += 1
                if int(c) == 4:
                    test_count4 += 1
                if int(c) == 5:
                    test_count5 += 1
    
    # count number of patches for each class in validation set
    for f in tqdm(validation_slideFolders):
        for c in os.listdir(patches_path + f):
            for p in os.listdir(patches_path + f + "/" + c):
                if int(c) == 1:
                    validation_count1 += 1
                if int(c) == 2:
                    validation_count2 += 1
                if int(c) == 3:
                    validation_count3 += 1
                if int(c) == 4:
                    validation_count4 += 1
                if int(c) == 5:
                    validation_count5 += 1

    # calculate errors from test set
    # error1: For stroma, (number of test patches / (number of train + test + validation patches)) - 0.2
    # the reason why it is 0.2 is because test set is 20% of entire data set
    error1 = np.absolute(test_count1 / (train_count1 + test_count1 + validation_count1) - 0.2)
    error2 = np.absolute(test_count2 / (train_count2 + test_count2 + validation_count2) - 0.2)
    error3 = np.absolute(test_count3 / (train_count3 + test_count3 + validation_count3) - 0.2)
    error4 = np.absolute(test_count4 / (train_count4 + test_count4 + validation_count4) - 0.2)
    error5 = np.absolute(test_count5 / (train_count5 + test_count5 + validation_count5) - 0.2)
    
    # calculate errors from validation set
    # error1: For stroma, (number of validation patches / (number of train + test + validation patches)) - 0.16
    # the reason why it is 0.16 is because validation set is 16% of entire data set
    error6 = np.absolute(validation_count1 / (train_count1 + test_count1 + validation_count1) - 0.16)
    error7 = np.absolute(validation_count2 / (train_count2 + test_count2 + validation_count2) - 0.16)
    error8 = np.absolute(validation_count3 / (train_count3 + test_count3 + validation_count3) - 0.16)
    error9 = np.absolute(validation_count4 / (train_count4 + test_count4 + validation_count4) - 0.16)
    error10 = np.absolute(validation_count5 / (train_count5 + test_count5 + validation_count5) - 0.16)
    
    # sum up errors from all 5 classes
    test_error = error1 + error2 + error3 + error4 + error5
    validation_error = error6 + error7 + error8 + error9 + error10

"""
Load train data
"""
print("Loading train data")

# "count" is for counting number of patches being loaded
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
x_train = []
y_train = []
patches1 = []
patches2 = []
patches3 = []
patches4 = []
patches5 = []

# "NbrOfAugmentation" is for number of augmentation patches needed
NbrOfAugmentation2 = 0
NbrOfAugmentation3 = 0
NbrOfAugmentation5 = 0

# set up number of patches of each class for train and validation sets
# train_count is the maximum number of patches available
# if train_count is not enough for training, we use image augmentation to
# create new images
if trainModelID == 1:
    # Stroma v. Benign v. 3 v. 4 v. 5
    NbrOfPatches1 = 10000
    
    if train_count2 >= 9000:
        NbrOfPatches2 = 9000
    else:
        NbrOfPatches2 = train_count2
    NbrOfAugmentation2 = 10000 - NbrOfPatches2
    
    if train_count3 >= 9000:
        NbrOfPatches3 = 9000
    else:
        NbrOfPatches3 = train_count3
    NbrOfAugmentation3 = 10000 - NbrOfPatches3
    
    NbrOfPatches4 = 10000
    
    if train_count5 >= 10000:
        NbrOfPatches5 = 10000
    else:
        NbrOfPatches5 = train_count5
    NbrOfAugmentation5 = 10000 - NbrOfPatches5
elif trainModelID == 2:
    # Stroma v. Benign, 3,  4, and 5
    NbrOfPatches1 = 10000
    
    if train_count2 >= 2500:
        NbrOfPatches2 = 2500
    else:
        NbrOfPatches2 = train_count2
    NbrOfAugmentation2 = 2500 - NbrOfPatches2
    
    if train_count3 >= 2500:
        NbrOfPatches3 = 2500
    else:
        NbrOfPatches3 = train_count3
    NbrOfAugmentation3 = 2500 - NbrOfPatches3
    
    NbrOfPatches4 = 2500
    
    if train_count5 >= 2500:
        NbrOfPatches5 = 2500
    else:
        NbrOfPatches5 = train_count5
    NbrOfAugmentation5 = 2500 - NbrOfPatches5
elif trainModelID == 3:
    # Benign v. 3, 4, and 5
    NbrOfPatches1 = 0
    
    if train_count2 >= 10000:
        NbrOfPatches2 = 10000
    else:
        NbrOfPatches2 = train_count2
    NbrOfAugmentation2 = 10000 - NbrOfPatches2
    
    if train_count3 >= 3333:
        NbrOfPatches3 = 3333
    else:
        NbrOfPatches3 = train_count3
    NbrOfAugmentation3 = 3333 - NbrOfPatches3
    
    NbrOfPatches4 = 3333
    
    if train_count5 >= 3333:
        NbrOfPatches5 = 3333
    else:
        NbrOfPatches5 = train_count5
    NbrOfAugmentation5 = 3333 - NbrOfPatches5
elif trainModelID == 4:
    # 3 v. 4 and 5
    NbrOfPatches1 = 0
    
    NbrOfPatches2 = 0
    NbrOfAugmentation2 = 0

    if train_count3 >= 10000:
        NbrOfPatches3 = 10000
    else:
        NbrOfPatches3 = train_count3
    NbrOfAugmentation3 = 10000 - NbrOfPatches3
    
    NbrOfPatches4 = 5000
    
    if train_count5 >= 5000:
        NbrOfPatches5 = 5000
    else:
        NbrOfPatches5 = train_count5
    NbrOfAugmentation5 = 5000 - NbrOfPatches5
elif trainModelID == 5:
    # 4 v. 5
    NbrOfPatches1 = 0
    
    NbrOfPatches2 = 0
    NbrOfAugmentation2 = 0
    
    NbrOfPatches3 = 0
    NbrOfAugmentation3 = 0
    
    NbrOfPatches4 = 10000
    
    if train_count5 >= 10000:
        NbrOfPatches5 = 10000
    else:
        NbrOfPatches5 = train_count5
    NbrOfAugmentation5 = 10000 - NbrOfPatches5

for f in tqdm(train_slideFolders):
    for c in os.listdir(patches_path + f):
        if int(c) == 1 and count1 < NbrOfPatches1:
            for p in os.listdir(patches_path + f + "/" + c):
                if count1 < NbrOfPatches1:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_train.append(img_data)
                    if trainModelID == 1:
                        y_train.append(0)
                    elif trainModelID == 2:
                        y_train.append(0)
                    patches1.append(img_data)
                    count1 += 1
                else:
                    break
        if int(c) == 2 and count2 < NbrOfPatches2:
            for p in os.listdir(patches_path + f + "/" + c):
                if count2 < NbrOfPatches2:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_train.append(img_data)
                    if trainModelID == 1:
                        y_train.append(1)
                    elif trainModelID == 2:
                        y_train.append(1)
                    elif trainModelID == 3:
                        y_train.append(0)
                    patches2.append(img_data)
                    count2 += 1
                else:
                    break
        if int(c) == 3 and count3 < NbrOfPatches3:
            for p in os.listdir(patches_path + f + "/" + c):
                if count3 < NbrOfPatches3:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_train.append(img_data)
                    if trainModelID == 1:
                        y_train.append(2)
                    elif trainModelID == 2:
                        y_train.append(1)
                    elif trainModelID == 3:
                        y_train.append(1)
                    elif trainModelID == 4:
                        y_train.append(0)
                    patches3.append(img_data)
                    count3 += 1
                else:
                    break
        if int(c) == 4 and count4 < NbrOfPatches4:
            for p in os.listdir(patches_path + f + "/" + c):
                if count4 < NbrOfPatches4:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_train.append(img_data)
                    if trainModelID == 1:
                        y_train.append(3)
                    elif trainModelID == 2:
                        y_train.append(1)
                    elif trainModelID == 3:
                        y_train.append(1)
                    elif trainModelID == 4:
                        y_train.append(1)
                    elif trainModelID == 5:
                        y_train.append(0)
                    patches4.append(img_data)
                    count4 += 1
                else:
                    break
        if int(c) == 5 and count5 < NbrOfPatches5:
            for p in os.listdir(patches_path + f + "/" + c):
                if count5 < NbrOfPatches5:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_train.append(img_data)
                    if trainModelID == 1:
                        y_train.append(4)
                    elif trainModelID == 2:
                        y_train.append(1)
                    elif trainModelID == 3:
                        y_train.append(1)
                    elif trainModelID == 4:
                        y_train.append(1)
                    elif trainModelID == 5:
                        y_train.append(1)
                    patches5.append(img_data)
                    count5 += 1
                else:
                    break

print("NbrOfTrainPatches:", count1, count2, count3, count4, count5)

"""
Load augmentation data
"""
print("Loading augmentation data")

def customAugmentation(img):
    """
    This function takes in an image and randomly applies an image augmentation
    method with random magnitude to create an new image. The new image is returned.
    """
    
    r = np.random.randint(low=1, high=7)
    if r == 1:
        temp = tf.image.rot90(img, k=np.random.randint(low=1, high=4))
    elif r == 2:
        temp = tf.image.random_brightness(img, max_delta=0.2)
    elif r == 3:
        temp = tf.image.random_hue(img, max_delta=0.2)
    elif r == 4:
        temp = tf.image.random_flip_left_right(img)
    elif r == 5:
        temp = tf.image.random_flip_up_down(img)
    elif r == 6:
        temp = tf.image.random_saturation(img, 0.6, 1.4)
    
    return temp

for i in trange(NbrOfAugmentation2):
    temp = patches2[np.random.randint(low=0, high=len(patches2))]
    for j in range(np.random.randint(low=2, high=6)):
        temp = customAugmentation(temp)
    x_train.append(temp)
    if trainModelID == 1:
        y_train.append(1)
    elif trainModelID == 2:
        y_train.append(1)
    elif trainModelID == 3:
        y_train.append(0)

for i in trange(NbrOfAugmentation3):
    temp = patches3[np.random.randint(low=0, high=len(patches3))]
    for j in range(np.random.randint(low=2, high=6)):
        temp = customAugmentation(temp)
    x_train.append(temp)
    if trainModelID == 1:
        y_train.append(2)
    elif trainModelID == 2:
        y_train.append(1)
    elif trainModelID == 3:
        y_train.append(1)
    elif trainModelID == 4:
        y_train.append(0)

for i in trange(NbrOfAugmentation5):
    temp = patches5[np.random.randint(low=0, high=len(patches5))]
    for j in range(np.random.randint(low=2, high=6)):
        temp = customAugmentation(temp)
    x_train.append(temp)
    if trainModelID == 1:
        y_train.append(4)
    elif trainModelID == 2:
        y_train.append(1)
    elif trainModelID == 3:
        y_train.append(1)
    elif trainModelID == 4:
        y_train.append(1)
    elif trainModelID == 5:
        y_train.append(1)

print("NbrOfAugmentationPatches:", NbrOfAugmentation2, NbrOfAugmentation3, NbrOfAugmentation5)

"""
Load test data
"""
print("Loading test data")

# "count" is for counting number of patches being loaded
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0

x_test = []
y_test = []

# set up number of patches of each class for test set
# test_count is the maximum number of patches available
# if test_count is not enough, the maximum number of patches that is available
# to all classes
if trainModelID == 1:
    # Stroma v. Benign v. 3 v. 4 v. 5
    NbrOfPatches1 = test_count5
    NbrOfPatches2 = test_count5
    NbrOfPatches3 = test_count5
    NbrOfPatches4 = test_count5
    NbrOfPatches5 = test_count5
elif trainModelID == 2:
    # Stroma v. Benign, 3, 4, and 5
    NbrOfPatches1 = test_count5 * 4
    NbrOfPatches2 = test_count5
    NbrOfPatches3 = test_count5
    NbrOfPatches4 = test_count5
    NbrOfPatches5 = test_count5
elif trainModelID == 3:
    # Benign v. 3, 4, and 5
    NbrOfPatches1 = 0
    NbrOfPatches2 = test_count2
    NbrOfPatches3 = int(test_count2 / 3)
    NbrOfPatches4 = int(test_count2 / 3)
    NbrOfPatches5 = int(test_count2 / 3)
elif trainModelID == 4:
    # 3 v. 4 and 5
    NbrOfPatches1 = 0
    NbrOfPatches2 = 0
    NbrOfPatches3 = test_count5 * 2
    NbrOfPatches4 = test_count5
    NbrOfPatches5 = test_count5
elif trainModelID == 5:
    # 4 v. 5
    NbrOfPatches1 = 0
    NbrOfPatches2 = 0
    NbrOfPatches3 = 0
    NbrOfPatches4 = test_count5
    NbrOfPatches5 = test_count5

for f in tqdm(test_slideFolders):
    for c in os.listdir(patches_path + f):
        if int(c) == 1 and count1 < NbrOfPatches1:
            for p in os.listdir(patches_path + f + "/" + c):
                if count1 < NbrOfPatches1:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_test.append(img_data)
                    if trainModelID == 1:
                        y_train.append(0)
                    elif trainModelID == 2:
                        y_train.append(0)
                    count1 += 1
                else:
                    break
        if int(c) == 2 and count2 < NbrOfPatches2:
            for p in os.listdir(patches_path + f + "/" + c):
                if count2 < NbrOfPatches2:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_test.append(img_data)
                    if trainModelID == 1:
                        y_train.append(1)
                    elif trainModelID == 2:
                        y_train.append(1)
                    elif trainModelID == 3:
                        y_train.append(0)
                    count2 += 1
                else:
                    break
        if int(c) == 3 and count3 < NbrOfPatches3:
            for p in os.listdir(patches_path + f + "/" + c):
                if count3 < NbrOfPatches3:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_test.append(img_data)
                    if trainModelID == 1:
                        y_train.append(2)
                    elif trainModelID == 2:
                        y_train.append(1)
                    elif trainModelID == 3:
                        y_train.append(1)
                    elif trainModelID == 4:
                        y_train.append(0)
                    count3 += 1
                else:
                    break
        if int(c) == 4 and count4 < NbrOfPatches4:
            for p in os.listdir(patches_path + f + "/" + c):
                if count4 < NbrOfPatches4:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_test.append(img_data)
                    if trainModelID == 1:
                        y_train.append(3)
                    elif trainModelID == 2:
                        y_train.append(1)
                    elif trainModelID == 3:
                        y_train.append(1)
                    elif trainModelID == 4:
                        y_train.append(1)
                    elif trainModelID == 5:
                        y_train.append(0)
                    count4 += 1
                else:
                    break
        if int(c) == 5 and count5 < NbrOfPatches5:
            for p in os.listdir(patches_path + f + "/" + c):
                if count5 < NbrOfPatches5:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_test.append(img_data)
                    if trainModelID == 1:
                        y_train.append(4)
                    elif trainModelID == 2:
                        y_train.append(1)
                    elif trainModelID == 3:
                        y_train.append(1)
                    elif trainModelID == 4:
                        y_train.append(1)
                    elif trainModelID == 5:
                        y_train.append(1)
                    count5 += 1
                else:
                    break

print("NbrOfTestPatches:", count1, count2, count3, count4, count5)

"""
Load validation data
"""
print("Loading validation data")

# "count" is for counting number of patches being loaded
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0

x_validation = []
y_validation = []

# set up number of patches of each class for validation set
# validation_count is the maximum number of patches available
# if validation_count is not enough, the maximum number of patches that is available
# to all classes
if trainModelID == 1:
    # Stroma v. Benign v. 3 v. 4 v. 5
    NbrOfPatches1 = validation_count5
    NbrOfPatches2 = validation_count5
    NbrOfPatches3 = validation_count5
    NbrOfPatches4 = validation_count5
    NbrOfPatches5 = validation_count5
elif trainModelID == 2:
    # Stroma v. Benign, 3, 4, and 5
    NbrOfPatches1 = validation_count5 * 4
    NbrOfPatches2 = validation_count5
    NbrOfPatches3 = validation_count5
    NbrOfPatches4 = validation_count5
    NbrOfPatches5 = validation_count5
elif trainModelID == 3:
    # Benign v. 3, 4, and 5
    NbrOfPatches1 = 0
    NbrOfPatches2 = validation_count2
    NbrOfPatches3 = int(validation_count2 / 3)
    NbrOfPatches4 = int(validation_count2 / 3)
    NbrOfPatches5 = int(validation_count2 / 3)
elif trainModelID == 4:
    # 3 v. 4 and 5
    NbrOfPatches1 = 0
    NbrOfPatches2 = 0
    NbrOfPatches3 = validation_count5 * 2
    NbrOfPatches4 = validation_count5
    NbrOfPatches5 = validation_count5
elif trainModelID == 5:
    # 4 v. 5
    NbrOfPatches1 = 0
    NbrOfPatches2 = 0
    NbrOfPatches3 = 0
    NbrOfPatches4 = validation_count5
    NbrOfPatches5 = validation_count5

for f in tqdm(validation_slideFolders):
    for c in os.listdir(patches_path + f):
        if int(c) == 1 and count1 < NbrOfPatches1:
            for p in os.listdir(patches_path + f + "/" + c):
                if count1 < NbrOfPatches1:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_validation.append(img_data)
                    if trainModelID == 1:
                        y_train.append(0)
                    elif trainModelID == 2:
                        y_train.append(0)
                    count1 += 1
                else:
                    break
        if int(c) == 2 and count2 < NbrOfPatches2:
            for p in os.listdir(patches_path + f + "/" + c):
                if count2 < NbrOfPatches2:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_validation.append(img_data)
                    if trainModelID == 1:
                        y_train.append(1)
                    elif trainModelID == 2:
                        y_train.append(1)
                    elif trainModelID == 3:
                        y_train.append(0)
                    count2 += 1
                else:
                    break
        if int(c) == 3 and count3 < NbrOfPatches3:
            for p in os.listdir(patches_path + f + "/" + c):
                if count3 < NbrOfPatches3:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_validation.append(img_data)
                    if trainModelID == 1:
                        y_train.append(2)
                    elif trainModelID == 2:
                        y_train.append(1)
                    elif trainModelID == 3:
                        y_train.append(1)
                    elif trainModelID == 4:
                        y_train.append(0)
                    count3 += 1
                else:
                    break
        if int(c) == 4 and count4 < NbrOfPatches4:
            for p in os.listdir(patches_path + f + "/" + c):
                if count4 < NbrOfPatches4:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_validation.append(img_data)
                    if trainModelID == 1:
                        y_train.append(3)
                    elif trainModelID == 2:
                        y_train.append(1)
                    elif trainModelID == 3:
                        y_train.append(1)
                    elif trainModelID == 4:
                        y_train.append(1)
                    elif trainModelID == 5:
                        y_train.append(0)
                    count4 += 1
                else:
                    break
        if int(c) == 5 and count5 < NbrOfPatches5:
            for p in os.listdir(patches_path + f + "/" + c):
                if count5 < NbrOfPatches5:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_validation.append(img_data)
                    if trainModelID == 1:
                        y_train.append(4)
                    elif trainModelID == 2:
                        y_train.append(1)
                    elif trainModelID == 3:
                        y_train.append(1)
                    elif trainModelID == 4:
                        y_train.append(1)
                    elif trainModelID == 5:
                        y_train.append(1)
                    count5 += 1
                else:
                    break

print("NbrOfValidationPatches:", count1, count2, count3, count4, count5)

"""
Print data info
"""
train_distribution = np.unique(y_train, return_counts=True)
test_distribution = np.unique(y_test, return_counts=True)
validation_distribution = np.unique(y_validation, return_counts=True)
print("y_train distribution:", train_distribution)
print("y_test distribution:", test_distribution)
print("y_validation distribution:", validation_distribution)

"""
Convert train data to numpy array, reshape and normalize
"""
print("Converting train data to numpy, reshape, and normalize")
x_train = np.stack(x_train, axis=0)
y_train = np.array(y_train)
x_test = np.stack(x_test, axis=0)
y_test = np.array(y_test)
x_validation = np.stack(x_validation, axis=0)
y_validation = np.array(y_validation)

x_train = x_train.reshape(len(x_train), 128, 128, 3) / 255.0
y_train = y_train.reshape(len(y_train), 1)
x_test = x_test.reshape(len(x_test), 128, 128, 3) / 255.0
y_test = y_test.reshape(len(y_test), 1)
x_validation = x_validation.reshape(len(x_validation), 128, 128, 3) / 255.0
y_validation = y_validation.reshape(len(y_validation), 1)

"""
Convert to one hot encoding
"""
y_train = tf.keras.utils.to_categorical(y_train, 2)
y_test = tf.keras.utils.to_categorical(y_test, 2)
y_validation = tf.keras.utils.to_categorical(y_validation, 2)

"""
Model setup
"""
print("Setting up a model")
denseNet = applications.DenseNet201(include_top=False, weights="imagenet", input_shape=(128, 128, 3),  pooling='avg')

output = denseNet.output
output = layers.Dropout(0.5)(output)
output = layers.Dense(2, activation='softmax')(output)

model = tf.keras.models.Model(denseNet.input, output)

model.trainable = True
regularizer = tf.keras.regularizers.l2(0.0001)
for layer in model.layers:
    for attr in ['kernel_regularizer']:
        if hasattr(layer, attr):
            setattr(layer, attr, regularizer)

model_json = model.to_json()
tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
model.save_weights(tmp_weights_path)
model = tf.keras.models.model_from_json(model_json)
model.load_weights(tmp_weights_path, by_name=True)

# find out current datetime
now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
if not os.path.isdir(result_path + dt_string):
    os.mkdir(result_path + dt_string)

# path for saving weights
checkpoint_filepath = result_path + dt_string + "/checkpoint/"
if not os.path.isdir(checkpoint_filepath):
    os.mkdir(checkpoint_filepath)

# learning rate scheduler
lrs = tf.keras.callbacks.LearningRateScheduler(step_decay)
# early stopping callback
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=25, verbose=1)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

opt = tf.keras.optimizers.SGD(learning_rate=0.0002, momentum=0.7)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer=opt,
             loss=loss,
             metrics=['accuracy','AUC'])
history = model.fit(x=x_train, y=y_train, epochs=400, batch_size=32, validation_data=(x_validation, y_validation), shuffle=True, callbacks=[lrs, es, checkpoint_callback])
model.load_weights(checkpoint_filepath)

"""
Accuracy, Loss, AUC, and Kappa plots
"""
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(['train', 'validation'])
plt.grid()
plt.savefig(result_path + dt_string + '/Accuracy.png')
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['train', 'validation'])
plt.grid()
plt.savefig(result_path + dt_string + '/Loss.png')
plt.close()

plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.legend(['train', 'validation'])
plt.grid()
plt.savefig(result_path + dt_string + '/AUC.png')
plt.close()

"""
Save results
"""
f = open(result_path + dt_string + "/testing_result.txt", "w")
f.write("y_train distribution: " + str(train_distribution) + "\n")
f.write("y_test distribution: " + str(test_distribution) + "\n")
f.write("y_validation distribution: " + str(validation_distribution) + "\n")
f.write("NbrOfAugmentationPatches: " + str(NbrOfAugmentation2) + " " + str(NbrOfAugmentation3) + " " + str(NbrOfAugmentation5) + "\n")
f.write(str(model.evaluate(x_test, y_test)) + "\n")

yhat = model.predict(x_test)

"""
Convert back to label encoding
"""
yhat = np.argmax(yhat, axis=1)
y_test = np.argmax(y_test, axis=1)

"""
Confusion Matrix
"""
confMatrix = confusion_matrix(y_test, yhat)

f.write("Confusion Matrix\n")
for i in confMatrix:
    f.write(str(i) + "\n")
f.close()

"""
Save misclassified image
True label: character
Predicted label: number
"""
temp = np.equal(yhat, y_test)
for idx in trange(len(temp)):
    if temp[idx] == False:
        im = Image.fromarray((x_test[idx] * 255).astype(np.uint8))
        if y_test[idx] == 0:
            im.save(result_path + dt_string + "/a_" + str(yhat[idx]) + "_" + str(idx) + ".tiff")
        elif y_test[idx] == 1:
            im.save(result_path + dt_string + "/b_" + str(yhat[idx]) + "_" + str(idx) + ".tiff")
        elif y_test[idx] == 2:
            im.save(result_path + dt_string + "/c_" + str(yhat[idx]) + "_" + str(idx) + ".tiff")
        elif y_test[idx] == 3:
            im.save(result_path + dt_string + "/d_" + str(yhat[idx]) + "_" + str(idx) + ".tiff")
        elif y_test[idx] == 4:
            im.save(result_path + dt_string + "/e_" + str(yhat[idx]) + "_" + str(idx) + ".tiff")