#!/usr/bin/env python3
# coding: utf-8

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
Set up GPU
"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5102)]
        )
    except RuntimeError as e:
        print(e)

"""
Split into train/test based on cases
"""
# patches_info_path = "./prostate-cancer-grade-assessment/patches_level1_128.csv"
# patches_path = "./prostate-cancer-grade-assessment/patches_level1_128/"
# slideFolders_path = "./prostate-cancer-grade-assessment/patches_level1_128_slideFolders.csv"

patches_info_path = "./prostate-cancer-grade-assessment/temp_patches_level1_128.csv"
patches_path = "./prostate-cancer-grade-assessment/temp_patches_level1_128/"
slideFolders_path = "./prostate-cancer-grade-assessment/temp_patches_level1_128_slideFolders.csv"

patches_id = np.genfromtxt(patches_info_path, delimiter=",", dtype='str', usecols=0)
slideFolders = np.genfromtxt(slideFolders_path, delimiter=",", dtype='str', usecols=0)

test_error = 1
validation_error = 1
while test_error > 0.05 or validation_error > 0.05:
    # np.random.shuffle(slideFolders)
    # train_slideFolders = slideFolders[:int(np.ceil(len(slideFolders) * 8 / 10))]
    # test_slideFolders = slideFolders[int(np.ceil(len(slideFolders) * 8 / 10)):]
    # validation_slideFolders = train_slideFolders[int(np.ceil(len(train_slideFolders) * 8 / 10)):]
    # train_slideFolders = train_slideFolders[:int(np.ceil(len(train_slideFolders) * 8 / 10))]

    # train_slideFolders = np.genfromtxt("./prostate-cancer-grade-assessment/train_slideFolders.csv", delimiter=",", dtype='str', usecols=0)
    # test_slideFolders = np.genfromtxt("./prostate-cancer-grade-assessment/test_slideFolders.csv", delimiter=",", dtype='str', usecols=0)
    # validation_slideFolders = np.genfromtxt("./prostate-cancer-grade-assessment/validation_slideFolders.csv", delimiter=",", dtype='str', usecols=0)

    train_slideFolders = np.genfromtxt("./prostate-cancer-grade-assessment/temp_train_slideFolders.csv", delimiter=",", dtype='str', usecols=0)
    test_slideFolders = np.genfromtxt("./prostate-cancer-grade-assessment/temp_test_slideFolders.csv", delimiter=",", dtype='str', usecols=0)
    validation_slideFolders = np.genfromtxt("./prostate-cancer-grade-assessment/temp_validation_slideFolders.csv", delimiter=",", dtype='str', usecols=0)

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

    error1 = np.absolute(test_count1 / (train_count1 + test_count1 + validation_count1) - 0.2)
    error2 = np.absolute(test_count2 / (train_count2 + test_count2 + validation_count2) - 0.2)
    error3 = np.absolute(test_count3 / (train_count3 + test_count3 + validation_count3) - 0.2)
    error4 = np.absolute(test_count4 / (train_count4 + test_count4 + validation_count4) - 0.2)
    error5 = np.absolute(test_count5 / (train_count5 + test_count5 + validation_count5) - 0.2)
    
    error6 = np.absolute(validation_count1 / (train_count1 + test_count1 + validation_count1) - 0.16)
    error7 = np.absolute(validation_count2 / (train_count2 + test_count2 + validation_count2) - 0.16)
    error8 = np.absolute(validation_count3 / (train_count3 + test_count3 + validation_count3) - 0.16)
    error9 = np.absolute(validation_count4 / (train_count4 + test_count4 + validation_count4) - 0.16)
    error10 = np.absolute(validation_count5 / (train_count5 + test_count5 + validation_count5) - 0.16)
    
    test_error = error1 + error2 + error3 + error4 + error5
    validation_error = error6 + error7 + error8 + error9 + error10

# np.savetxt("./prostate-cancer-grade-assessment/temp_train_slideFolders.csv", train_slideFolders, delimiter=",", fmt="%s")
# np.savetxt("./prostate-cancer-grade-assessment/temp_test_slideFolders.csv", test_slideFolders, delimiter=",", fmt="%s")
# np.savetxt("./prostate-cancer-grade-assessment/temp_validation_slideFolders.csv", validation_slideFolders, delimiter=",", fmt="%s")


"""
Load train data
"""
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
NbrOfAugmentation1 = 0
NbrOfAugmentation2 = 0
NbrOfAugmentation3 = 0
NbrOfAugmentation4 = 0
NbrOfAugmentation5 = 0

# Stroma v. Benign + Aug v. 3 + Aug v. 4 and 5 + Aug
# NbrOfPatches1 = 10000
# NbrOfPatches2 = train_count2
# NbrOfAugmentation2 = 10000 - train_count2
# NbrOfPatches3 = train_count3
# NbrOfAugmentation3 = 10000 - train_count3
# NbrOfPatches4 = 5000
# NbrOfPatches5 = train_count5
# NbrOfAugmentation5 = 5000 - train_count5

# Stroma v. Benign v. 3 v. 4 v. 5
# NbrOfPatches1 = 8000
# NbrOfAugmentation1 = 9000 - NbrOfPatches1
# NbrOfPatches2 = 8000
# NbrOfAugmentation2 = 9000 - NbrOfPatches2
# NbrOfPatches3 = 8000
# NbrOfAugmentation3 = 9000 - NbrOfPatches3
# NbrOfPatches4 = 8000
# NbrOfAugmentation4 = 9000 - NbrOfPatches4
# if train_count5 >= 8000:
#     NbrOfPatches5 = 8000
#     NbrOfAugmentation5 = 9000 - NbrOfPatches5
# else:
#     NbrOfPatches5 = train_count5
#     NbrOfAugmentation5 = 9000 - NbrOfPatches5
# num_per_case = [5, 25, 21, 13]

# Stroma v. Benign, 3,  4, and 5
# NbrOfPatches1 = 18000
# NbrOfAugmentation1 = 20000 - NbrOfPatches1
# NbrOfPatches2 = 4500
# NbrOfAugmentation2 = 5000 - NbrOfPatches2
# NbrOfPatches3 = 4500
# NbrOfAugmentation3 = 5000 - NbrOfPatches3
# NbrOfPatches4 = 4500
# NbrOfAugmentation4 = 5000 - NbrOfPatches4
# NbrOfPatches5 = 4500
# NbrOfAugmentation5 = 5000 - NbrOfPatches5
# num_per_case = [6, 17, 16, 10, 131]

# Benign v. 3, 4, and 5
# NbrOfPatches1 = 0
# NbrOfAugmentation1 = 0
# NbrOfPatches2 = 16000
# NbrOfAugmentation2 = 20000 - NbrOfPatches2
# NbrOfPatches3 = 5500
# NbrOfAugmentation3 = 6666 - NbrOfPatches3
# NbrOfPatches4 = 5500
# NbrOfAugmentation4 = 6666 - NbrOfPatches4
# NbrOfPatches5 = 5500
# NbrOfAugmentation5 = 6666 - NbrOfPatches5
# num_per_case = [None, 175, 20, 12, 160]

# Benign + Aug v. 3 + Aug
# NbrOfPatches1 = 0
# NbrOfPatches2 = train_count2
# NbrOfAugmentation2 = 10000 - train_count2
# if train_count3 < 10000:
#     NbrOfPatches3 = train_count3
#     NbrOfAugmentation3 = 10000 - train_count3
# else:
#     NbrOfPatches3 = 10000
# NbrOfPatches4 = 0
# NbrOfPatches5 = 0

# Benign + Aug v. 4
# NbrOfPatches1 = 0
# NbrOfPatches2 = train_count2
# NbrOfAugmentation2 = 10000 - train_count2
# NbrOfPatches3 = 0
# NbrOfPatches4 = 10000
# NbrOfPatches5 = 0

# 3 v. 4 and 5
# NbrOfPatches1 = 0
# NbrOfAugmentation1 = 0
# NbrOfPatches2 = 0
# NbrOfAugmentation2 = 0
# NbrOfPatches3 = 16000
# NbrOfAugmentation3 = 20000 - NbrOfPatches3
# NbrOfPatches4 = 8000
# NbrOfAugmentation4 = 10000 - NbrOfPatches4
# if train_count5 >= 8000:
#     NbrOfPatches5 = 8000
#     NbrOfAugmentation5 = 10000 - NbrOfPatches5
# else:
#     NbrOfPatches5 = train_count5
#     NbrOfAugmentation5 = 10000 - NbrOfPatches5
# num_per_case = [None, None, 30, 13, None]

# 3 + Aug v. 4 and 5 + Aug
# NbrOfPatches1 = 0
# NbrOfPatches2 = 0
# NbrOfPatches3 = train_count3
# NbrOfAugmentation3 = 10000 - train_count3
# NbrOfPatches4 = 5000
# NbrOfPatches5 = train_count5
# NbrOfAugmentation5 = 5000 - train_count5

# 3 v. 4
# NbrOfPatches1 = 0
# NbrOfPatches2 = 0
# NbrOfPatches3 = train_count3
# NbrOfAugmentation3 = 10000 - train_count3
# NbrOfPatches4 = 10000
# NbrOfPatches5 = 0

# 4 v. 5
NbrOfPatches1 = 0
NbrOfAugmentation1 = 0
NbrOfPatches2 = 0
NbrOfAugmentation2 = 0
NbrOfPatches3 = 0
NbrOfAugmentation3 = 0
NbrOfPatches4 = train_count5
NbrOfAugmentation4 = 13000 - NbrOfPatches4
NbrOfPatches5 = train_count5
NbrOfAugmentation5 = 13000 - train_count5
num_per_case = [None, None, None, 13, None]

for f in tqdm(train_slideFolders):
    count1_num_per_case = 0
    count2_num_per_case = 0
    count3_num_per_case = 0
    count4_num_per_case = 0
    count5_num_per_case = 0
    for c in os.listdir(patches_path + f):
        if int(c) == 1 and count1 < NbrOfPatches1:
            for p in os.listdir(patches_path + f + "/" + c):
                if count1 < NbrOfPatches1:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_train.append(img_data)
                    y_train.append(0)
                    patches1.append(img_data)
                    count1 += 1

                    count1_num_per_case += 1
                    if count1_num_per_case >= int((NbrOfPatches1 - count1) * num_per_case[0] / len(train_slideFolders)):
                        break
                else:
                    break
        if int(c) == 2 and count2 < NbrOfPatches2:
            for p in os.listdir(patches_path + f + "/" + c):
                if count2 < NbrOfPatches2:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_train.append(img_data)
                    y_train.append(0)
                    patches2.append(img_data)
                    count2 += 1

                    count2_num_per_case += 1
                    if count2_num_per_case >= int((NbrOfPatches2 - count2) * num_per_case[1] / len(train_slideFolders)):
                        break
                else:
                    break
        if int(c) == 3 and count3 < NbrOfPatches3:
            for p in os.listdir(patches_path + f + "/" + c):
                if count3 < NbrOfPatches3:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_train.append(img_data)
                    y_train.append(0)
                    patches3.append(img_data)
                    count3 += 1

                    count3_num_per_case += 1
                    if count3_num_per_case >= int((NbrOfPatches3 - count3) * num_per_case[2] / len(train_slideFolders)):
                        break
                else:
                    break
        if int(c) == 4 and count4 < NbrOfPatches4:
            for p in os.listdir(patches_path + f + "/" + c):
                if count4 < NbrOfPatches4:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_train.append(img_data)
                    y_train.append(0)
                    patches4.append(img_data)
                    count4 += 1

                    count4_num_per_case += 1
                    if count4_num_per_case >= int((NbrOfPatches4 - count4) * num_per_case[3] / len(train_slideFolders)):
                        break
                else:
                    break
        if int(c) == 5 and count5 < NbrOfPatches5:
            for p in os.listdir(patches_path + f + "/" + c):
                if count5 < NbrOfPatches5:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_train.append(img_data)
                    y_train.append(1)
                    patches5.append(img_data)
                    count5 += 1

                    # count5_num_per_case += 1
                    # if count5_num_per_case >= int((NbrOfPatches5 - count5) * num_per_case[4] / len(train_slideFolders)):
                    #     break
                else:
                    break

print("NbrOfTrainPatches:", count1, count2, count3, count4, count5)

"""
Load augmentation data
"""
def customAugmentation(img):
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

for i in trange(NbrOfAugmentation1):
    temp = patches1[np.random.randint(low=0, high=len(patches1))]
    for j in range(np.random.randint(low=2, high=6)):
        temp = customAugmentation(temp)
    x_train.append(temp)
    y_train.append(0)

for i in trange(NbrOfAugmentation2):
    temp = patches2[np.random.randint(low=0, high=len(patches2))]
    for j in range(np.random.randint(low=2, high=6)):
        temp = customAugmentation(temp)
    x_train.append(temp)
    y_train.append(0)

for i in trange(NbrOfAugmentation3):
    temp = patches3[np.random.randint(low=0, high=len(patches3))]
    for j in range(np.random.randint(low=2, high=6)):
        temp = customAugmentation(temp)
    x_train.append(temp)
    y_train.append(0)

for i in trange(NbrOfAugmentation4):
    temp = patches4[np.random.randint(low=0, high=len(patches4))]
    for j in range(np.random.randint(low=2, high=6)):
        temp = customAugmentation(temp)
    x_train.append(temp)
    y_train.append(0)

for i in trange(NbrOfAugmentation5):
    temp = patches5[np.random.randint(low=0, high=len(patches5))]
    for j in range(np.random.randint(low=2, high=6)):
        temp = customAugmentation(temp)
    x_train.append(temp)
    y_train.append(1)

print("NbrOfAugmentationPatches:", NbrOfAugmentation1, NbrOfAugmentation2, NbrOfAugmentation3, NbrOfAugmentation4, NbrOfAugmentation5)

"""
Load test data
"""
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
x_test = []
y_test = []

# Stroma v. Benign v. 3 v. 4 and 5
# NbrOfPatches1 = test_count2
# NbrOfPatches2 = test_count2
# NbrOfPatches3 = test_count2
# NbrOfPatches4 = int(test_count2 / 2)
# NbrOfPatches5 = int(test_count2 / 2)

# Stroma v. Benign v. 3 v. 4 v. 5
# NbrOfPatches1 = test_count5
# NbrOfPatches2 = test_count5
# NbrOfPatches3 = test_count5
# NbrOfPatches4 = test_count5
# NbrOfPatches5 = test_count5
# num_per_case = [5, 30, 21, 13]

# Stroma v. Benign, 3, 4, and 5
# NbrOfPatches1 = test_count5 * 4
# NbrOfPatches2 = test_count5
# NbrOfPatches3 = test_count5
# NbrOfPatches4 = test_count5
# NbrOfPatches5 = test_count5
# num_per_case = [7, 30, 21, 13]

# Benign v. 3, 4, and 5
# NbrOfPatches1 = 0
# NbrOfPatches2 = test_count2
# NbrOfPatches3 = int(test_count2 / 3)
# NbrOfPatches4 = int(test_count2 / 3)
# NbrOfPatches5 = int(test_count2 / 3)
# num_per_case = [None, None, 17, 12, 200]

# Benign v. 3
# NbrOfPatches1 = 0
# NbrOfPatches2 = test_count2
# NbrOfPatches3 = test_count2
# NbrOfPatches4 = 0
# NbrOfPatches5 = 0

# Benign v. 4
# NbrOfPatches1 = 0
# NbrOfPatches2 = test_count2
# NbrOfPatches3 = 0
# NbrOfPatches4 = test_count2
# NbrOfPatches5 = 0

# 3 v. 4 and 5
# NbrOfPatches1 = 0
# NbrOfPatches2 = 0
# NbrOfPatches3 = test_count5 * 2
# NbrOfPatches4 = test_count5
# NbrOfPatches5 = test_count5
# num_per_case = [None, None, 34, 13, None]

# 3 v. 4
# NbrOfPatches1 = 0
# NbrOfPatches2 = 0
# NbrOfPatches3 = test_count3
# NbrOfPatches4 = test_count3
# NbrOfPatches5 = 0

# 4 v. 5
NbrOfPatches1 = 0
NbrOfPatches2 = 0
NbrOfPatches3 = 0
NbrOfPatches4 = test_count5
NbrOfPatches5 = test_count5
num_per_case = [None, None, None, 13, None]

for f in tqdm(test_slideFolders):
    count1_num_per_case = 0
    count2_num_per_case = 0
    count3_num_per_case = 0
    count4_num_per_case = 0
    count5_num_per_case = 0
    for c in os.listdir(patches_path + f):
        if int(c) == 1 and count1 < NbrOfPatches1:
            for p in os.listdir(patches_path + f + "/" + c):
                if count1 < NbrOfPatches1:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_test.append(img_data)
                    y_test.append(0)
                    count1 += 1

                    count1_num_per_case += 1
                    if count1_num_per_case >= int((NbrOfPatches1 - count1) * num_per_case[0] / len(test_slideFolders)):
                        break
                else:
                    break
        if int(c) == 2 and count2 < NbrOfPatches2:
            for p in os.listdir(patches_path + f + "/" + c):
                if count2 < NbrOfPatches2:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_test.append(img_data)
                    y_test.append(0)
                    count2 += 1

                    # count2_num_per_case += 1
                    # if count2_num_per_case >= int((NbrOfPatches2 - count2) * num_per_case[1] / len(test_slideFolders)):
                    #     break
                else:
                    break
        if int(c) == 3 and count3 < NbrOfPatches3:
            for p in os.listdir(patches_path + f + "/" + c):
                if count3 < NbrOfPatches3:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_test.append(img_data)
                    y_test.append(0)
                    count3 += 1

                    count3_num_per_case += 1
                    if count3_num_per_case >= int((NbrOfPatches3 - count3) * num_per_case[2] / len(test_slideFolders)):
                        break
                else:
                    break
        if int(c) == 4 and count4 < NbrOfPatches4:
            for p in os.listdir(patches_path + f + "/" + c):
                if count4 < NbrOfPatches4:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_test.append(img_data)
                    y_test.append(0)
                    count4 += 1

                    count4_num_per_case += 1
                    if count4_num_per_case >= int((NbrOfPatches4 - count4) * num_per_case[3] / len(test_slideFolders)):
                        break
                else:
                    break
        if int(c) == 5 and count5 < NbrOfPatches5:
            for p in os.listdir(patches_path + f + "/" + c):
                if count5 < NbrOfPatches5:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_test.append(img_data)
                    y_test.append(1)
                    count5 += 1

                    # count5_num_per_case += 1
                    # if count5_num_per_case >= int((NbrOfPatches5 - count5) * num_per_case[4] / len(test_slideFolders)):
                    #     break
                else:
                    break

print("NbrOfTestPatches:", count1, count2, count3, count4, count5)

"""
Load validation data
"""
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
x_validation = []
y_validation = []

# Stroma v. Benign v. 3 v. 4 and 5
# NbrOfPatches1 = validation_count2
# NbrOfPatches2 = validation_count2
# NbrOfPatches3 = validation_count2
# NbrOfPatches4 = int(validation_count2 / 2)
# NbrOfPatches5 = int(validation_count2 / 2)

# Stroma v. Benign v. 3 v. 4 v. 5
# NbrOfPatches1 = validation_count5
# NbrOfPatches2 = validation_count5
# NbrOfPatches3 = validation_count5
# NbrOfPatches4 = validation_count5
# NbrOfPatches5 = validation_count5
# num_per_case = [5, 44, 21, 14]

# Stroma v. Benign, 3, 4, and 5
# NbrOfPatches1 = 6732
# NbrOfPatches2 = 1683
# NbrOfPatches3 = 1683
# NbrOfPatches4 = 1683
# NbrOfPatches5 = 1683
# num_per_case = [6, 26, 18, 12, 150]

# Benign v. 3, 4, and 5
# NbrOfPatches1 = 0
# NbrOfPatches2 = validation_count2
# NbrOfPatches3 = int(validation_count2 / 3)
# NbrOfPatches4 = int(validation_count2 / 3)
# NbrOfPatches5 = int(validation_count2 / 3)
# num_per_case = [None, None, 17, 12, 200]

# Benign v. 3
# NbrOfPatches1 = 0
# NbrOfPatches2 = validation_count2
# NbrOfPatches3 = validation_count2
# NbrOfPatches4 = 0
# NbrOfPatches5 = 0

# Benign v. 4
# NbrOfPatches1 = 0
# NbrOfPatches2 = validation_count2
# NbrOfPatches3 = 0
# NbrOfPatches4 = validation_count2
# NbrOfPatches5 = 0

# 3 v. 4 and 5
# NbrOfPatches1 = 0
# NbrOfPatches2 = 0
# NbrOfPatches3 = validation_count5 * 2
# NbrOfPatches4 = validation_count5
# NbrOfPatches5 = validation_count5
# num_per_case = [None, None, 34, 14, None]

# 3 v. 4
# NbrOfPatches1 = 0
# NbrOfPatches2 = 0
# NbrOfPatches3 = validation_count3
# NbrOfPatches4 = validation_count3
# NbrOfPatches5 = 0

# 4 v. 5
NbrOfPatches1 = 0
NbrOfPatches2 = 0
NbrOfPatches3 = 0
NbrOfPatches4 = validation_count5
NbrOfPatches5 = validation_count5
num_per_case = [None, None, None, 14, None]

for f in tqdm(validation_slideFolders):
    count1_num_per_case = 0
    count2_num_per_case = 0
    count3_num_per_case = 0
    count4_num_per_case = 0
    count5_num_per_case = 0
    for c in os.listdir(patches_path + f):
        if int(c) == 1 and count1 < NbrOfPatches1:
            for p in os.listdir(patches_path + f + "/" + c):
                if count1 < NbrOfPatches1:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_validation.append(img_data)
                    y_validation.append(0)
                    count1 += 1

                    count1_num_per_case += 1
                    if count1_num_per_case >= int((NbrOfPatches1 - count1) * num_per_case[0] / len(validation_slideFolders)):
                        break
                else:
                    break
        if int(c) == 2 and count2 < NbrOfPatches2:
            for p in os.listdir(patches_path + f + "/" + c):
                if count2 < NbrOfPatches2:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_validation.append(img_data)
                    y_validation.append(0)
                    count2 += 1

                    # count2_num_per_case += 1
                    # if count2_num_per_case >= int((NbrOfPatches2 - count2) * num_per_case[1] / len(validation_slideFolders)):
                    #     break
                else:
                    break
        if int(c) == 3 and count3 < NbrOfPatches3:
            for p in os.listdir(patches_path + f + "/" + c):
                if count3 < NbrOfPatches3:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_validation.append(img_data)
                    y_validation.append(0)
                    count3 += 1

                    count3_num_per_case += 1
                    if count3_num_per_case >= int((NbrOfPatches3 - count3) * num_per_case[2] / len(validation_slideFolders)):
                        break
                else:
                    break
        if int(c) == 4 and count4 < NbrOfPatches4:
            for p in os.listdir(patches_path + f + "/" + c):
                if count4 < NbrOfPatches4:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_validation.append(img_data)
                    y_validation.append(0)
                    count4 += 1

                    count4_num_per_case += 1
                    if count4_num_per_case >= int((NbrOfPatches4 - count4) * num_per_case[3] / len(validation_slideFolders)):
                        break
                else:
                    break
        if int(c) == 5 and count5 < NbrOfPatches5:
            for p in os.listdir(patches_path + f + "/" + c):
                if count5 < NbrOfPatches5:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_validation.append(img_data)
                    y_validation.append(1)
                    count5 += 1

                    # count5_num_per_case += 1
                    # if count5_num_per_case >= int((NbrOfPatches5 - count5) * num_per_case[4] / len(validation_slideFolders)):
                    #     break
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

def step_decay(epoch, lr):
    initial_lrate = 0.0002
    decay = 0.02

    return initial_lrate * (1 / (1 + decay * epoch))

now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
if not os.path.isdir("./model_result/" + dt_string):
    os.mkdir("./model_result/" + dt_string)
    
checkpoint_filepath = "./model_result/" + dt_string + "/checkpoint/"
# checkpoint_filepath = "./model_result/2021_03_23_22_21_32/checkpoint/"
if not os.path.isdir(checkpoint_filepath):
    os.mkdir(checkpoint_filepath)

lrs = tf.keras.callbacks.LearningRateScheduler(step_decay)
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
plt.savefig("./model_result/" + dt_string + '/Accuracy.png')
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['train', 'validation'])
plt.grid()
plt.savefig("./model_result/" + dt_string + '/Loss.png')
plt.close()

plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.legend(['train', 'validation'])
plt.grid()
plt.savefig("./model_result/" + dt_string + '/AUC.png')
plt.close()

"""
Save results
"""
f = open("./model_result/" + dt_string + "/testing_result.txt", "w")
f.write("y_train distribution: " + str(train_distribution) + "\n")
f.write("y_test distribution: " + str(test_distribution) + "\n")
f.write("y_validation distribution: " + str(validation_distribution) + "\n")
f.write("NbrOfAugmentationPatches: " + str(NbrOfAugmentation1) + " " + str(NbrOfAugmentation2) + " " + str(NbrOfAugmentation3) + " " + str(NbrOfAugmentation4) + " "+ str(NbrOfAugmentation5) + "\n")
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
"""
temp = np.equal(yhat, y_test)
for idx in trange(len(temp)):
    if temp[idx] == False:
        im = Image.fromarray((x_test[idx] * 255).astype(np.uint8))
        if y_test[idx] == 0:
            im.save("./model_result/" + dt_string + "/a_" + str(yhat[idx]) + "_" + str(idx) + ".tiff")
        elif y_test[idx] == 1:
            im.save("./model_result/" + dt_string + "/b_" + str(yhat[idx]) + "_" + str(idx) + ".tiff")
        elif y_test[idx] == 2:
            im.save("./model_result/" + dt_string + "/c_" + str(yhat[idx]) + "_" + str(idx) + ".tiff")
        elif y_test[idx] == 3:
            im.save("./model_result/" + dt_string + "/d_" + str(yhat[idx]) + "_" + str(idx) + ".tiff")
        elif y_test[idx] == 4:
            im.save("./model_result/" + dt_string + "/e_" + str(yhat[idx]) + "_" + str(idx) + ".tiff")