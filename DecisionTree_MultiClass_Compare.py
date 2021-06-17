#!/usr/bin/env python3
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tempfile
from tensorflow.keras import layers, models, applications
from sklearn.metrics import confusion_matrix
from PIL import Image
import os
from datetime import datetime
from tqdm import tqdm, trange
import seaborn as sn

"""
Set up GPU
"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
    except RuntimeError as e:
        print(e)

"""
Split into train/test based on cases
"""
patches_info_path = "./prostate-cancer-grade-assessment/patches_level1_128.csv"
patches_path = "./prostate-cancer-grade-assessment/patches_level1_128/"
slideFolders_path = "./prostate-cancer-grade-assessment/patches_level1_128_slideFolders.csv"

patches_id = np.genfromtxt(patches_info_path, delimiter=",", dtype='str', usecols=0)
slideFolders = np.genfromtxt(slideFolders_path, delimiter=",", dtype='str', usecols=0)

test_error = 1
validation_error = 1
while test_error > 0.05 or validation_error > 0.05:
    # np.random.shuffle(slideFolders)
    # train_slideFolders = slideFolders[:int(np.ceil(len(slideFolders) * 7 / 10))]
    # test_slideFolders = slideFolders[int(np.ceil(len(slideFolders) * 7 / 10)):]
    # validation_slideFolders = train_slideFolders[int(np.ceil(len(train_slideFolders) * 8 / 10)):]
    # train_slideFolders = train_slideFolders[:int(np.ceil(len(train_slideFolders) * 8 / 10))]

    train_slideFolders = np.genfromtxt("./prostate-cancer-grade-assessment/train_slideFolders.csv", delimiter=",", dtype='str', usecols=0)
    test_slideFolders = np.genfromtxt("./prostate-cancer-grade-assessment/test_slideFolders.csv", delimiter=",", dtype='str', usecols=0)
    validation_slideFolders = np.genfromtxt("./prostate-cancer-grade-assessment/validation_slideFolders.csv", delimiter=",", dtype='str', usecols=0)

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

    error1 = np.absolute(test_count1 / (train_count1 + test_count1 + validation_count1) - 0.3)
    error2 = np.absolute(test_count2 / (train_count2 + test_count2 + validation_count2) - 0.3)
    error3 = np.absolute(test_count3 / (train_count3 + test_count3 + validation_count3) - 0.3)
    error4 = np.absolute(test_count4 / (train_count4 + test_count4 + validation_count4) - 0.3)
    error5 = np.absolute(test_count5 / (train_count5 + test_count5 + validation_count5) - 0.3)
    
    error6 = np.absolute(validation_count1 / (train_count1 + test_count1 + validation_count1) - 0.14)
    error7 = np.absolute(validation_count2 / (train_count2 + test_count2 + validation_count2) - 0.14)
    error8 = np.absolute(validation_count3 / (train_count3 + test_count3 + validation_count3) - 0.14)
    error9 = np.absolute(validation_count4 / (train_count4 + test_count4 + validation_count4) - 0.14)
    error10 = np.absolute(validation_count5 / (train_count5 + test_count5 + validation_count5) - 0.14)
    
    test_error = error1 + error2 + error3 + error4 + error5
    validation_error = error6 + error7 + error8 + error9 + error10

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
NbrOfPatches1 = test_count5
NbrOfPatches2 = test_count5
NbrOfPatches3 = test_count5
NbrOfPatches4 = test_count5
NbrOfPatches5 = test_count5

# Stroma v. Benign + 3 + 4 + 5
# NbrOfPatches1 = test_count5 * 4
# NbrOfPatches2 = test_count5
# NbrOfPatches3 = test_count5
# NbrOfPatches4 = test_count5
# NbrOfPatches5 = test_count5

# Benign v. 3, 4, and 5
# NbrOfPatches1 = 0
# NbrOfPatches2 = test_count2
# NbrOfPatches3 = int(test_count2 / 3)
# NbrOfPatches4 = int(test_count2 / 3)
# NbrOfPatches5 = int(test_count2 / 3)

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

# 3 v. 4
# NbrOfPatches1 = 0
# NbrOfPatches2 = 0
# NbrOfPatches3 = test_count3
# NbrOfPatches4 = test_count3
# NbrOfPatches5 = 0

for f in tqdm(test_slideFolders):
    for c in os.listdir(patches_path + f):
        if int(c) == 1 and count1 < NbrOfPatches1:
            for p in os.listdir(patches_path + f + "/" + c):
                if count1 < NbrOfPatches1:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_test.append(img_data)
                    y_test.append(0)
                    count1 += 1
                else:
                    break
        if int(c) == 2 and count2 < NbrOfPatches2:
            for p in os.listdir(patches_path + f + "/" + c):
                if count2 < NbrOfPatches2:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_test.append(img_data)
                    y_test.append(1)
                    count2 += 1
                else:
                    break
        if int(c) == 3 and count3 < NbrOfPatches3:
            for p in os.listdir(patches_path + f + "/" + c):
                if count3 < NbrOfPatches3:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_test.append(img_data)
                    y_test.append(2)
                    count3 += 1
                else:
                    break
        if int(c) == 4 and count4 < NbrOfPatches4:
            for p in os.listdir(patches_path + f + "/" + c):
                if count4 < NbrOfPatches4:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_test.append(img_data)
                    y_test.append(3)
                    count4 += 1
                else:
                    break
        if int(c) == 5 and count5 < NbrOfPatches5:
            for p in os.listdir(patches_path + f + "/" + c):
                if count5 < NbrOfPatches5:
                    img = Image.open(patches_path + f + "/" + c + "/" + p)
                    img_data = np.asarray(img)
                    x_test.append(img_data)
                    y_test.append(4)
                    count5 += 1
                else:
                    break

print("NbrOfTestPatches:", count1, count2, count3, count4, count5)

"""
Print data info
"""
test_distribution = np.unique(y_test, return_counts=True)
print("y_test distribution:", test_distribution)

"""
Convert train data to numpy array, reshape and normalize
"""
x_test = np.stack(x_test, axis=0)
y_test = np.array(y_test)

x_test = x_test.reshape(len(x_test), 128, 128, 3) / 255.0
y_test = y_test.reshape(len(y_test), 1)

"""
Convert to one hot encoding
"""
y_test = tf.keras.utils.to_categorical(y_test, 5)

"""
Reconstruct model
"""
def reconstruct_model(checkpoint_filepath, ifMC):
    denseNet = applications.DenseNet201(include_top=False, weights=None, input_shape=(128, 128, 3),  pooling='avg')

    output = denseNet.output
    output = layers.Dropout(0.5)(output)
    if ifMC == 0:
        output = layers.Dense(2, activation='softmax')(output)
    elif ifMC == 1:
        output = layers.Dense(5, activation='softmax')(output)

    model = tf.keras.models.Model(denseNet.input, output)

    opt = tf.keras.optimizers.SGD(learning_rate=0.0002, momentum=0.7)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=opt,
                 loss=loss,
                 metrics=['accuracy','AUC'])
    model.load_weights(checkpoint_filepath)
    return model

"""
Model 1v setup
"""
print("Setting up model 1v")
checkpoint_filepath_1v = "./model_result/2021_01_21_22_44_16/checkpoint/"
# checkpoint_filepath_1v = "./model_result/2021_03_23_22_21_32/checkpoint/"
model_1v = reconstruct_model(checkpoint_filepath_1v, 0)

"""
Model 2v setup
"""
print("Setting up model 2v")
checkpoint_filepath_2v = "./model_result/2021_01_22_22_08_24/checkpoint/"
# checkpoint_filepath_2v = "./model_result/2021_03_24_20_58_22/checkpoint/"
model_2v = reconstruct_model(checkpoint_filepath_2v, 0)

"""
Model 3v setup
"""
print("Setting up model 3v")
checkpoint_filepath_3v = "./model_result/2021_01_23_18_59_44/checkpoint/"
# checkpoint_filepath_3v = "./model_result/2021_03_25_21_10_05/checkpoint/"
model_3v = reconstruct_model(checkpoint_filepath_3v, 0)

"""
Model 4v setup
"""
print("Setting up model 4v")
checkpoint_filepath_4v = "./model_result/2021_02_12_22_24_50/checkpoint/"
# checkpoint_filepath_4v = "./model_result/2021_03_26_21_11_33/checkpoint/"
model_4v = reconstruct_model(checkpoint_filepath_4v, 0)

"""
Model Multi-Class setup
"""
print("Setting up model MC")
checkpoint_filepath_MC = "./model_result/2021_02_14_21_02_23/checkpoint/"
# checkpoint_filepath_MC = "./model_result/2021_03_22_23_04_31/checkpoint/"
model_MC = reconstruct_model(checkpoint_filepath_MC, 1)

"""
Decision Tree result
"""
yhat_DT = []

print("Applying decision tree")
yhat_DT = model_1v.predict(x_test)
yhat_DT = np.argmax(yhat_DT, axis=1)

yhat_temp = model_2v.predict(x_test[yhat_DT == 1])
yhat_temp = np.argmax(yhat_temp, axis=1)
count = 0
for idx in trange(len(yhat_DT)):
    if yhat_DT[idx] == 1:
        if yhat_temp[count] == 0:
            yhat_DT[idx] = 1
        else:
            yhat_DT[idx] = 2
        count += 1

yhat_temp = model_3v.predict(x_test[yhat_DT == 2])
yhat_temp = np.argmax(yhat_temp, axis=1)
count = 0
for idx in trange(len(yhat_DT)):
    if yhat_DT[idx] == 2:
        if yhat_temp[count] == 0:
            yhat_DT[idx] = 2
        else:
            yhat_DT[idx] = 3
        count += 1

yhat_temp = model_4v.predict(x_test[yhat_DT == 3])
yhat_temp = np.argmax(yhat_temp, axis=1)
count = 0
for idx in trange(len(yhat_DT)):
    if yhat_DT[idx] == 3:
        if yhat_temp[count] == 0:
            yhat_DT[idx] = 3
        else:
            yhat_DT[idx] = 4
        count += 1

y_test = np.argmax(y_test, axis=1)
confMatrix = confusion_matrix(y_test, yhat_DT)
for i in confMatrix:
    print(i)

"""
Save DT result
"""
print("Saving DT result")
now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")

if not os.path.isdir("./DT_MC_result/" + dt_string):
    os.mkdir("./DT_MC_result/" + dt_string)
if not os.path.isdir("./DT_MC_result/" + dt_string + "/DT/"):
    os.mkdir("./DT_MC_result/" + dt_string + "/DT/")

temp = np.equal(yhat_DT, y_test)
for idx in trange(len(temp)):
    if temp[idx] == False:
        im = Image.fromarray((x_test[idx] * 255).astype(np.uint8))
        if y_test[idx] == 0:
            im.save("./DT_MC_result/" + dt_string + "/DT/" + "/a_" + str(yhat_DT[idx]) + "_" + str(idx) + ".tiff")
        elif y_test[idx] == 1:
            im.save("./DT_MC_result/" + dt_string + "/DT/" + "/b_" + str(yhat_DT[idx]) + "_" + str(idx) + ".tiff")
        elif y_test[idx] == 2:
            im.save("./DT_MC_result/" + dt_string + "/DT/" + "/c_" + str(yhat_DT[idx]) + "_" + str(idx) + ".tiff")
        elif y_test[idx] == 3:
            im.save("./DT_MC_result/" + dt_string + "/DT/" + "/d_" + str(yhat_DT[idx]) + "_" + str(idx) + ".tiff")
        elif y_test[idx] == 4:
            im.save("./DT_MC_result/" + dt_string + "/DT/" + "/e_" + str(yhat_DT[idx]) + "_" + str(idx) + ".tiff")

plt.figure(figsize=(10, 7))
plt.title("Decision Flow - Confusion Matrix")
ax = sn.heatmap(confMatrix,
                fmt="d",
                annot=True,
                annot_kws={"size":16})
ax.set_xticklabels(["stroma", "benign", "3", "4", "5"])
ax.set_yticklabels(["stroma", "benign", "3", "4", "5"])
ax.set(ylabel="True Label", xlabel="Predicted Label")
plt.savefig("./DT_MC_result/" + dt_string + "/DT/DT_confusionMatrix.jpg", dpi=300)

"""
Multi-Class Classification result
"""
print("Applying multi-class classification")
yhat_MC = model_MC.predict(x_test)
yhat_MC = np.argmax(yhat_MC, axis=1)
confMatrix = confusion_matrix(y_test, yhat_MC)
for i in confMatrix:
    print(i)

"""
Save MC result
"""
print("Saving MC result")
if not os.path.isdir("./DT_MC_result/" + dt_string + "/MC/"):
    os.mkdir("./DT_MC_result/" + dt_string + "/MC/")

temp = np.equal(yhat_MC, y_test)
for idx in trange(len(temp)):
    if temp[idx] == False:
        im = Image.fromarray((x_test[idx] * 255).astype(np.uint8))
        if y_test[idx] == 0:
            im.save("./DT_MC_result/" + dt_string + "/MC/" + "/a_" + str(yhat_MC[idx]) + "_" + str(idx) + ".tiff")
        elif y_test[idx] == 1:
            im.save("./DT_MC_result/" + dt_string + "/MC/" + "/b_" + str(yhat_MC[idx]) + "_" + str(idx) + ".tiff")
        elif y_test[idx] == 2:
            im.save("./DT_MC_result/" + dt_string + "/MC/" + "/c_" + str(yhat_MC[idx]) + "_" + str(idx) + ".tiff")
        elif y_test[idx] == 3:
            im.save("./DT_MC_result/" + dt_string + "/MC/" + "/d_" + str(yhat_MC[idx]) + "_" + str(idx) + ".tiff")
        elif y_test[idx] == 4:
            im.save("./DT_MC_result/" + dt_string + "/MC/" + "/e_" + str(yhat_MC[idx]) + "_" + str(idx) + ".tiff")

plt.figure(figsize=(10, 7))
plt.title("Multi-class - Confusion Matrix")
ax = sn.heatmap(confMatrix,
                fmt="d",
                annot=True,
                annot_kws={"size":16})
ax.set_xticklabels(["stroma", "benign", "3", "4", "5"])
ax.set_yticklabels(["stroma", "benign", "3", "4", "5"])
ax.set(ylabel="True Label", xlabel="Predicted Label")
plt.savefig("./DT_MC_result/" + dt_string + "/MC/MC_confusionMatrix.jpg", dpi=300)