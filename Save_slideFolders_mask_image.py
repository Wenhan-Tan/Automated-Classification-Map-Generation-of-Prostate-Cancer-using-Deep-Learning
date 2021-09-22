#!/usr/bin/env python3
# coding: utf-8

"""
Author:
    Wenhan Tan (wenhan.johnson.tan@hotmail.com)
Date:
    2021/9

Description:
    This script splits each slide into patches and classify each patch using
    5 trained models (4 for decision flow & 1 for multi-class). Then color each
    classified patch based on its result and merge them into a classification
    map. Classification maps are saved on your computer.

Input:
    Provided by Radboud:
        1) Radboud images and masks
        2) train.csv
    Provided by this work:
        1) new_train.csv (cleaned Radboud data, from script *Clean_data.py*)
        2) test_slideFolders.csv (from script *Train_test_validation_split.py*)
        3) 5 trained models weights (4 for decision flow & 1 for multi-class,
        from script *Train.py*)

Output:
    1) Classification maps (example filename: #ID_#tilesize_#GleasonGrade.jpg)

Usage:
    To reproduce results, simply run this script in terminal. Make sure you
    have all the python packages and input files ready.

    To use it on a different dataset, there are 2 ways:
        1) Download the trained weights from GitHub and use them in your code.
        2) Go through all the input files and make sure you have them in the 
           same format for your dataset. Change both input and output filepath
           based on your file locations.
"""

# Remember to download and install "openslide"
# "ctypes" is imported for using "openslide" if "...cannot find library..." shows up
import ctypes
from ctypes.util import find_library
_lib = ctypes.cdll.LoadLibrary(find_library("./openslide-win64-20171122/bin/libopenslide-0.dll"))
import openslide
from openslide import deepzoom
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from sklearn.metrics import confusion_matrix
from tqdm import trange
import time

"""
Set GPU memory limit (Change the limit based on your GPU)
"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            # Memory limit is 4.5G (RTX 2060 max is 6G)
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)]
        )
    except RuntimeError as e:
        print(e)

"""
Filepath (classification map path is written at the end of this script)
"""
# Path for "new_train.csv", cleaned data
# Provided by this work
new_train_info_path = "./prostate-cancer-grade-assessment/new_train.csv"

# Path for slide info including provider and grade
# Provided by Radboud
info_path = "./prostate-cancer-grade-assessment/train.csv"

# Path for H&E
# Provided by Radboud
images_path = "./prostate-cancer-grade-assessment/train_images/"

# Path for masks
# Provided by Radboud
masks_path = "./prostate-cancer-grade-assessment/train_label_masks/"

# Path for 5 trained models weights
# Provided by this work
checkpoint_filepath_1v = "./model_result/2021_01_21_22_44_16/checkpoint/"
checkpoint_filepath_2v = "./model_result/2021_01_22_22_08_24/checkpoint/"
checkpoint_filepath_3v = "./model_result/2021_01_23_18_59_44/checkpoint/"
checkpoint_filepath_4v = "./model_result/2021_02_12_22_24_50/checkpoint/"
checkpoint_filepath_MC = "./model_result/2021_02_14_21_02_23/checkpoint/"

# Path for slideFolders
# Provided by this work
test_slideFolders_path = "./prostate-cancer-grade-assessment/test_slideFolders.csv"

"""
Read in data
"""
print("Reading in data")

# ID for each slide
image_id = np.genfromtxt(info_path, delimiter=",", dtype='str', skip_header=1, usecols=0)

# Add ".tiff" and "_mask.tiff" at the end of "image_id" and "mask_id"
image_format = ".tiff"
mask_format = "_mask.tiff"
mask_id = np.char.add(image_id, mask_format)
image_id = np.char.add(image_id, image_format)

# Color map for Radboud
cmap1 = mpl.colors.ListedColormap(['white', 'gray', 'green', 'yellow', 'orange', 'red'])
# Color map for Karolinska
cmap2 = mpl.colors.ListedColormap(['white', 'gray', 'red'])

# Read in csv files
train_image_id = np.genfromtxt(new_train_info_path, delimiter=",", dtype='str', usecols=0)
train_mask_id = np.genfromtxt(new_train_info_path, delimiter=",", dtype='str', usecols=1)
train_image_fullInfo = np.genfromtxt(new_train_info_path, delimiter=",", dtype='str', usecols=(0, 1, 2, 3, 4))
test_slideFolders = np.genfromtxt(test_slideFolders_path, delimiter=",", dtype='str', usecols=0)

"""
Reconstruct model
"""
def reconstruct_model(checkpoint_filepath, ifMC):
    """
    This function reconstruct models with input trained models. The model is
    DenseNet 201 + DropOut layer. If it is a multi-class model, there are 5
    nodes at the softmax layer. If it is a decision flow model, there are only 2
    nodes at the softmax layer. Trained model is then returned.
    """

    # Use DenseNet 201
    denseNet = applications.DenseNet201(include_top=False, weights=None, input_shape=(128, 128, 3),  pooling='avg')

    output = denseNet.output
    output = layers.Dropout(0.5)(output) # Add Dropout layer
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
    model.load_weights(checkpoint_filepath) # Load trained weights
    return model

"""
Model 1v setup / 1st decision flow model
"""
print("Setting up model 1 decision flow")
model_1v = reconstruct_model(checkpoint_filepath_1v, 0)
print("Done")

"""
Model 2v setup / 2nd decision flow model
"""
print("Setting up model 2 decision flow")
model_2v = reconstruct_model(checkpoint_filepath_2v, 0)
print("Done")

"""
Model 3v setup / 3rd decision flow model
"""
print("Setting up model 3 decision flow")
model_3v = reconstruct_model(checkpoint_filepath_3v, 0)
print("Done")

"""
Model 4v setup / 4th decision flow model
"""
print("Setting up model 4 decision flow")
model_4v = reconstruct_model(checkpoint_filepath_4v, 0)
print("Done")

"""
Model Multi-Class setup / multi-class model
"""
print("Setting up model multi-class")
model_MC = reconstruct_model(checkpoint_filepath_MC, 1)
print("Done")

"""
This function takes in x data and outputs y data with probabilities for
decision flow method
"""
def DF(x_test):
    temp = model_1v.predict(x_test)
    yhat_DF = np.argmax(temp, axis=1)
    yhat_DF_magnitude = np.max(temp, axis=1)
    
    if (yhat_DF == 1).sum() > 0:
        temp = model_2v.predict(x_test[yhat_DF == 1])
        yhat_temp = np.argmax(temp, axis=1)
        count = 0
        for idx in range(len(yhat_DF)):
            if yhat_DF[idx] == 1:
                if yhat_temp[count] == 0:
                    yhat_DF[idx] = 1
                else:
                    yhat_DF[idx] = 2
                yhat_DF_magnitude[idx] = np.max(temp, axis=1)[count]
                count += 1

    if (yhat_DF == 2).sum() > 0:
        temp = model_3v.predict(x_test[yhat_DF == 2])
        yhat_temp = np.argmax(temp, axis=1)
        count = 0
        for idx in range(len(yhat_DF)):
            if yhat_DF[idx] == 2:
                if yhat_temp[count] == 0:
                    yhat_DF[idx] = 2
                else:
                    yhat_DF[idx] = 3
                yhat_DF_magnitude[idx] = np.max(temp, axis=1)[count]
                count += 1
    
    if (yhat_DF == 3).sum() > 0:
        temp = model_4v.predict(x_test[yhat_DF == 3])
        yhat_temp = np.argmax(temp, axis=1)
        count = 0
        for idx in range(len(yhat_DF)):
            if yhat_DF[idx] == 3:
                if yhat_temp[count] == 0:
                    yhat_DF[idx] = 3
                else:
                    yhat_DF[idx] = 4
                yhat_DF_magnitude[idx] = np.max(temp, axis=1)[count]
                count += 1
                
    return yhat_DF, yhat_DF_magnitude

"""
This function takes in x data and outputs y data with probabilities for
multi-class method
"""
def MC(x_test):
    temp = model_MC.predict(x_test)
    yhat_MC = np.argmax(temp, axis=1)
    yhat_MC_magnitue = np.max(temp, axis=1)
    return yhat_MC, yhat_MC_magnitue

"""
MAIN CODE:
The following code generates classification maps and output them with provided 
H&E and mask images side by side
"""
print("Outputting classification maps")

# Each slide has three levels opened by openslide: 0, 1 & 2
# 0 is too small and 2 is too large
level = 1
sz = 128 # Don't change it. Size of each patch

# Size of squares to be colored. Lower it can increase resolution but takes a
# lot more time and memory
tile_size = 40
overlap = int((sz - tile_size) / 2) # Size of overlapping areas between patches

for test_slideFolders_idx in trange(len(test_slideFolders)):
    for new_train_idx in range(len(train_image_id)):
        if test_slideFolders[test_slideFolders_idx] in train_image_id[new_train_idx]:
            # Put H&E, mask, decision flow & multi-class side by side
            fig, ax = plt.subplots(1, 4, figsize=(30, 30))

            # Save H&E
            im = openslide.OpenSlide(images_path + train_image_id[new_train_idx])
            im2 = im.read_region(location=(0, 0), level=level, \
                              size=(im.level_dimensions[level][0],\
                                    im.level_dimensions[level][1]))

            width = im.level_dimensions[level][0]
            height = im.level_dimensions[level][1]

            # 0:3 because the 4th element, alpha, is not being used
            data = np.array(im2)[:, :, 0:3]
            ax[0].imshow(data)

            # Split into patches
            dpz = deepzoom.DeepZoomGenerator(im, tile_size=tile_size, overlap=overlap, limit_bounds=False)
            width = dpz.level_tiles[dpz.level_count - 3][0]
            height = dpz.level_tiles[dpz.level_count - 3][1]
            offset = int(np.ceil(overlap / tile_size))

            tiles1 = []
            for j in range(offset, dpz.level_tiles[dpz.level_count - 3][1] - offset):
                for i in range(offset, dpz.level_tiles[dpz.level_count - 3][0] - offset):
                    tiles1.append(np.asarray(dpz.get_tile(dpz.level_count - 3, (i, j))))
            tiles1 = np.stack(tiles1, axis=0)
            
            im.close()

            # Filter empty patches
            temp_idx_list = []
            for p in tiles1:
                # Select tiles with cells based on sum of pixels
                # 0.85: lower it can increase the amount of patches being classified
                if p.sum() < (255 * 3 * 128 * 128 * 0.85):
                    temp_idx_list.append(True)
                else:
                    temp_idx_list.append(False)

            # Save decision flow classification map
            predicted_mask_data_DF = [[255, 255, 255]] * len(temp_idx_list) # White
            temp_predicted_mask_data_DF, temp_predicted_mask_data_DF_magnitude = DF(tiles1[temp_idx_list] / 255.0)
            temp_predicted_mask_data_DF += 1
            count = 0
            for temp_idx in range(len(predicted_mask_data_DF)):
                if temp_idx_list[temp_idx] == True:
                    if temp_predicted_mask_data_DF[count] == 1:
                        # Gray
                        predicted_mask_data_DF[temp_idx] = (np.array([127, 127, 127]) * temp_predicted_mask_data_DF_magnitude[count])
                    elif temp_predicted_mask_data_DF[count] == 2:
                        # Green
                        predicted_mask_data_DF[temp_idx] = (np.array([0, 255, 0]) * temp_predicted_mask_data_DF_magnitude[count])
                    elif temp_predicted_mask_data_DF[count] == 3:
                        # Yellow
                        predicted_mask_data_DF[temp_idx] = (np.array([255, 255, 0]) * temp_predicted_mask_data_DF_magnitude[count])
                    elif temp_predicted_mask_data_DF[count] == 4:
                        # Orange
                        predicted_mask_data_DF[temp_idx] = (np.array([255, 165, 0]) * temp_predicted_mask_data_DF_magnitude[count])
                    elif temp_predicted_mask_data_DF[count] == 5:
                        # Red
                        predicted_mask_data_DF[temp_idx] = (np.array([255, 0, 0]) * temp_predicted_mask_data_DF_magnitude[count])
                    count += 1

            # Resize to original size
            predicted_mask_data_DF = np.array(predicted_mask_data_DF).reshape(height - 2 * offset, width - 2 * offset, 3).astype(np.uint8)
            ax[2].imshow(predicted_mask_data_DF)

            # Save multi-class classification map
            predicted_mask_data_MC = [[255, 255, 255]] * len(temp_idx_list) # White
            temp_predicted_mask_data_MC, temp_predicted_mask_data_MC_magnitude = MC(tiles1[temp_idx_list] / 255.0)
            temp_predicted_mask_data_MC += 1
            count = 0
            for temp_idx in range(len(predicted_mask_data_MC)):
                if temp_idx_list[temp_idx] == True:
                    if temp_predicted_mask_data_MC[count] == 1:
                        # Gray
                        predicted_mask_data_MC[temp_idx] = (np.array([127, 127, 127]) * temp_predicted_mask_data_MC_magnitude[count])
                    elif temp_predicted_mask_data_MC[count] == 2:
                        # Green
                        predicted_mask_data_MC[temp_idx] = (np.array([0, 255, 0]) * temp_predicted_mask_data_MC_magnitude[count])
                    elif temp_predicted_mask_data_MC[count] == 3:
                        # Yellow
                        predicted_mask_data_MC[temp_idx] = (np.array([255, 255, 0]) * temp_predicted_mask_data_MC_magnitude[count])
                    elif temp_predicted_mask_data_MC[count] == 4:
                        # Orange
                        predicted_mask_data_MC[temp_idx] = (np.array([255, 165, 0]) * temp_predicted_mask_data_MC_magnitude[count])
                    elif temp_predicted_mask_data_MC[count] == 5:
                        # Red
                        predicted_mask_data_MC[temp_idx] = (np.array([255, 0, 0]) * temp_predicted_mask_data_MC_magnitude[count])
                    count += 1

            # Resize to original size
            predicted_mask_data_MC = np.array(predicted_mask_data_MC).reshape(height - 2 * offset, width - 2 * offset, 3).astype(np.uint8)
            ax[3].imshow(predicted_mask_data_MC)
            
            # Save original mask
            im = openslide.OpenSlide(masks_path + train_mask_id[new_train_idx])
            im2 = im.read_region(location=(0, 0), level=level, \
                              size=(im.level_dimensions[level][0],\
                                    im.level_dimensions[level][1]))
            im.close()

            mask_data = np.array(im2)[:, :, 0:3]
            # Check if data provider is Radboud or Karolinska
            if train_image_fullInfo[new_train_idx][2] == 'karolinska':
                ax[1].imshow(mask_data[:, :, 0], cmap=cmap2, vmin=0, vmax=2, interpolation='nearest')
            else:
                ax[1].imshow(mask_data[:, :, 0], cmap=cmap1, vmin=0, vmax=5, interpolation='nearest')
            
            # Path for classification map
            # Naming: #ID_#tilesize_#GleasonGrade.jpg
            classification_map_path = "./prostate-cancer-grade-assessment/test_slideFolders_result/" + test_slideFolders[test_slideFolders_idx] + "_" + str(tile_size) + "_" + train_image_fullInfo[new_train_idx][4] + ".jpg"
            
            # Save entire image
            plt.savefig(classification_map_path)
            plt.close()
            break