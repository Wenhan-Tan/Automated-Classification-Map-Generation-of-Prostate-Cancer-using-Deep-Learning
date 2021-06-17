#!/usr/bin/env python3
# coding: utf-8

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
from tqdm import trange, tqdm
import csv

"""
Set up GPU
"""
# tf.compat.v1.disable_eager_execution()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)]
        )
    except RuntimeError as e:
        print(e)

################################## GLOBAL ###############################

# new_train_info_path = "./prostate-cancer-grade-assessment/new_train.csv"
# new_test_info_path = "./prostate-cancer-grade-assessment/new_test.csv"
# info_path = "./prostate-cancer-grade-assessment/train.csv"
# images_path = "./prostate-cancer-grade-assessment/train_images/"
# masks_path = "./prostate-cancer-grade-assessment/train_label_masks/"

# image_format = ".tiff"
# mask_format = "_mask.tiff"

# image_id = np.genfromtxt(info_path, delimiter=",", dtype='str',\
#                          skip_header=1, usecols=0)
# mask_id = np.char.add(image_id, mask_format)
# image_id = np.char.add(image_id, image_format)

# image_info = np.genfromtxt(info_path, delimiter=",", dtype='str',\
#                          skip_header=1, usecols=(1,2,3))

# cmap1 = mpl.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])
# cmap2 = mpl.colors.ListedColormap(['black', 'gray', 'red'])

# train_image_id = np.genfromtxt(new_train_info_path, delimiter=",", dtype='str', usecols=0)
# train_mask_id = np.genfromtxt(new_train_info_path, delimiter=",", dtype='str', usecols=1)
# train_image_info = np.genfromtxt(new_train_info_path, delimiter=",", dtype='str', usecols=(2,3,4))
# test_image_id = np.genfromtxt(new_test_info_path, delimiter=",", dtype='str', usecols=0)
# test_mask_id = np.genfromtxt(new_test_info_path, delimiter=",", dtype='str', usecols=1)
# test_image_info = np.genfromtxt(new_test_info_path, delimiter=",", dtype='str', usecols=(2,3,4))

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
DT inference
"""
def DT(x_test):
    temp = model_1v.predict(x_test)
    yhat_DT = np.argmax(temp, axis=1)
    yhat_DT_magnitude = np.max(temp, axis=1)
    
    if (yhat_DT == 1).sum() > 0:
        temp = model_2v.predict(x_test[yhat_DT == 1])
        yhat_temp = np.argmax(temp, axis=1)
        count = 0
        for idx in range(len(yhat_DT)):
            if yhat_DT[idx] == 1:
                if yhat_temp[count] == 0:
                    yhat_DT[idx] = 1
                else:
                    yhat_DT[idx] = 2
                yhat_DT_magnitude[idx] = np.max(temp, axis=1)[count]
                count += 1

    if (yhat_DT == 2).sum() > 0:
        temp = model_3v.predict(x_test[yhat_DT == 2])
        yhat_temp = np.argmax(temp, axis=1)
        count = 0
        for idx in range(len(yhat_DT)):
            if yhat_DT[idx] == 2:
                if yhat_temp[count] == 0:
                    yhat_DT[idx] = 2
                else:
                    yhat_DT[idx] = 3
                yhat_DT_magnitude[idx] = np.max(temp, axis=1)[count]
                count += 1
    
    if (yhat_DT == 3).sum() > 0:
        temp = model_4v.predict(x_test[yhat_DT == 3])
        yhat_temp = np.argmax(temp, axis=1)
        count = 0
        for idx in range(len(yhat_DT)):
            if yhat_DT[idx] == 3:
                if yhat_temp[count] == 0:
                    yhat_DT[idx] = 3
                else:
                    yhat_DT[idx] = 4
                yhat_DT_magnitude[idx] = np.max(temp, axis=1)[count]
                count += 1
                
    return yhat_DT, yhat_DT_magnitude

"""
MC inference
"""
def MC(x_test):
    temp = model_MC.predict(x_test)
    yhat_MC = np.argmax(temp, axis=1)
    yhat_DT_magnitude = np.max(temp, axis=1)
    return yhat_MC, yhat_DT_magnitude

"""
Save slidefolders classification result
"""
# new_train_info_path = "./prostate-cancer-grade-assessment/new_train.csv"
new_train_info_path = "./prostate-cancer-grade-assessment/new_test_kaggle_remove_ka.csv"
images_path = "./prostate-cancer-grade-assessment/train_images/"
masks_path = "./prostate-cancer-grade-assessment/train_label_masks/"

train_image_id = np.genfromtxt(new_train_info_path, delimiter=",", dtype='str', usecols=0)
train_image_mask_id = np.genfromtxt(new_train_info_path, delimiter=",", dtype='str', usecols=1)
train_image_fullInfo = np.genfromtxt(new_train_info_path, delimiter=",", dtype='str', usecols=(0, 1, 2, 3, 4))
test_slideFolders = np.genfromtxt("./prostate-cancer-grade-assessment/test_slideFolders.csv", delimiter=",", dtype='str', usecols=0)
train_slideFolders = np.genfromtxt("./prostate-cancer-grade-assessment/train_slideFolders.csv", delimiter=",", dtype='str', usecols=0)
validation_slideFolders = np.genfromtxt("./prostate-cancer-grade-assessment/validation_slideFolders.csv", delimiter=",", dtype='str', usecols=0)
kaggle_remove_ka_slideFolders = np.genfromtxt("./prostate-cancer-grade-assessment/kaggle_remove_ka_slideFolders.csv", delimiter=",", dtype='str', usecols=0)

level = 1
sz = 128
tile_size = 126
overlap = int((sz - tile_size) / 2)
# f = open("./prostate-cancer-grade-assessment/train_slideFolders_classification_tilesize_126.csv", "a+", newline="")
# f = open("./prostate-cancer-grade-assessment/validation_slideFolders_classification_tilesize_126.csv", "a+", newline="")
# f = open("./prostate-cancer-grade-assessment/test_slideFolders_classification_tilesize_126.csv", "a+", newline="")
f = open("./prostate-cancer-grade-assessment/kaggle_remove_ka_slideFolders_classification_tilesize_126.csv", "a+", newline="")
wr = csv.writer(f)

# for kaggle_remove_ka_slideFolders_idx in trange(len(kaggle_remove_ka_slideFolders)):
# for train_slideFolders_idx in trange(len(train_slideFolders)):
# for validation_slideFolders_idx in trange(len(validation_slideFolders)):
# for test_slideFolders_idx in trange(len(test_slideFolders)):
for kaggle_remove_ka_slideFolders_idx in trange(1000, len(kaggle_remove_ka_slideFolders)):
# for train_slideFolders_idx in trange(0, 2):
# for validation_slideFolders_idx in range(0, 1):
# for test_slideFolders_idx in range(531, 533):
    for new_train_idx in range(len(train_image_id)):
        if kaggle_remove_ka_slideFolders[kaggle_remove_ka_slideFolders_idx] in train_image_id[new_train_idx]:
        # if train_slideFolders[train_slideFolders_idx] in train_image_id[new_train_idx]:
        # if validation_slideFolders[validation_slideFolders_idx] in train_image_id[new_train_idx]:
        # if test_slideFolders[test_slideFolders_idx] in train_image_id[new_train_idx]:
            # H&E
            im = openslide.OpenSlide(images_path + train_image_id[new_train_idx])
            
            # Select tiles
            dpz = deepzoom.DeepZoomGenerator(im, tile_size=tile_size, overlap=overlap, limit_bounds=False)

            width = dpz.level_tiles[dpz.level_count - 3][0]
            height = dpz.level_tiles[dpz.level_count - 3][1]
            offset = int(np.ceil(overlap / tile_size))

            tiles1 = []
            for j in range(offset, dpz.level_tiles[dpz.level_count - 3][1] - offset):
                for i in range(offset, dpz.level_tiles[dpz.level_count - 3][0] - offset):
                    tiles1.append(np.asarray(dpz.get_tile(dpz.level_count - 3, (i, j))))
            try:
                tiles1 = np.stack(tiles1, axis=0)
            except:
                break
            
            im.close()
            
            temp_idx_list = []
            for p in tiles1:
                if p.sum() < (12533760 * 0.85):
                    temp_idx_list.append(True)
                else:
                    temp_idx_list.append(False)
            
            # DT
            predicted_mask_data_DT = [0] * len(temp_idx_list)
            temp_predicted_mask_data_DT, temp_predicted_mask_data_DT_magnitude = DT(tiles1[temp_idx_list] / 255.0)
            temp_predicted_mask_data_DT += 1
            count = 0
            for temp_idx in range(len(predicted_mask_data_DT)):
                if temp_idx_list[temp_idx] == True:
                    predicted_mask_data_DT[temp_idx] = temp_predicted_mask_data_DT[count]
                    count += 1
            
            # MC
            predicted_mask_data_MC = [0] * len(temp_idx_list)
            temp_predicted_mask_data_MC, temp_predicted_mask_data_MC_magnitude = MC(tiles1[temp_idx_list] / 255.0)
            temp_predicted_mask_data_MC += 1
            count = 0
            for temp_idx in range(len(predicted_mask_data_MC)):
                if temp_idx_list[temp_idx] == True:
                    predicted_mask_data_MC[temp_idx] = temp_predicted_mask_data_MC[count]
                    count += 1
            
            # Extract DT features
            temp1 = 0
            temp2 = 0
            temp3 = 0
            temp4 = 0
            temp5 = 0
            count = 0
            for i in predicted_mask_data_DT:
                if i == 1:
                    temp1 += temp_predicted_mask_data_DT_magnitude[count]
                elif i == 2:
                    temp2 += temp_predicted_mask_data_DT_magnitude[count]
                elif i == 3:
                    temp3 += temp_predicted_mask_data_DT_magnitude[count]
                elif i == 4:
                    temp4 += temp_predicted_mask_data_DT_magnitude[count]
                elif i == 5:
                    temp5 += temp_predicted_mask_data_DT_magnitude[count]
                
                if i != 0:
                    count += 1
            
            # if np.equal(predicted_mask_data_DT, 1).sum() > 0:
            #     temp1 = temp1 / np.equal(predicted_mask_data_DT, 1).sum()
            # if np.equal(predicted_mask_data_DT, 2).sum() > 0:
            #     temp2 = temp2 / np.equal(predicted_mask_data_DT, 2).sum()
            # if np.equal(predicted_mask_data_DT, 3).sum() > 0:
            #     temp3 = temp3 / np.equal(predicted_mask_data_DT, 3).sum()
            # if np.equal(predicted_mask_data_DT, 4).sum() > 0:
            #     temp4 = temp4 / np.equal(predicted_mask_data_DT, 4).sum()
            # if np.equal(predicted_mask_data_DT, 5).sum() > 0:
            #     temp5 = temp5 / np.equal(predicted_mask_data_DT, 5).sum()
            
            unique_DT = np.unique(predicted_mask_data_DT, return_counts=True)
            # length = np.sum(temp_idx_list)
            temp_list_DT = [0, 0, 0, 0, 0, temp1, temp2, temp3, temp4, temp5]
            for temp_idx in range(len(unique_DT[0])):
                # if int(unique_DT[0][temp_idx]) == 1:
                #     temp_list_DT[0] = unique_DT[1][temp_idx] / length
                # if int(unique_DT[0][temp_idx]) == 2:
                #     temp_list_DT[1] = unique_DT[1][temp_idx] / length
                # if int(unique_DT[0][temp_idx]) == 3:
                #     temp_list_DT[2] = unique_DT[1][temp_idx] / length
                # if int(unique_DT[0][temp_idx]) == 4:
                #     temp_list_DT[3] = unique_DT[1][temp_idx] / length
                # if int(unique_DT[0][temp_idx]) == 5:
                #     temp_list_DT[4] = unique_DT[1][temp_idx] / length
                if int(unique_DT[0][temp_idx]) == 1:
                    temp_list_DT[0] = unique_DT[1][temp_idx]
                if int(unique_DT[0][temp_idx]) == 2:
                    temp_list_DT[1] = unique_DT[1][temp_idx]
                if int(unique_DT[0][temp_idx]) == 3:
                    temp_list_DT[2] = unique_DT[1][temp_idx]
                if int(unique_DT[0][temp_idx]) == 4:
                    temp_list_DT[3] = unique_DT[1][temp_idx]
                if int(unique_DT[0][temp_idx]) == 5:
                    temp_list_DT[4] = unique_DT[1][temp_idx]
            
            x_list_DT = temp_list_DT
            y_list_DT = [train_image_fullInfo[new_train_idx][4]]
            
            # Extract MC features
            temp1 = 0
            temp2 = 0
            temp3 = 0
            temp4 = 0
            temp5 = 0
            count = 0
            for i in predicted_mask_data_MC:
                if i == 1:
                    temp1 += temp_predicted_mask_data_MC_magnitude[count]
                elif i == 2:
                    temp2 += temp_predicted_mask_data_MC_magnitude[count]
                elif i == 3:
                    temp3 += temp_predicted_mask_data_MC_magnitude[count]
                elif i == 4:
                    temp4 += temp_predicted_mask_data_MC_magnitude[count]
                elif i == 5:
                    temp5 += temp_predicted_mask_data_MC_magnitude[count]
                
                if i != 0:
                    count += 1
            
            # if np.equal(predicted_mask_data_MC, 1).sum() > 0:
            #     temp1 = temp1 / np.equal(predicted_mask_data_MC, 1).sum()
            # if np.equal(predicted_mask_data_MC, 2).sum() > 0:
            #     temp2 = temp2 / np.equal(predicted_mask_data_MC, 2).sum()
            # if np.equal(predicted_mask_data_MC, 3).sum() > 0:
            #     temp3 = temp3 / np.equal(predicted_mask_data_MC, 3).sum()
            # if np.equal(predicted_mask_data_MC, 4).sum() > 0:
            #     temp4 = temp4 / np.equal(predicted_mask_data_MC, 4).sum()
            # if np.equal(predicted_mask_data_MC, 5).sum() > 0:
            #     temp5 = temp5 / np.equal(predicted_mask_data_MC, 5).sum()
            
            unique_MC = np.unique(predicted_mask_data_MC, return_counts=True)
            temp_list_MC = [0, 0, 0, 0, 0, temp1, temp2, temp3, temp4, temp5]
            for temp_idx in range(len(unique_MC[0])):
                # if int(unique_MC[0][temp_idx]) == 1:
                #     temp_list_MC[0] = unique_MC[1][temp_idx] / length
                # if int(unique_MC[0][temp_idx]) == 2:
                #     temp_list_MC[1] = unique_MC[1][temp_idx] / length
                # if int(unique_MC[0][temp_idx]) == 3:
                #     temp_list_MC[2] = unique_MC[1][temp_idx] / length
                # if int(unique_MC[0][temp_idx]) == 4:
                #     temp_list_MC[3] = unique_MC[1][temp_idx] / length
                # if int(unique_MC[0][temp_idx]) == 5:
                #     temp_list_MC[4] = unique_MC[1][temp_idx] / length
                if int(unique_MC[0][temp_idx]) == 1:
                    temp_list_MC[0] = unique_MC[1][temp_idx]
                if int(unique_MC[0][temp_idx]) == 2:
                    temp_list_MC[1] = unique_MC[1][temp_idx]
                if int(unique_MC[0][temp_idx]) == 3:
                    temp_list_MC[2] = unique_MC[1][temp_idx]
                if int(unique_MC[0][temp_idx]) == 4:
                    temp_list_MC[3] = unique_MC[1][temp_idx]
                if int(unique_MC[0][temp_idx]) == 5:
                    temp_list_MC[4] = unique_MC[1][temp_idx]
            
            x_list_MC = temp_list_MC

            # slideFolders_classification = x_list_DT + x_list_MC + y_list_DT
            slideFolders_classification = temp_predicted_mask_data_DT.tolist() + temp_predicted_mask_data_DT_magnitude.tolist() + temp_predicted_mask_data_MC.tolist() + temp_predicted_mask_data_MC_magnitude.tolist() + y_list_DT
            wr.writerow(slideFolders_classification)
            break

f.close()