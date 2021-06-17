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
from sklearn.metrics import confusion_matrix
from tqdm import trange, tqdm
import time

"""
Set up GPU
"""
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

new_train_info_path = "./prostate-cancer-grade-assessment/new_train.csv"
new_test_info_path = "./prostate-cancer-grade-assessment/new_test.csv"
info_path = "./prostate-cancer-grade-assessment/train.csv"
images_path = "./prostate-cancer-grade-assessment/train_images/"
masks_path = "./prostate-cancer-grade-assessment/train_label_masks/"

image_format = ".tiff"
mask_format = "_mask.tiff"

image_id = np.genfromtxt(info_path, delimiter=",", dtype='str',\
                         skip_header=1, usecols=0)
mask_id = np.char.add(image_id, mask_format)
image_id = np.char.add(image_id, image_format)

image_info = np.genfromtxt(info_path, delimiter=",", dtype='str',\
                         skip_header=1, usecols=(1,2,3))

cmap1 = mpl.colors.ListedColormap(['white', 'gray', 'green', 'yellow', 'orange', 'red'])
cmap2 = mpl.colors.ListedColormap(['white', 'gray', 'red'])

train_image_id = np.genfromtxt(new_train_info_path, delimiter=",", dtype='str', usecols=0)
train_mask_id = np.genfromtxt(new_train_info_path, delimiter=",", dtype='str', usecols=1)
train_image_info = np.genfromtxt(new_train_info_path, delimiter=",", dtype='str', usecols=(2,3,4))
test_image_id = np.genfromtxt(new_test_info_path, delimiter=",", dtype='str', usecols=0)
test_mask_id = np.genfromtxt(new_test_info_path, delimiter=",", dtype='str', usecols=1)
test_image_info = np.genfromtxt(new_test_info_path, delimiter=",", dtype='str', usecols=(2,3,4))

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
# checkpoint_filepath_1v = "./model_result/2021_01_13_20_44_02/checkpoint/"
model_1v = reconstruct_model(checkpoint_filepath_1v, 0)

"""
Model 2v setup
"""
print("Setting up model 2v")
checkpoint_filepath_2v = "./model_result/2021_01_22_22_08_24/checkpoint/"
# checkpoint_filepath_2v = "./model_result/2021_03_24_20_58_22/checkpoint/"
# checkpoint_filepath_2v = "./model_result/2021_01_14_21_10_54/checkpoint/"
model_2v = reconstruct_model(checkpoint_filepath_2v, 0)

"""
Model 3v setup
"""
print("Setting up model 3v")
checkpoint_filepath_3v = "./model_result/2021_01_23_18_59_44/checkpoint/"
# checkpoint_filepath_3v = "./model_result/2021_03_25_21_10_05/checkpoint/"
# checkpoint_filepath_3v = "./model_result/2021_01_16_21_59_18/checkpoint/"
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
# checkpoint_filepath_MC = "./model_result/2021_01_11_20_41_16/checkpoint/"
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
    yhat_MC_magnitue = np.max(temp, axis=1)
    return yhat_MC, yhat_MC_magnitue

"""
Display predicted mask data
"""
new_train_info_path = "./prostate-cancer-grade-assessment/new_train.csv"
images_path = "./prostate-cancer-grade-assessment/train_images/"
masks_path = "./prostate-cancer-grade-assessment/train_label_masks/"

train_image_id = np.genfromtxt(new_train_info_path, delimiter=",", dtype='str', usecols=0)
train_image_mask_id = np.genfromtxt(new_train_info_path, delimiter=",", dtype='str', usecols=1)
train_image_fullInfo = np.genfromtxt(new_train_info_path, delimiter=",", dtype='str', usecols=(0, 1, 2, 3, 4))
test_slideFolders = np.genfromtxt("./prostate-cancer-grade-assessment/test_slideFolders.csv", delimiter=",", dtype='str', usecols=0)
train_slideFolders = np.genfromtxt("./prostate-cancer-grade-assessment/train_slideFolders.csv", delimiter=",", dtype='str', usecols=0)
validation_slideFolders = np.genfromtxt("./prostate-cancer-grade-assessment/validation_slideFolders.csv", delimiter=",", dtype='str', usecols=0)

level = 1
sz = 128
tile_size = 40
overlap = int((sz - tile_size) / 2)
x_list_DT = []
y_list_DT = []
x_list_MC = []
y_list_MC = []
misclassified = [3, 6, 7, 10, 11, 12, 13, 17, 19, 21, 22, 23, 25, 27, 29, 30, 31, 33, 35,
37, 39, 40, 43, 45, 46, 49, 53, 54, 56, 59, 61, 63, 64, 68, 69, 70, 71, 73,
74, 83, 84, 88, 90, 93, 96, 99, 100, 101, 104, 105, 106, 107, 110, 111, 112,
113, 116, 118, 121, 124, 125, 127, 128, 129, 130, 131, 134, 136, 138, 140, 143,
145, 147, 148, 149, 152, 154, 158, 162, 163, 169, 170, 172, 175, 176, 177, 178,
179, 180, 184, 187, 189, 190, 193, 198, 199, 200, 202, 205, 206, 208, 209, 211,
212, 213, 220, 226, 227, 228, 231, 235, 237, 239, 240, 242, 245, 247, 248, 249,
251, 253, 257, 259, 261, 262, 265, 266, 268, 272, 273, 275, 276, 277, 279, 282,
284, 286, 287, 290, 292, 294, 299, 300, 301, 303, 304, 305, 309, 311, 313, 315,
318, 322, 323, 334, 335,
338, 345, 348, 349, 350, 352, 354, 356, 359, 362, 363, 364, 365, 366, 368, 369,
375, 378, 380, 382, 383, 386, 388, 392, 393, 395, 397, 398, 400, 403, 404, 412,
414, 417, 418, 421, 425, 426, 428, 429, 433, 434, 435, 436, 438, 439, 443, 448,
449, 450, 451, 454, 455, 457, 459, 463, 468, 470, 472, 474, 475, 476, 477,
478, 481, 485, 487, 488, 490, 492, 494, 495, 497, 501, 502, 505, 506, 508, 511,
512, 514, 516, 517, 521, 523, 524, 527, 528, 529, 530, 533, 534, 535, 540, 544,
548, 551, 553, 556, 557, 559, 560, 562, 563, 566, 574, 575, 577, 578, 579, 580,
581, 582, 584, 585, 586, 587, 589, 591, 592, 596, 597, 598, 599, 601, 603,
604, 605, 609, 612, 613, 615, 617, 618, 623, 626, 627, 628, 629, 630, 631, 632,
635, 640, 642, 644,645, 652, 654, 657, 659, 660, 661, 662, 667, 669, 678, 679,
683, 684, 686, 688, 694, 695, 696, 697, 699,
702, 704, 706, 708, 712, 714, 715, 718, 721, 722, 725, 727, 729, 731, 732, 733,
735, 736, 738, 741, 742, 744, 745, 747, 749, 753, 754, 755, 757, 758, 759, 761,
762, 763, 764, 768, 769, 770, 772, 773, 776,
781, 786, 789, 791, 792, 795, 798, 799, 801, 802, 803, 809, 810, 814, 815, 820,
821, 822, 824, 825, 826, 829, 830, 832, 834, 836, 839, 840, 843, 846, 847, 849,
856, 857, 858, 860, 863, 868, 869, 871, 875,
877, 881, 882, 884, 885, 887, 889, 890, 892, 893, 894, 895, 897, 901, 908, 910,
911, 914, 915, 923, 924, 926, 931, 932, 933, 935, 936, 937, 938, 939, 940, 943,
944, 946, 947, 949, 953, 954, 957, 960, 962, 963, 964, 967, 969, 970, 972, 974,
977, 979, 982, 984, 985, 987, 988, 989, 991, 992, 998, 1001, 1002, 1003, 1006,
1007, 1008, 1011, 1015, 1016, 1018, 1019, 1022, 1024, 1028, 1029, 1030, 1031, 1032,
1033, 1034, 1035, 1036,
1037, 1038, 1040, 1041, 1044, 1045, 1046, 1048, 1050, 1051, 1053, 1055, 1056,
1061, 1062, 1065, 1066, 1067, 1071, 1074, 1079, 1080, 1081, 1084, 1085, 1086,
1087, 1088, 1089, 1091, 1095, 1098, 1100, 1101, 1104,
1107, 1108, 1113, 1115, 1117, 1120, 1124, 1125, 1134, 1137, 1139, 1140, 1141,
1143, 1144, 1145, 1148, 1150, 1152, 1155, 1156, 1157, 1160, 1162, 1166, 1169,
1170, 1173, 1174, 1176, 1182, 1183, 1185, 1186, 1188, 1191, 1192, 1193, 1194,
1196, 1202, 1203, 1206, 1210, 1213, 1215, 1218, 1219, 1227, 1228, 1229, 1232,
1234, 1235, 1239, 1240, 1241, 1242, 1246, 1247, 1249, 1250, 1252, 1253, 1254,
1255, 1256, 1258, 1259, 1260]

# for train_slideFolders_idx in trange(len(train_slideFolders)):
# for validation_slideFolders_idx in trange(len(validation_slideFolders)):
# for test_slideFolders_idx in trange(len(test_slideFolders)):
# for train_slideFolders_idx in trange(335, 345):
for test_slideFolders_idx in trange(400, 405):
# for validation_slideFolders_idx in trange(36, 37):
    for new_train_idx in range(len(train_image_id)):
        # if train_slideFolders[train_slideFolders_idx] in train_image_id[new_train_idx]:
        # if validation_slideFolders[validation_slideFolders_idx] in train_image_id[new_train_idx]:
        if test_slideFolders[test_slideFolders_idx] in train_image_id[new_train_idx]:
            # start_time = time.time()
            fig, ax = plt.subplots(1, 4, figsize=(30, 30))
            
            # H&E
            im = openslide.OpenSlide(images_path + train_image_id[new_train_idx])
            im2 = im.read_region(location=(0, 0), level=level, \
                              size=(im.level_dimensions[level][0],\
                                    im.level_dimensions[level][1]))

            width = im.level_dimensions[level][0]
            height = im.level_dimensions[level][1]

            data = np.array(im2)[:, :, 0:3]
            ax[0].imshow(data)
            
            # Select tiles
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

            temp_idx_list = []
            for p in tiles1:
                if p.sum() < (12533760 * 0.85):
                    temp_idx_list.append(True)
                else:
                    temp_idx_list.append(False)

            # DT
            predicted_mask_data_DT = [[255, 255, 255]] * len(temp_idx_list)
            temp_predicted_mask_data_DT, temp_predicted_mask_data_DT_magnitude = DT(tiles1[temp_idx_list] / 255.0)
            temp_predicted_mask_data_DT += 1
            count = 0
            for temp_idx in range(len(predicted_mask_data_DT)):
                if temp_idx_list[temp_idx] == True:
                    if temp_predicted_mask_data_DT[count] == 1:
                        predicted_mask_data_DT[temp_idx] = (np.array([127, 127, 127]) * temp_predicted_mask_data_DT_magnitude[count])
                    elif temp_predicted_mask_data_DT[count] == 2:
                        predicted_mask_data_DT[temp_idx] = (np.array([0, 255, 0]) * temp_predicted_mask_data_DT_magnitude[count])
                    elif temp_predicted_mask_data_DT[count] == 3:
                        predicted_mask_data_DT[temp_idx] = (np.array([255, 255, 0]) * temp_predicted_mask_data_DT_magnitude[count])
                    elif temp_predicted_mask_data_DT[count] == 4:
                        predicted_mask_data_DT[temp_idx] = (np.array([255, 165, 0]) * temp_predicted_mask_data_DT_magnitude[count])
                    elif temp_predicted_mask_data_DT[count] == 5:
                        predicted_mask_data_DT[temp_idx] = (np.array([255, 0, 0]) * temp_predicted_mask_data_DT_magnitude[count])
                    count += 1

            predicted_mask_data_DT = np.array(predicted_mask_data_DT).reshape(height - 2 * offset, width - 2 * offset, 3).astype(np.uint8)
#             predicted_mask_data_DT = np.array(predicted_mask_data_DT).reshape(int(height / 128), int(width / 128), 3).astype(np.uint8)
#             predicted_mask_data_DT = cv2.resize(predicted_mask_data_DT, (int(width / 128) * 128, int(height / 128) * 128), interpolation = cv2.INTER_NEAREST)
            ax[2].imshow(predicted_mask_data_DT)

            # MC
            predicted_mask_data_MC = [[255, 255, 255]] * len(temp_idx_list)
            temp_predicted_mask_data_MC, temp_predicted_mask_data_MC_magnitude = MC(tiles1[temp_idx_list] / 255.0)
            temp_predicted_mask_data_MC += 1
            count = 0
            for temp_idx in range(len(predicted_mask_data_MC)):
                if temp_idx_list[temp_idx] == True:
                    if temp_predicted_mask_data_MC[count] == 1:
                        predicted_mask_data_MC[temp_idx] = (np.array([127, 127, 127]) * temp_predicted_mask_data_MC_magnitude[count])
                    elif temp_predicted_mask_data_MC[count] == 2:
                        predicted_mask_data_MC[temp_idx] = (np.array([0, 255, 0]) * temp_predicted_mask_data_MC_magnitude[count])
                    elif temp_predicted_mask_data_MC[count] == 3:
                        predicted_mask_data_MC[temp_idx] = (np.array([255, 255, 0]) * temp_predicted_mask_data_MC_magnitude[count])
                    elif temp_predicted_mask_data_MC[count] == 4:
                        predicted_mask_data_MC[temp_idx] = (np.array([255, 165, 0]) * temp_predicted_mask_data_MC_magnitude[count])
                    elif temp_predicted_mask_data_MC[count] == 5:
                        predicted_mask_data_MC[temp_idx] = (np.array([255, 0, 0]) * temp_predicted_mask_data_MC_magnitude[count])
                    count += 1
                    
            predicted_mask_data_MC = np.array(predicted_mask_data_MC).reshape(height - 2 * offset, width - 2 * offset, 3).astype(np.uint8)
#             predicted_mask_data_MC = np.array(predicted_mask_data_MC).reshape(int(height / 128), int(width / 128), 3).astype(np.uint8)
#             predicted_mask_data_MC = cv2.resize(predicted_mask_data_MC, (int(width / 128) * 128, int(height / 128) * 128), interpolation = cv2.INTER_NEAREST)
            ax[3].imshow(predicted_mask_data_MC)
            
            # Mask
            im = openslide.OpenSlide(masks_path + train_image_mask_id[new_train_idx])
            im2 = im.read_region(location=(0, 0), level=level, \
                              size=(im.level_dimensions[level][0],\
                                    im.level_dimensions[level][1]))
            im.close()

            mask_data = np.array(im2)[:, :, 0:3]
            if train_image_fullInfo[new_train_idx][2] == 'karolinska':
                ax[1].imshow(mask_data[:, :, 0], cmap=cmap2, vmin=0, vmax=2, interpolation='nearest')
            else:
                ax[1].imshow(mask_data[:, :, 0], cmap=cmap1, vmin=0, vmax=5, interpolation='nearest')

            # plt.savefig("./prostate-cancer-grade-assessment/train_slideFolders_result/" + train_slideFolders[train_slideFolders_idx] + "_" + str(tile_size) + ".jpg")
            plt.savefig("./prostate-cancer-grade-assessment/test_slideFolders_result/misclassified/" + test_slideFolders[test_slideFolders_idx] + "_" + str(tile_size) + "_" + train_image_fullInfo[new_train_idx][4] + ".jpg")
            # plt.savefig("./prostate-cancer-grade-assessment/validation_slideFolders_result/" + validation_slideFolders[validation_slideFolders_idx] + "_" + str(tile_size) + ".jpg")
            plt.close()
            # print("--- %s seconds ---" % (time.time() - start_time))
            break