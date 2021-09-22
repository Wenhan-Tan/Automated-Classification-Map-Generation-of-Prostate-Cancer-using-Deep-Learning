#!/usr/bin/env python3
# coding: utf-8

"""
Author:
    Wenhan Tan
Date:
    2021/9

Description:
    This script first randomize slides first and split them into train (64%), test
    (20%), and validation (16%) sets. Then counts number of stroma, benign, pattern
    3, 4, and 5 patches and check whether each one of them contribute 64% to train,
    20% to test, and 16% to validation sets. If not, because each slide has different
    amount of patches, repeat the whole process again until percentages of different
    class patches matches the "64%-20%-16%" ratio. Finally, successfully train,
    test, and validation slide IDs will be saved in 3 csv files for future
    training purpose.

Input:
    Provided by this work:
        1) patches_level1_128.csv (from script *Extract_patches.py*)
        2) Patch files ("patches_path", from script *Extract_patches.py*)
        3) patches_level1_128_slideFolders.csv (from script *Extract_patches.py*)

Output:
    1) train_slideFolders.csv
    2) test_slideFolders.csv
    3) validation_slideFolders.csv

Usage:
    To reproduce results, simply run this script in terminal. Make sure you
    have all the python packages and input files ready.

    To use it on a different dataset, go through all the input files and make
    sure you have them in the same format for your dataset. Change input
    filepath and output filepath based on your file locations.
"""

import numpy as np
import tensorflow as tf
import os
from tqdm.notebook import tqdm

"""
Filepath
"""
# Input path
patches_info_path = "./prostate-cancer-grade-assessment/patches_level1_128.csv"
patches_path = "./prostate-cancer-grade-assessment/patches_level1_128/"
slideFolders_path = "./prostate-cancer-grade-assessment/patches_level1_128_slideFolders.csv"

# Output path
train_slideFolders_path = "./prostate-cancer-grade-assessment/train_slideFolders.csv"
test_slideFolders_path = "./prostate-cancer-grade-assessment/test_slideFolders.csv"
validation_slideFolders_path = "./prostate-cancer-grade-assessment/validation_slideFolders.csv"

"""
Read in data
"""
print("Reading in data")
patches_id = np.genfromtxt(patches_info_path, delimiter=",", dtype='str', usecols=0)
slideFolders = np.genfromtxt(slideFolders_path, delimiter=",", dtype='str', usecols=0)

"""
Split into train/test/validation based on slides
"""
print("Splitting into train/test/validation sets")
test_error = 1
validation_error = 1

# check if either test error or validation error is too large
while test_error > 0.05 or validation_error > 0.05:
    # keep shuffling slides until both test error and validation error are low
    np.random.shuffle(slideFolders)

    # train set gets the first 80%
    train_slideFolders = slideFolders[:int(np.ceil(len(slideFolders) * 8 / 10))]
    # test set gets the last 20%
    test_slideFolders = slideFolders[int(np.ceil(len(slideFolders) * 8 / 10)):]
    # validation set gets the last 20% from train set
    validation_slideFolders = train_slideFolders[int(np.ceil(len(train_slideFolders) * 8 / 10)):]
    # train set gets the first 80% from the original train set
    train_slideFolders = train_slideFolders[:int(np.ceil(len(train_slideFolders) * 8 / 10))]

    # for each train, test, and validation, there are 5 classes
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
Save into 3 csv files
"""
print("Saving 3 csv files")
np.savetxt(train_slideFolders_path, train_slideFolders, delimiter=",", fmt="%s")
np.savetxt(test_slideFolders_path, test_slideFolders, delimiter=",", fmt="%s")
np.savetxt(validation_slideFolders_path, validation_slideFolders, delimiter=",", fmt="%s")