#!/usr/bin/env python3
# coding: utf-8

"""
Author:
    Wenhan Tan
Date:
    2021/9

Description:
    This script removes slides that have inconsistent labels and mask images.
    This is done by checking whether one of the first two largest number of
    pixels matches the first score in a label. Slides that have consistent labels
    and mask images will be saved in a csv file called new_train.csv. Slides
    missing their corresponding mask images are also removed. The csv file will
    be used later for training.

Input:
    Provided by Radboud:
        1) train.csv
        2) Radboud mask image path ("masks_path")

Output:
    1) new_train.csv ("new_train_path")

Usage:
    To reproduce results, simply run this script in terminal. Make sure you
    have all the python packages and input files ready.

    To use it on a different dataset, go through all the input files and make
    sure you have them in the same format for your dataset. Change input
    filepath and output filepath based on your file locations.
"""

# Remember to download and install "openslide"
# "ctypes" is imported for using "openslide" if "...cannot find library..." shows up
import ctypes
from ctypes.util import find_library
_lib = ctypes.cdll.LoadLibrary(find_library("./openslide-win64-20171122/bin/libopenslide-0.dll"))
import openslide
import os
import numpy as np
import csv
from tqdm.notebook import trange

def grade(tile):
    """
    This function takes in a mask patch and returns 2 lists. Score list is
    the percentage of each class pixels in the entire input patch. Relative
    score list is number of each class pixels except benign over number of
    benign pixels.
    """

    score = [] # percentage of each class pixels in the entire input patch
    relative_score = [] # number of each class pixels except benign over number of benign pixels
    tissue_count = 1

    for i in range(1, 6):
        count = 0
        for row in tile:
            comparsion = row == [i, 0, 0]
            count += comparsion.all(axis=1).sum()
        score.append(count / (len(tile) ** 2))
        if i == 1:
            if count > 0:
                tissue_count = count
            else:
                tissue_count = 1
        else:
            relative_score.append(count / tissue_count)
    return score, relative_score

def create_new_train(level=2):
    """
    This function checks whether each slide's grade label matches its mask image.
    Some slides have inconsistence between their labels and their masks. So either
    the label or the mask image is wrong. These slides will be removed from the
    new_train.csv. To check whether they match for a slide, this functions finds
    the first two classes with the largest number of pixels and if either one is
    the same as the first score in the label, this slide will be saved for use.
    For example, a slide with a label of 3 + 4, if the first or the second largest
    number of pixels is pattern 3, then this slide will be in new_train.csv. Slides
    missing their corresponding mask images are also removed.
    """
    
    good_image_id = []
    good_mask_id = []
    good_image_info = []

    for i in trange(len(image_id)):
        # Check if masks exist
        if os.path.isfile(masks_path + mask_id[i]):
            ifBad = 0
            # Check if provider is radboud
            if image_info[i][0] == 'radboud':
                # Read mask
                im = openslide.OpenSlide(masks_path + mask_id[i])
                im2 = im.read_region(location=(0, 0), level=level, \
                                      size=(im.level_dimensions[level][0],\
                                            im.level_dimensions[level][1]))
                im.close()
                mask_data = np.asarray(im2)[:, :, 0:3]

                # Count pixels
                score, relative_score = grade(mask_data)
                
                # Check if ISUP grade > 0
                if int(image_info[i][1]) > 0:
                    # Find Gleason score
                    first_score = int(image_info[i][2][0])
                    first_largest = relative_score.index(sorted(relative_score, reverse=True)[0]) + 2
                    second_largest = relative_score.index(sorted(relative_score, reverse=True)[1]) + 2
                    
                    # Check whether one of the two classes with the most pixel amount
                    # is the same as the first score in gleason grade
                    if first_largest == first_score or second_largest == first_score:
                        ifBad = 0
                    else:
                        ifBad = 1
                else:
                    if (np.argmax(score) + 1) <= 2:
                        ifBad = 0
                    else:
                        ifBad = 1
            else:
                ifBad = 1
        else:
            ifBad = 1
            
        if ifBad:
            pass
        else:
            good_image_id.append(image_id[i])
            good_mask_id.append(mask_id[i])
            good_image_info.append(image_info[i])
    
    return np.array([good_image_id]).T, np.array([good_mask_id]).T, np.array([good_image_info])[0]

"""
Filepath
"""
info_path = "./prostate-cancer-grade-assessment/train.csv"
masks_path = "./prostate-cancer-grade-assessment/train_label_masks/"

# Output new_train.csv path
new_train_path = "./prostate-cancer-grade-assessment/new_train.csv"

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

# Read in csv files
image_info = np.genfromtxt(info_path, delimiter=",", dtype='str',\
                         skip_header=1, usecols=(1,2,3))

"""
Clean data
"""
print("Cleaning data")
new_train_image_id, new_train_mask_id, new_train_image_info = create_new_train()

"""
Save into new_train.csv
"""
print("Saving data")
new_train_content = np.concatenate((new_train_image_id, new_train_mask_id, new_train_image_info), axis=1)
np.savetxt(new_train_path, new_train_content, delimiter=",", fmt="%s")