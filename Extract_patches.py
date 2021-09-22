#!/usr/bin/env python3
# coding: utf-8

"""
Author:
    Wenhan Tan
Date:
    2021/9

Description:
    This script splits each slide into patches and store them based on their
    class. The way to determine class is by counting pixels of each class from
    mask patches. For example, if a mask patch is completely green, its
    corresponding slide patch will be in benign class. Patches that contain more
    than one color will not be used. Also, patches are saved under their
    original slide first and then saved based on their class. Saved patches will
    be used later for train/test/valiation split by another script
    (Train_test_validation_split.csv).

Input:
    Provided by Radboud:
        1) Radboud slides and masks ("images_patch" & "masks_path")
    
    Provided by this work:
        1) new_train.csv (cleaned Radboud data, from script *Clean_data.py*)

Output:
    1) Patch files (example filename: /#slideID/#classNumber/#patchFilename.tiff)
    2) A csv file containing patch filenames and class
    (example filename: /prostate-cancer-grade-assessment/patches_level1_128.csv)
    3) A csv file containing slide IDs of extracted patches
    (example filename: /prostate-cancer-grade-assessment/patches_level1_128_slideFolders.csv)

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
from openslide import deepzoom
import os
import numpy as np
import csv
from PIL import Image
from tqdm.notebook import trange

def if_clean(tile):
    """
    This function returns true if the input patch does not contain black
    areas. The input patch is a slide patch.
    """

    for row in tile:
        comparsion = row < 10
        if comparsion.sum() > 0:
            return False
    return True

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

def extract_patches(img_id, msk_id, level=1, sz=128):
    """
    This function splits each slide into patches and save them onto your
    computer. Each patch is saved based on its class (stroma, benign, pattern 3,
    4, and 5). Patches from the same slide is under the same slide id folder. A
    csv file containing each patch's filename and class is generated. A csv file
    containing slides IDs of extracted patches is also generated.
    """

    patches = [] # Store patch filename and class
    
    # Here 128 doesn't work for some reasons, so 126 is chosen. No overlapping
    # between patches. Lower this number can generating patches with overlapping
    # areas.
    tile_size = 126
    overlap = int((sz - tile_size) / 2) # Size of overlapping areas between patches

    # Output csv path for patch filename and class
    csv_path = "./prostate-cancer-grade-assessment/patches_level1_" + str(sz) + ".csv"
    
    for i in trange(len(img_id)):
        # Read slides
        im = openslide.OpenSlide(images_path + img_id[i])
        
        dpz = deepzoom.DeepZoomGenerator(im, tile_size=tile_size, overlap=overlap, limit_bounds=False)
        width = dpz.level_tiles[dpz.level_count - 3][0]
        height = dpz.level_tiles[dpz.level_count - 3][1]
        offset = int(np.ceil(overlap / tile_size))
        
        # Store slide tiles into a dictionary
        tiles1 = {}
        count = 0
        for j in range(offset, dpz.level_tiles[dpz.level_count - 3][1] - 1 - offset):
            for k in range(offset, dpz.level_tiles[dpz.level_count - 3][0] - 1 - offset):
                tiles1[count] = np.asarray(dpz.get_tile(dpz.level_count - 3, (k, j)))
                count += 1
        
        im.close()
        
        # Read masks
        im = openslide.OpenSlide(masks_path + msk_id[i])
        
        dpz = deepzoom.DeepZoomGenerator(im, tile_size=tile_size, overlap=overlap, limit_bounds=False)
        width = dpz.level_tiles[dpz.level_count - 3][0]
        height = dpz.level_tiles[dpz.level_count - 3][1]
        offset = int(np.ceil(overlap / tile_size))
        
        # Store mask tiles into a dictionary
        tiles2 = {}
        count = 0
        for j in range(offset, dpz.level_tiles[dpz.level_count - 3][1] - 1 - offset):
            for k in range(offset, dpz.level_tiles[dpz.level_count - 3][0] - 1 - offset):
                tiles2[count] = np.asarray(dpz.get_tile(dpz.level_count - 3, (k, j)))
                count += 1
        
        im.close()

        if len(tiles1) > 0:
            # Rank slide tiles by counting color pixels on mask tiles
            tiles2 = {k: v for k, v in sorted(tiles2.items(), key=lambda item: item[1].sum(), reverse=True)}
            
            iteration = 0
            # Score mask tiles
            for tile_idx in tiles2.keys():
                score, relative_score = grade(tiles2[tile_idx])
                
                # Find class
                # Patches contain more than 1 color are not used
                tile_score = -1
                if np.sum(score) == 1:
                    if np.sort(relative_score)[:3].sum() == 0:
                        if score[np.argmax(relative_score) + 1] > 0.4:
                            tile_score = np.argmax(relative_score) + 2
                        elif score[0] > 0.7:
                            tile_score = 1

                if tile_score >= 1:
                    if if_clean(tiles1[tile_idx]):
                        # Save patches
                        im = Image.fromarray(tiles1[tile_idx])
                        temp1 = "./prostate-cancer-grade-assessment/patches_level1_" + \
                                str(sz) + "/"
                        temp2 = img_id[i].split(".")[0] + "/"
                        temp3 = str(tile_score) + "/"

                        if not os.path.isdir(temp1):
                            os.mkdir(temp1)
                        
                        if not os.path.isdir(temp1 + temp2):
                            os.mkdir(temp1 + temp2)
                        
                        if not os.path.isdir(temp1 + temp2 + temp3):
                            os.mkdir(temp1 + temp2 + temp3)
                        
                        # Filename example: /#slideID/#classNumber/#patchFilename.tiff
                        im.save(temp1 + temp2 + temp3 + str(iteration) + img_id[i])
                        
                        # Append patch filenamd and class to csv
                        patches.append([str(iteration) + img_id[i], str(tile_score)])

                iteration += 1

    # Save csv file containing patch filename and class
    np.savetxt(csv_path, patches, delimiter=",", fmt="%s")

    # Save csv file containing slide IDs of extracted patches
    folders = os.listdir(temp1)
    csv_content = []
    for folder in folders:
        if os.path.isdir(folder):
            csv_content.append(folder)
    np.savetxt(patches_level1_128_slideFolders_path, csv_content, delimiter=",", fmt="%s")

"""
Filepath
"""
images_path = "./prostate-cancer-grade-assessment/train_images/"
masks_path = "./prostate-cancer-grade-assessment/train_label_masks/"
new_train_info_path = "./prostate-cancer-grade-assessment/new_train.csv"

# Output csv path for slide IDs
patches_level1_128_slideFolders_path = "./prostate-cancer-grade-assessment/patches_level1_128_slideFolders.csv"

"""
Read in data
"""
print("Reading in data")
train_image_id = np.genfromtxt(new_train_info_path, delimiter=",", dtype='str', usecols=0)
train_mask_id = np.genfromtxt(new_train_info_path, delimiter=",", dtype='str', usecols=1)

"""
Extract patches and save filenames into a csv
"""
print("Extracting patches")
extract_patches(train_image_id, train_mask_id, sz=128)