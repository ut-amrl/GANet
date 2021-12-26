#!/bin/bash

# This script is used to generate disparity map from depth map.

BASELINE="0.6"
FX="480.0"

SOURCE_DATASET_BASE_PATH="/robodata/srabiee/AirSim_IVSLAM/cityenv_wb/"
SOURCE_DEPTH_FOLDER_NAME="img_depth"
TARGET_DISPARITY_FOLDER_NAME="img_disp"
TARGET_DISPARITY_COLOR_FOLDER_NAME="img_disp_color"

# Go through all the folders in the source dataset and generate disparity maps
# for each folder.
for folder in $(ls $SOURCE_DATASET_BASE_PATH); do
    # Skip if it is not a directory.
    if [ ! -d "$SOURCE_DATASET_BASE_PATH/$folder" ]; then
        continue
    fi

    # Only process if the folder name starts with "03"
    if [[ $folder != 04* ]]; then
        continue
    fi

    echo "Processing folder: $folder"
    SOURCE_DEPTH_FOLDER="$SOURCE_DATASET_BASE_PATH/$folder/$SOURCE_DEPTH_FOLDER_NAME"
    TARGET_DISPARITY_FOLDER="$SOURCE_DATASET_BASE_PATH/$folder/$TARGET_DISPARITY_FOLDER_NAME"
    TARGET_DISPARITY_COLOR_FOLDER="$SOURCE_DATASET_BASE_PATH/$folder/$TARGET_DISPARITY_COLOR_FOLDER_NAME"
    python ../generate_disparity_from_depth.py \
    --baseline=$BASELINE \
    --fx=$FX \
    --depth_folder=$SOURCE_DEPTH_FOLDER \
    --disparity_folder=$TARGET_DISPARITY_FOLDER \
    --disparity_color_folder=$TARGET_DISPARITY_COLOR_FOLDER

done
