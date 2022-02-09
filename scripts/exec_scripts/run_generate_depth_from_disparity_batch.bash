#!/bin/bash

# This script is used to generate disparity map from depth map.


# --------------
# Adjust these parameters
# --------------

# SESSIONS="ALL"
# SESSIONS="01007"
SESSIONS=" 01005 01010 01015 01020 02005 02010 02015 02020 "

BASELINE="0.6"
FX="480.0" # 
MAX_DISPARITY="192"

# SOURCE_DATASET_BASE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_sample4_00/model_ensemble_epoch_32_e034/cityenv_wb_test_unc/"
SOURCE_DATASET_BASE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01/model_ensemble_epoch_50_e012/cityenv_wb/"
TARGET_DEPTH_FOLDER_NAME="img_depth_pred"
SOURCE_DISPARITY_FOLDER_NAME="disp_pred"
# -----------------

if [ $SESSIONS == "ALL" ]; then
  pushd $SOURCE_DATASET_BASE_PATH
  # Get the list of all folders in the current directory
  folders=$(ls -d */ | sed 's/\///g')
  popd
  SESSIONS_LIST=$folders
else
  SESSIONS_LIST="$SESSIONS"
fi

# Go through all the folders in the source dataset and generate disparity maps
# for each folder.
for folder in $SESSIONS_LIST; do

    # Skip if it is not a directory.
    if [ ! -d ${SOURCE_DATASET_BASE_PATH}/${folder} ]; then
        echo "WARNING: No directory found for ${SOURCE_DATASET_BASE_PATH}/${folder}"
        continue
    fi

    # # Only process if the folder name starts with "03"
    # if [[ $folder != 04* ]]; then
    #     continue
    # fi

    echo "Processing folder: $folder"
    TARGET_DEPTH_FOLDER="$SOURCE_DATASET_BASE_PATH/$folder/$TARGET_DEPTH_FOLDER_NAME"
    SOURCE_DISPARITY_FOLDER="$SOURCE_DATASET_BASE_PATH/$folder/$SOURCE_DISPARITY_FOLDER_NAME"


    python ../generate_depth_from_disparity.py \
    --baseline=$BASELINE \
    --fx=$FX \
    --depth_folder=$TARGET_DEPTH_FOLDER \
    --disparity_folder=$SOURCE_DISPARITY_FOLDER \
    --max_disparity=$MAX_DISPARITY

done
