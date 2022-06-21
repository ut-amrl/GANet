#!/bin/bash

# This script is used to generate disparity map from depth map.


# --------------
# Adjust these parameters
# --------------

# SESSIONS="ALL"
# SESSIONS="01007"
# SESSIONS=" 01005 01010 01015 01020 02005 02010 02015 02020 "

# test_01.list 
# SESSIONS=" 01007 01012 01017 01022 01027 01032 02007 02012 02017 02022 02027 02032 "

# valid_01.list 
# SESSIONS=" 01005 01010 01015 01020 02005 02010 02015 02020 "

# training_01.list 
# SESSIONS=" 01006 01008 01009 01011 01013 01014 01016 01018 01019 01021 01023 01024 01025 01026 01028 01029 01030 01031 01033 01034 01035 01036 02006 02008 02009 02011 02013 02014 02016 02018 02019 02021 02023 02024 02025 02026 02028 02029 02030 02031 02033 02034 02035 02036 "

# testing_OOD_01.list
# SESSIONS=" 04005 04006 04007 04008 04009 04010 04011 04012 04013 04014 04015 04016 04017 04018 04019 04020 04021 04022 04023 04024 04025 04026 04027 04028 04029 04030 04031 04032 04033 04034 04035 04036 03005 03006 03007 03008 03009 03010 03011 03012 03013 03014 03015 03016 03017 03018 03019 03020 03021 03022 03023 03024 03025 03026 03027 03028 03029 03030 03031 03032 03033 03034 03035 03036 "

# Data from the Neighborhood Environment
# testing_OOD_N_01.list
SESSIONS=" 01000 01001 01002 01003 "


BASELINE="0.6"
FX="480.0" # 
MAX_DISPARITY="192"

# SOURCE_DATASET_BASE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_sample4_00/model_ensemble_epoch_32_e034/cityenv_wb_test_unc/"
SOURCE_DATASET_BASE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model_ensemble_epoch_14_e012/neighborhood_wb/"
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
