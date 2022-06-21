#!/bin/bash

# This script is used run evaluate_depth_predictions.py

# # training_01.list
# SESSIONS=" 01006 01008 01009 01011 01013 01014 01016 01018 01019 01021 01023 01024 01025 01026 01028 01029 01030 01031 01033 01034 01035 01036 02006 02008 02009 02011 02013 02014 02016 02018 02019 02021 02023 02024 02025 02026 02028 02029 02030 02031 02033 02034 02035 02036 "
# # training_01.list Part 1
# SESSIONS=" 01006 01008 01009 01011 01013 01014 01016 01018 01019 01021 01023 01024 01025 01026 01028 01029 01030 01031 01033 01034 01035 01036 "
# # training_01.list Part 2
# SESSIONS=" 02006 02008 02009 02011 02013 02014 02016 02018 02019 02021 02023 02024 02025 02026 02028 02029 02030 02031 02033 02034 02035 02036 "

# # validation_01.list 
# SESSIONS=" 01005 01010 01015 01020 02005 02010 02015 02020 "

# # # test_01.list 
# SESSIONS=" 01007 01012 01017 01022 01027 01032 02007 02012 02017 02022 02027 02032 "

# testing_OOD_01.list
# SESSIONS=" 04005 04006 04007 04008 04009 04010 04011 04012 04013 04014 04015 04016 04017 04018 04019 04020 04021 04022 04023 04024 04025 04026 04027 04028 04029 04030 04031 04032 04033 04034 04035 04036 03005 03006 03007 03008 03009 03010 03011 03012 03013 03014 03015 03016 03017 03018 03019 03020 03021 03022 03023 03024 03025 03026 03027 03028 03029 03030 03031 03032 03033 03034 03035 03036 "
# # testing_OOD_01.list part 1
# SESSIONS=" 04005 04006 04007 04008 04009 04010 04011 04012 04013 04014 04015 04016 04017 04018 04019 04020 04021 04022 04023 04024 04025 04026 04027 04028 04029 04030 04031 04032 04033 04034 04035 04036 "
# testing_OOD_01.list part 2
# SESSIONS=" 03005 03006 03007 03008 03009 03010 03011 03012 03013 03014 03015 03016 03017 03018 03019 03020 03021 03022 03023 03024 03025 03026 03027 03028 03029 03030 03031 03032 03033 03034 03035 03036 "

# Data from the Neighborhood Environment
# testing_OOD_N_01.list
SESSIONS=" 01000 01001 01002 01003 "


# SOURCE_DATASET_BASE_PATH="/robodata/srabiee/AirSim_IVSLAM/cityenv_wb/"
# DEPTH_PREDICTIONS_BASE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01/model_ensemble_epoch_50_e012/cityenv_wb/"
# DEPTH_PREDICTIONS_BASE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model_ensemble_epoch_14_e012_tmp/cityenv_wb/"
# DEPTH_PREDICTIONS_BASE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model0_r0.0_mcdropoutOff_epoch_14/cityenv_wb/"

SOURCE_DATASET_BASE_PATH="/robodata/user_data/srabiee/AirSim_BNN/neighborhood_wb/"
DEPTH_PREDICTIONS_BASE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model0_r0.0_mcdropoutOff_epoch_14/neighborhood_wb/"

OUTPUT_ERR_VIS_FOLDER="depth_pred_err_vis_abs1.0_rel_0.2_maxRange30_ds"
OUTPUT_BINARY_ERR_VIS_FOLDER="depth_pred_err_binary_vis_abs1.0_rel_0.2_maxRange30_ds"
OUTPUT_DEPTH_ERR_LABEL_FOLDER="depth_err_labels_abs1.0_rel_0.2_maxRange30_ds"
OUTPUT_VALID_PIXEL_MASK_FOLDER="valid_pixel_mask_abs1.0_rel_0.2_maxRange30_ds"

# -------- Ensemble model ganet_deep_airsim_01
SOURCE_RGB_FOLDER="img_left"
GT_DEPTH_FOLDER="img_depth"
PRED_DEPTH_FOLDER="img_depth_pred"


MAX_RANGE="30.0"
MIN_RANGE="1.0"
DEPTH_ERR_THRESH_ABS="1.0" # 1.0 , 1.5
DEPTH_ERR_THRESH_REL="0.2" # 0.1 , 0.3
IMAGE_HEIGHT="300"
IMAGE_WIDTH="480"
# IMAGE_HEIGHT="600" 
# IMAGE_WIDTH="960" # both ground truth and predicted depth images are resized to this size


# Go through all the folders in the source dataset and generate disparity maps
# for each folder.
for folder in $SESSIONS; do

  # Skip if it is not a directory.
  if [ ! -d ${SOURCE_DATASET_BASE_PATH}/${folder} ]; then
    echo "WARNING: No directory found for ${SOURCE_DATASET_BASE_PATH}/${folder}"
    continue
  fi
  if [ ! -d ${DEPTH_PREDICTIONS_BASE_PATH}/${folder} ]; then
    echo "WARNING: No directory found for ${DEPTH_PREDICTIONS_BASE_PATH}/${folder}"
    continue
  fi

  echo "Processing folder: $folder"

  SOURCE_RGB_FOLDER_PATH="$SOURCE_DATASET_BASE_PATH/$folder/$SOURCE_RGB_FOLDER"
  GT_DEPTH_FOLDER_PATH="$SOURCE_DATASET_BASE_PATH/$folder/$GT_DEPTH_FOLDER"
  PRED_DEPTH_FOLDER_PATH="$DEPTH_PREDICTIONS_BASE_PATH/$folder/$PRED_DEPTH_FOLDER"
  OUTPUT_ERR_VIS_FOLDER_PATH="$DEPTH_PREDICTIONS_BASE_PATH/$folder/$OUTPUT_ERR_VIS_FOLDER"
  OUTPUT_BINARY_ERR_VIS_FOLDER_PATH="$DEPTH_PREDICTIONS_BASE_PATH/$folder/$OUTPUT_BINARY_ERR_VIS_FOLDER"
  OUTPUT_DEPTH_ERR_LABELS_FOLDER_PATH="$DEPTH_PREDICTIONS_BASE_PATH/$folder/$OUTPUT_DEPTH_ERR_LABEL_FOLDER"
  OUTPUT_VALID_PIXEL_MASK_FOLDER_PATH="$DEPTH_PREDICTIONS_BASE_PATH/$folder/$OUTPUT_VALID_PIXEL_MASK_FOLDER"

  mkdir -p $OUTPUT_ERR_VIS_FOLDER
  mkdir -p $OUTPUT_BINARY_ERR_VIS_FOLDER_PATH
  mkdir -p $OUTPUT_DEPTH_ERR_LABELS_FOLDER_PATH
  mkdir -p $OUTPUT_VALID_PIXEL_MASK_FOLDER_PATH

  CUDA_VISIBLE_DEVICES="0" \
  python ../evaluation/evaluate_depth_predictions.py \
  --rgb_img_folder=$SOURCE_RGB_FOLDER_PATH \
  --pred_depth_folder=$PRED_DEPTH_FOLDER_PATH \
  --gt_depth_folder=$GT_DEPTH_FOLDER_PATH \
  --output_err_vis_folder=$OUTPUT_ERR_VIS_FOLDER_PATH \
  --output_binary_err_vis_folder=$OUTPUT_BINARY_ERR_VIS_FOLDER_PATH \
  --output_depth_err_labels_folder=$OUTPUT_DEPTH_ERR_LABELS_FOLDER_PATH \
  --output_valid_pixel_mask_folder=$OUTPUT_VALID_PIXEL_MASK_FOLDER_PATH \
  --max_range=$MAX_RANGE \
  --min_range=$MIN_RANGE \
  --depth_err_thresh_abs=$DEPTH_ERR_THRESH_ABS \
  --depth_err_thresh_rel=$DEPTH_ERR_THRESH_REL \
  --image_height=$IMAGE_HEIGHT \
  --image_width=$IMAGE_WIDTH 


done