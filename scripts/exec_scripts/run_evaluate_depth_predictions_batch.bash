#!/bin/bash

# This script is used run evaluate_depth_predictions.py

# airsim_wb_test_01.list 
SESSIONS=" 01007 01012 01017 01022 01027 01032 02007 02012 02017 02022 02027 02032 "

SOURCE_DATASET_BASE_PATH="/robodata/srabiee/AirSim_IVSLAM/cityenv_wb/"
DEPTH_PREDICTIONS_BASE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01/model_ensemble_epoch_50_e012/cityenv_wb/"
# OUTPUT_ERR_VIS_FOLDER="depth_pred_err_vis_abs1.0_rel_0.1_maxRange30"
# OUTPUT_BINARY_ERR_VIS_FOLDER="depth_pred_err_binary_vis_abs1.0_rel_0.1_maxRange30"
OUTPUT_ERR_VIS_FOLDER="depth_pred_err_vis_abs1.0_rel_0.1_maxRange30_ds"
OUTPUT_BINARY_ERR_VIS_FOLDER="depth_pred_err_binary_vis_abs1.0_rel_0.1_maxRange30_ds"


# -------- Ensemble model ganet_deep_airsim_01
SOURCE_RGB_FOLDER="img_left"
GT_DEPTH_FOLDER="img_depth"
PRED_DEPTH_FOLDER="img_depth_pred"


MAX_RANGE="30.0"
MIN_RANGE="1.0"
DEPTH_ERR_THRESH_ABS="1.0" # 1.0 , 1.5
DEPTH_ERR_THRESH_REL="0.1" # 0.1 , 0.3
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

  mkdir -p $OUTPUT_ERR_VIS_FOLDER
  mkdir -p $OUTPUT_BINARY_ERR_VIS_FOLDER_PATH

  CUDA_VISIBLE_DEVICES="0" \
  python ../evaluation/evaluate_depth_predictions.py \
  --rgb_img_folder=$SOURCE_RGB_FOLDER_PATH \
  --pred_depth_folder=$PRED_DEPTH_FOLDER_PATH \
  --gt_depth_folder=$GT_DEPTH_FOLDER_PATH \
  --output_err_vis_folder=$OUTPUT_ERR_VIS_FOLDER_PATH \
  --output_binary_err_vis_folder=$OUTPUT_BINARY_ERR_VIS_FOLDER_PATH \
  --max_range=$MAX_RANGE \
  --min_range=$MIN_RANGE \
  --depth_err_thresh_abs=$DEPTH_ERR_THRESH_ABS \
  --depth_err_thresh_rel=$DEPTH_ERR_THRESH_REL \
  --image_height=$IMAGE_HEIGHT \
  --image_width=$IMAGE_WIDTH 


done