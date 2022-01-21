#!/bin/bash

# This script is used run evaluate_depth_predictions.py

SOURCE_RGB_FOLDER="/robodata/srabiee/AirSim_IVSLAM/cityenv_wb/01007/img_left"
GT_DEPTH_FOLDER="/robodata/srabiee/AirSim_IVSLAM/cityenv_wb/01007/img_depth"
PRED_DEPTH_FOLDER="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_sample4_00/model_ensemble_epoch_32_e034/cityenv_wb/01007/img_depth_pred"

# OUTPUT_ERR_VIS_FOLDER="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_sample4_00/model_ensemble_epoch_32_e034/cityenv_wb/01008/depth_pred_err_vis_abs1.5_rel_0.3"
# OUTPUT_ERR_VIS_FOLDER="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_sample4_00/model_ensemble_epoch_32_e034/cityenv_wb/01008/depth_pred_err_vis_abs1.0_rel_0.1"
# OUTPUT_ERR_VIS_FOLDER="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_sample4_00/model_ensemble_epoch_32_e034/cityenv_wb/01008/depth_pred_err_vis_abs1.0_rel_0.2"
OUTPUT_ERR_VIS_FOLDER="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_sample4_00/model_ensemble_epoch_32_e034/cityenv_wb/01007/depth_pred_err_vis_abs1.0_rel_0.2_maxRange30"

MAX_RANGE="30.0"
MIN_RANGE="1.0"
DEPTH_ERR_THRESH_ABS="1.0" # 1.0 , 1.5
DEPTH_ERR_THRESH_REL="0.2" # 0.1 , 0.3
# IMAGE_HEIGHT="300"
# IMAGE_WIDTH="480"
IMAGE_HEIGHT="600"
IMAGE_WIDTH="960"

CUDA_VISIBLE_DEVICES="7" \
python ../evaluation/evaluate_depth_predictions.py \
--rgb_img_folder=$SOURCE_RGB_FOLDER \
--pred_depth_folder=$PRED_DEPTH_FOLDER \
--gt_depth_folder=$GT_DEPTH_FOLDER \
--output_err_vis_folder=$OUTPUT_ERR_VIS_FOLDER \
--max_range=$MAX_RANGE \
--min_range=$MIN_RANGE \
--depth_err_thresh_abs=$DEPTH_ERR_THRESH_ABS \
--depth_err_thresh_rel=$DEPTH_ERR_THRESH_REL \
--image_height=$IMAGE_HEIGHT \
--image_width=$IMAGE_WIDTH 