#!/bin/bash

# This script is used run calibrate_uncertainty_estimates.py

# -------- Ensemble model ganet_deep_airsim_01
GT_BASE_PATH="/robodata/srabiee/AirSim_IVSLAM/cityenv_wb"
PRED_BASE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01/model_ensemble_epoch_50_e012/cityenv_wb"
TRAINING_LIST='../../lists/airsim_wb_training_01.list'
VALIDATION_LIST='../../lists/airsim_wb_validation_01.list'
# OUTPUT_SAVE_PATH="/robodata/user_data/srabiee/results/ipr/nn_models/ganet_deep_airsim_01/model_ensemble_epoch_50_e012_calib_unc"
OUTPUT_SAVE_PATH="/robodata/user_data/srabiee/results/ipr/nn_models/ganet_deep_airsim_01/model_ensemble_epoch_50_e012_calib_unc_2" # large batch size

IMAGE_HEIGHT="300"
IMAGE_WIDTH="480"


mkdir -p $OUTPUT_SAVE_PATH

CUDA_VISIBLE_DEVICES="0" \
python ../training/calibrate_uncertainty_estimates.py \
--pred_data_path=$PRED_BASE_PATH \
--gt_data_path=$GT_BASE_PATH \
--save_path=$OUTPUT_SAVE_PATH \
--image_height=$IMAGE_HEIGHT \
--image_width=$IMAGE_WIDTH \
--num_workers=0 \
--num_epochs=100 \
--batch_size=60 \
--use_gpu=True \
--training_list=$TRAINING_LIST \
--val_list=$VALIDATION_LIST \
--subsample_factor="0.2" 


exit 0