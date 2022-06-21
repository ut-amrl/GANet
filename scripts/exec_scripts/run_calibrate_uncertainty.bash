#!/bin/bash

# This script is used run calibrate_uncertainty_estimates.py

# # -------- Ensemble model ganet_deep_airsim_01
# GT_BASE_PATH="/robodata/srabiee/AirSim_IVSLAM/cityenv_wb"
# PRED_BASE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01/model_ensemble_epoch_50_e012/cityenv_wb"
# TRAINING_LIST='../../lists/airsim_wb_training_01.list'
# VALIDATION_LIST='../../lists/airsim_wb_validation_01.list'
# # OUTPUT_SAVE_PATH="/robodata/user_data/srabiee/results/ipr/nn_models/ganet_deep_airsim_01/model_ensemble_epoch_50_e012_calib_unc"
# OUTPUT_SAVE_PATH="/robodata/user_data/srabiee/results/ipr/nn_models/ganet_deep_airsim_01/model_ensemble_epoch_50_e012_calib_unc_2" # large batch size

# # -------- MCDropout model ganet_deep_airsim_01 with sample size of 3
GT_BASE_PATH="/robodata/srabiee/AirSim_IVSLAM/cityenv_wb"
PRED_BASE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model0_r0.1_mcdropout3_epoch_14/cityenv_wb"
TRAINING_LIST='../../lists/airsim_wb_training_01.list'
VALIDATION_LIST='../../lists/airsim_wb_validation_01.list'
OUTPUT_SAVE_PATH="/robodata/user_data/srabiee/results/ipr/nn_models/ganet_deep_airsim_01/model0_r0.1_mcdropout3_epoch_14_calib_unc_50" # A new model trained from random weigths (after fixing the data label issues)

LOGS_COMMENT="MCDropout3_rate_0.1_model_0_Unc_calibration_50_fixed_data"
RESUME_FROM="/robodata/user_data/srabiee/results/ipr/nn_models/ganet_deep_airsim_01/model0_r0.1_mcdropout3_epoch_14_calib_unc_2/calib_unc_epoch_399.pth"

BATCH_SIZE=30 # 60, 30
IMAGE_HEIGHT="300"
IMAGE_WIDTH="480"


mkdir -p $OUTPUT_SAVE_PATH

## Resume training
# CUDA_VISIBLE_DEVICES="5" \
# python ../training/calibrate_uncertainty_estimates.py \
# --pred_data_path=$PRED_BASE_PATH \
# --gt_data_path=$GT_BASE_PATH \
# --save_path=$OUTPUT_SAVE_PATH \
# --image_height=$IMAGE_HEIGHT \
# --image_width=$IMAGE_WIDTH \
# --num_workers=0 \
# --num_epochs=400 \
# --batch_size=$BATCH_SIZE \
# --use_gpu=True \
# --training_list=$TRAINING_LIST \
# --val_list=$VALIDATION_LIST \
# --subsample_factor="0.2" \
# --logs_comment=$LOGS_COMMENT \
# --resume=$RESUME_FROM

# exit 0

# Train from scratch
CUDA_VISIBLE_DEVICES="0,1" \
python ../training/calibrate_uncertainty_estimates.py \
--pred_data_path=$PRED_BASE_PATH \
--gt_data_path=$GT_BASE_PATH \
--save_path=$OUTPUT_SAVE_PATH \
--image_height=$IMAGE_HEIGHT \
--image_width=$IMAGE_WIDTH \
--num_workers=0 \
--num_epochs=400 \
--batch_size=$BATCH_SIZE \
--use_gpu=True \
--training_list=$TRAINING_LIST \
--val_list=$VALIDATION_LIST \
--subsample_factor="0.2" \
--logs_comment=$LOGS_COMMENT 


exit 0