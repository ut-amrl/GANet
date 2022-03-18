#!/bin/bash

# This script is used to run evaluate_uncertainty_predictions_pixelwise.py



# --------------------------------------------------------------------
# ------------ Run WITHOUT predicted uncertainty calibration----------
# --------------------------------------------------------------------


# # ---------- 
# # Test ensemble model trained on sample data
# # PREDICTIONS_BASE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_sample4_00/model_ensemble_epoch_32_e034/cityenv_wb_test_unc/"
# # PREDICTED_DEPTH_UNC_FOLDER_NAME="depth_uncertainty_pred"
# # PATCH_DATASET_NAME="test_1_ganet_v0"
# # PATCH_DATASET_PATH="/robodata/user_data/srabiee/IVOA/GANET/ganet_deep_airsim_sample4_00_epoch_32_e034_v0_p70_rg30_errTh1.0_0.2_NoGP/"
# # DEPTH_ERR_THRESH_ABS="1.0" # 1.0 , 1.5
# # DEPTH_ERR_THRESH_REL="0.2" # 0.1 , 0.3

# # ----------
# # Test ensemble model trained on full training_01
# PREDICTIONS_BASE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01/model_ensemble_epoch_50_e012/cityenv_wb/"
# PREDICTED_DEPTH_UNC_FOLDER_NAME="depth_uncertainty_pred"
# PATCH_DATASET_NAME="test_01_ganet_v0"
# PATCH_DATASET_PATH="/robodata/user_data/srabiee/IVOA/GANET/ganet_deep_airsim_01_ensemble_epoch_50_e012_v0_p36_rg30_errTh1.0_0.1_NoGP_ds/"
# DEPTH_ERR_THRESH_ABS="1.0" # 1.0 , 1.5
# DEPTH_ERR_THRESH_REL="0.1" # 0.1 , 0.3

# # ----------
# # Test MCDropout model trained on full training_01 (inference with sample size 3)
# GT_BASE_PATH="/robodata/srabiee/AirSim_IVSLAM/cityenv_wb"
# PREDICTIONS_BASE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model0_r0.1_mcdropout3_epoch_14/cityenv_wb/"
# PREDICTED_DEPTH_UNC_FOLDER_NAME="depth_uncertainty_pred"
# PATCH_DATASET_NAME="test_01_ganet_v0"
# PATCH_DATASET_PATH="/robodata/user_data/srabiee/IVOA/GANET/ganet_deep_airsim_01_mcdropoutOff_r0.0_epoch_14_p36_rg30_errTh1.0_0.2_NoGP_ds/"
# DEPTH_ERR_THRESH_ABS="1.0" # 1.0 , 1.5
# DEPTH_ERR_THRESH_REL="0.2" # 0.1, 0.2 , 0.3

# # ----------
# # Test MCDropout model trained on full training_01 (inference with sample size 10)
# GT_BASE_PATH="/robodata/srabiee/AirSim_IVSLAM/cityenv_wb"
# PREDICTIONS_BASE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model0_r0.1_mcdropout10_epoch_14/cityenv_wb/"
# PREDICTED_DEPTH_UNC_FOLDER_NAME="depth_uncertainty_pred"
# PATCH_DATASET_NAME="test_01_ganet_v0"
# PATCH_DATASET_PATH="/robodata/user_data/srabiee/IVOA/GANET/ganet_deep_airsim_01_mcdropoutOff_r0.0_epoch_14_p36_rg30_errTh1.0_0.2_NoGP_ds/"
# DEPTH_ERR_THRESH_ABS="1.0" # 1.0 , 1.5
# DEPTH_ERR_THRESH_REL="0.2" # 0.1, 0.2 , 0.3


# # ----------
# # Test the ensemble model that was trained with MCDropout (GANetDropout) trained on full training_01
# GT_BASE_PATH="/robodata/srabiee/AirSim_IVSLAM/cityenv_wb"
# PREDICTIONS_BASE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model_ensemble_epoch_14_e012/cityenv_wb/" 
# PREDICTED_DEPTH_UNC_FOLDER_NAME="depth_uncertainty_pred"
# PATCH_DATASET_NAME="test_01_ganet_v0"
# PATCH_DATASET_PATH="/robodata/user_data/srabiee/IVOA/GANET/ganet_deep_airsim_01_mcdropoutOff_r0.0_epoch_14_p36_rg30_errTh1.0_0.2_NoGP_ds/"
# DEPTH_ERR_THRESH_ABS="1.0" # 1.0 , 1.5
# DEPTH_ERR_THRESH_REL="0.2" # 0.1, 0.2 , 0.3


# SAVE_FAILURE_PREDICTION_VIS_IMAGES="true"
# OUT_PRED_FAILURE_VIS_FOLDER_NAME="failure_pred_vis"
# OUT_PRED_FAILURE_PATCH_VIS_FOLDER_NAME="failure_pred_patch_vis"
# PRED_DEPTH_FOLDER_NAME="img_depth_pred"

# CUDA_VISIBLE_DEVICES="0" \
# python ../evaluation/evaluate_uncertainty_predictions_pixelwise.py \
# --gt_data_path=$GT_BASE_PATH \
# --predictions_base_path=$PREDICTIONS_BASE_PATH \
# --pred_depth_unc_folder=$PREDICTED_DEPTH_UNC_FOLDER_NAME \
# --pred_depth_folder=$PRED_DEPTH_FOLDER_NAME \
# --output_pred_failure_vis_folder=$OUT_PRED_FAILURE_VIS_FOLDER_NAME \
# --output_pred_failure_patch_vis_folder=$OUT_PRED_FAILURE_PATCH_VIS_FOLDER_NAME \
# --patch_dataset_name=$PATCH_DATASET_NAME \
# --patch_dataset_path=$PATCH_DATASET_PATH \
# --depth_err_thresh_abs=$DEPTH_ERR_THRESH_ABS \
# --depth_err_thresh_rel=$DEPTH_ERR_THRESH_REL \
# --patch_size=70 \
# --num_workers=8 \
# --save_failure_pred_vis_images=$SAVE_FAILURE_PREDICTION_VIS_IMAGES



# --------------------------------------------------------------------
# ------------ Run WITH predicted uncertainty calibration-------------
# --------------------------------------------------------------------


# # ----------
# # Test MCDropout model trained on full training_01 (inference with sample size 3)
# GT_BASE_PATH="/robodata/srabiee/AirSim_IVSLAM/cityenv_wb"
# PREDICTIONS_BASE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model0_r0.1_mcdropout3_epoch_14/cityenv_wb/"
# # PREDICTIONS_BASE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model0_r0.1_mcdropout3_epoch_14/cityenv_wb_manual_calib/"
# PREDICTED_DEPTH_UNC_FOLDER_NAME="depth_uncertainty_pred"
# PATCH_DATASET_NAME="test_01_ganet_v0"
# # PATCH_DATASET_NAME="test_tmp"
# # PATCH_DATASET_PATH="/robodata/user_data/srabiee/IVOA/GANET/ganet_deep_airsim_01_mcdropoutOff_r0.0_epoch_14_p36_rg30_errTh1.0_0.2_NoGP_ds/"
# PATCH_DATASET_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model0_r0.0_mcdropoutOff_epoch_14/cityenv_wb/"
# DEPTH_ERR_THRESH_ABS="1.0" # 1.0 , 1.5
# DEPTH_ERR_THRESH_REL="0.2" # 0.1, 0.2 , 0.3
# UNC_CALIBRATION_MODEL="/robodata/user_data/srabiee/results/ipr/nn_models/ganet_deep_airsim_01/model0_r0.1_mcdropout3_epoch_14_calib_unc_50/calib_unc_epoch_100.pth"

# SAVE_FAILURE_PREDICTION_VIS_IMAGES="false"
# OUT_PRED_FAILURE_VIS_FOLDER_NAME="failure_pred_vis_calibrated"
# OUT_PRED_FAILURE_PATCH_VIS_FOLDER_NAME="failure_pred_patch_vis_calibrated"
# PRED_DEPTH_FOLDER_NAME="img_depth_pred"


# # ----------
# # Test MCDropout model trained on full training_01 (inference with sample size 10)
# PREDICTIONS_BASE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model0_r0.1_mcdropout10_epoch_14/cityenv_wb/"
# PREDICTED_DEPTH_UNC_FOLDER_NAME="depth_uncertainty_pred"
# PATCH_DATASET_NAME="test_01_ganet_v0"
# PATCH_DATASET_PATH="/robodata/user_data/srabiee/IVOA/GANET/ganet_deep_airsim_01_mcdropoutOff_r0.0_epoch_14_p36_rg30_errTh1.0_0.2_NoGP_ds/"
# DEPTH_ERR_THRESH_ABS="1.0" # 1.0 , 1.5
# DEPTH_ERR_THRESH_REL="0.2" # 0.1, 0.2 , 0.3
# UNC_CALIBRATION_MODEL="/robodata/user_data/srabiee/results/ipr/nn_models/ganet_deep_airsim_01/model0_r0.1_mcdropout10_epoch_14_calib_unc_00/calib_unc_epoch_82.pth"

# SAVE_FAILURE_PREDICTION_VIS_IMAGES="true"
# OUT_PRED_FAILURE_VIS_FOLDER_NAME="failure_pred_vis_calibrated"
# OUT_PRED_FAILURE_PATCH_VIS_FOLDER_NAME="failure_pred_patch_vis_calibrated"
# PRED_DEPTH_FOLDER_NAME="img_depth_pred"


# # ----------
# Test the ensemble model that was trained with MCDropout (GANetDropout) trained on full training_01
GT_BASE_PATH="/robodata/srabiee/AirSim_IVSLAM/cityenv_wb"
# PREDICTIONS_BASE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model_ensemble_epoch_14_e012/cityenv_wb/"
PREDICTIONS_BASE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model_ensemble_epoch_14_e012/cityenv_wb_manual_calib/" 
PREDICTED_DEPTH_UNC_FOLDER_NAME="depth_uncertainty_pred"
PATCH_DATASET_NAME="test_01_ganet_v0"
# PATCH_DATASET_NAME="test_tmp"
PATCH_DATASET_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model0_r0.0_mcdropoutOff_epoch_14/cityenv_wb/"
DEPTH_ERR_THRESH_ABS="1.0" # 1.0 , 1.5
DEPTH_ERR_THRESH_REL="0.2" # 0.1, 0.2 , 0.3
UNC_CALIBRATION_MODEL="/robodata/user_data/srabiee/results/ipr/nn_models/ganet_deep_airsim_01_dropout/model_ensemble_epoch_14_e012_calib_unc_50/calib_unc_epoch_100.pth"

SAVE_FAILURE_PREDICTION_VIS_IMAGES="true"
OUT_PRED_FAILURE_VIS_FOLDER_NAME="failure_pred_vis_calibrated"
OUT_PRED_FAILURE_PATCH_VIS_FOLDER_NAME="failure_pred_patch_vis_calibrated"
PRED_DEPTH_FOLDER_NAME="img_depth_pred"





CUDA_VISIBLE_DEVICES="0" \
python ../evaluation/evaluate_uncertainty_predictions_pixelwise.py \
--gt_data_path=$GT_BASE_PATH \
--predictions_base_path=$PREDICTIONS_BASE_PATH \
--pred_depth_unc_folder=$PREDICTED_DEPTH_UNC_FOLDER_NAME \
--pred_depth_folder=$PRED_DEPTH_FOLDER_NAME \
--output_pred_failure_vis_folder=$OUT_PRED_FAILURE_VIS_FOLDER_NAME \
--output_pred_failure_patch_vis_folder=$OUT_PRED_FAILURE_PATCH_VIS_FOLDER_NAME \
--patch_dataset_name=$PATCH_DATASET_NAME \
--patch_dataset_path=$PATCH_DATASET_PATH \
--depth_err_thresh_abs=$DEPTH_ERR_THRESH_ABS \
--depth_err_thresh_rel=$DEPTH_ERR_THRESH_REL \
--patch_size=70 \
--num_workers=8 \
--save_failure_pred_vis_images=$SAVE_FAILURE_PREDICTION_VIS_IMAGES \
--unc_calibration_model=$UNC_CALIBRATION_MODEL