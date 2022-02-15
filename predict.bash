#!/bin/bash

FX="480.0"
BASELINE="0.6"

# # Test the model trained on sample data 
# CUDA_VISIBLE_DEVICES=6,7 python predict.py \
#                   --batchSize=2 \
#                   --crop_height=600 \
#                   --crop_width=960 \
#                   --max_disp=192 \
#                   --scale_factor=0.5 \
#                   --subsample_factor=0.2 \
#                   --threads=16 \
#                   --airsim_fmt=1 \
#                   --resume='/robodata/user_data/srabiee/results/ipr/nn_models/ganet_deep_airsim_sample4_00/model_instance_0/_epoch_28.pth' \
#                   --model='GANet_deep' \
#                   --data_path='' \
#                   --test_list='lists/airsim_wb_sample_session.list' \
#                   --save_path='/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_sample4_00/model0_epoch28/cityenv_wb/' 

# exit

# --------------------------------------------


# # Test ensemble model trained on sample data
# ENSEMBLE_MODELS=("/robodata/user_data/srabiee/results/ipr/nn_models/ganet_deep_airsim_sample4_00/model_instance_0/_epoch_32.pth" 

# "/robodata/user_data/srabiee/results/ipr/nn_models/ganet_deep_airsim_sample4_00/model_instance_3/_epoch_32.pth" 

# "/robodata/user_data/srabiee/results/ipr/nn_models/ganet_deep_airsim_sample4_00/model_instance_4/_epoch_32.pth"
# )
# # SAVE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_sample4_00/model_ensemble_epoch_32_e034/cityenv_wb/"

# # Generate depth uncertainty images
# SAVE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_sample4_00/model_ensemble_epoch_32_e034/cityenv_wb_test_unc/"

# TEST_LIST='lists/airsim_wb_sample_session.list'
# # TEST_LIST='lists/airsim_wb_test_00.list'

# ----------

# # Test ensemble model trained on sample data
# ENSEMBLE_MODELS=("/robodata/user_data/srabiee/results/ipr/nn_models/ganet_deep_airsim_00/model_instance_0/_epoch_19.pth" 
# )
# SAVE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_00/model_instance_0_epoch_19/cityenv_wb/"

# TEST_LIST='lists/airsim_wb_sample_session.list'
# # TEST_LIST='lists/airsim_wb_test_00.list'

# ----------

# Test ensemble model trained on full training_01
ENSEMBLE_MODELS=(
"/robodata/user_data/srabiee/results/ipr/nn_models/ganet_deep_airsim_01/model_instance_0/_epoch_50.pth" 

"/robodata/user_data/srabiee/results/ipr/nn_models/ganet_deep_airsim_01/model_instance_1/_epoch_50.pth" 

"/robodata/user_data/srabiee/results/ipr/nn_models/ganet_deep_airsim_01/model_instance_2/_epoch_50.pth"
)

SAVE_PATH="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01/model_ensemble_epoch_50_e012/cityenv_wb/"


TEST_LIST='lists/airsim_wb_test_01.list'
# TEST_LIST='lists/airsim_wb_validation_01.list'
# TEST_LIST='lists/airsim_wb_training_01.list'

# ----------

# ---- Multi GPU
ENSEMBLE_MODELS_STR=""
for model in "${ENSEMBLE_MODELS[@]}"
do
  ENSEMBLE_MODELS_STR="$ENSEMBLE_MODELS_STR  $model"
done

CUDA_VISIBLE_DEVICES=0,1,2 python predict.py \
                  --batchSize=21 \
                  --crop_height=600 \
                  --crop_width=960 \
                  --max_disp=192 \
                  --fx=$FX \
                  --baseline=$BASELINE \
                  --scale_factor=0.5 \
                  --subsample_factor=0.2 \
                  --threads=16 \
                  --airsim_fmt=1 \
                  --model_paths $ENSEMBLE_MODELS_STR \
                  --model='GANet_deep' \
                  --data_path='' \
                  --test_list=$TEST_LIST \
                  --save_path=$SAVE_PATH 

exit


# --- Single GPU
# ENSEMBLE_MODELS_STR=""
# for model in "${ENSEMBLE_MODELS[@]}"
# do
#   ENSEMBLE_MODELS_STR="$ENSEMBLE_MODELS_STR  $model"
# done

# CUDA_VISIBLE_DEVICES=0 python predict.py \
#                   --batchSize=1 \
#                   --crop_height=600 \
#                   --crop_width=960 \
#                   --max_disp=192 \
#                   --fx=$FX \
#                   --baseline=$BASELINE \
#                   --scale_factor=0.5 \
#                   --subsample_factor=0.2 \
#                   --threads=16 \
#                   --airsim_fmt=1 \
#                   --model_paths $ENSEMBLE_MODELS_STR \
#                   --model='GANet_deep' \
#                   --data_path='' \
#                   --test_list=$TEST_LIST \
#                   --save_path=$SAVE_PATH 

# exit