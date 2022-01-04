#!/bin/bash

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


# Test ensemble model trained on sample data
ENSEMBLE_MODELS=("/robodata/user_data/srabiee/results/ipr/nn_models/ganet_deep_airsim_sample4_00/model_instance_0/_epoch_32.pth" 

"/robodata/user_data/srabiee/results/ipr/nn_models/ganet_deep_airsim_sample4_00/model_instance_3/_epoch_32.pth" 

"/robodata/user_data/srabiee/results/ipr/nn_models/ganet_deep_airsim_sample4_00/model_instance_4/_epoch_32.pth"
)

ENSEMBLE_MODELS_STR=""
for model in "${ENSEMBLE_MODELS[@]}"
do
  ENSEMBLE_MODELS_STR="$ENSEMBLE_MODELS_STR  $model"
done

CUDA_VISIBLE_DEVICES=6,7 python predict.py \
                  --batchSize=2 \
                  --crop_height=600 \
                  --crop_width=960 \
                  --max_disp=192 \
                  --scale_factor=0.5 \
                  --subsample_factor=0.2 \
                  --threads=16 \
                  --airsim_fmt=1 \
                  --model_paths $ENSEMBLE_MODELS_STR \
                  --model='GANet_deep' \
                  --data_path='' \
                  --test_list='lists/airsim_wb_sample_session.list' \
                  --save_path='/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_sample4_00/model_ensemble_epoch_32_e034/cityenv_wb/' 

exit