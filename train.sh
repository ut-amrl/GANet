


# # **** Selected *****
# # Train on the 00 training dataset (initialize with pretrained model on KITTI 2015)
# # Generate a random seed number for the training
# MODEL_NUMBER="0"
# PRETRAINED_MODEL="/robodata/user_data/srabiee/results/ipr/nn_models/ganet_deep_airsim_00/model_instance_0/_epoch_2.pth"
# START_EPOCH=3

# MODEL_NAME="model_instance_${MODEL_NUMBER}"
# OUTPUT_PATH="/robodata/user_data/srabiee/results/ipr/nn_models/ganet_deep_airsim_00/${MODEL_NAME}/"
# LOG_FILE="$OUTPUT_PATH/training_log.txt"
# mkdir -p $OUTPUT_PATH
# RANDOM_SEED=$(date +%s)
# echo "Random seed: $RANDOM_SEED" | tee $LOG_FILE
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py  \
#                 --batchSize=4 \
#                 --testBatchSize=4 \
#                 --startEpoch=$START_EPOCH \
#                 --crop_height=600 \
#                 --crop_width=960 \
#                 --max_disp=192 \
#                 --thread=16 \
#                 --training_list='lists/airsim_wb_training_00.list' \
#                 --val_list='lists/airsim_wb_validation_00.list' \
#                 --save_path=$OUTPUT_PATH \
#                 --resume=$PRETRAINED_MODEL \
#                 --model='GANet_deep' \
#                 --airsim=1 \
#                 --seed=$RANDOM_SEED \
#                 --scale_factor=0.5 \
#                 --subsample_factor_train=0.2 \
#                 --subsample_factor_val=0.05 \
#                 --nEpochs=100 2>&1 | tee -a  $LOG_FILE
# exit



# # **** Selected *****
# # Sample training on a small subset of the data  
# MODEL_NUMBER="2"
# PRETRAINED_MODEL="pretrained_models/kitti2015_final.pth"
# START_EPOCH=1

# MODEL_NAME="model_instance_${MODEL_NUMBER}"
# OUTPUT_PATH="/robodata/user_data/srabiee/results/ipr/nn_models/ganet_deep_airsim_sample4_00/${MODEL_NAME}/"
# LOG_FILE="$OUTPUT_PATH/training_log.txt"
# mkdir -p $OUTPUT_PATH
# RANDOM_SEED=$(date +%s)
# CUDA_VISIBLE_DEVICES=2,3 python -u train.py \
#                 --batchSize=2 \
#                 --testBatchSize=2 \
#                 --startEpoch=$START_EPOCH \
#                 --crop_height=600 \
#                 --crop_width=960 \
#                 --max_disp=192 \
#                 --thread=16 \
#                 --training_list='lists/airsim_wb_sample_training_00.list' \
#                 --val_list='lists/airsim_wb_sample_validation_00.list' \
#                 --save_path=$OUTPUT_PATH \
#                 --resume=$PRETRAINED_MODEL \
#                 --model='GANet_deep' \
#                 --airsim=1 \
#                 --scale_factor=0.5 \
#                 --subsample_factor_train=1.0 \
#                 --subsample_factor_val=0.2 \
#                 --nEpochs=100 2>&1 | tee -a  $LOG_FILE
# exit

# **** Selected *****
# Sample training on a small subset of the data without using pretrained weights) 
MODEL_NUMBER="0"
PRETRAINED_MODEL=""
START_EPOCH=1

MODEL_NAME="model_instance_${MODEL_NUMBER}"
OUTPUT_PATH="/robodata/user_data/srabiee/results/ipr/nn_models/ganet_deep_airsim_sample4_NoPr_00/${MODEL_NAME}/"
LOG_FILE="$OUTPUT_PATH/training_log.txt"
mkdir -p $OUTPUT_PATH
RANDOM_SEED=$(date +%s)
CUDA_VISIBLE_DEVICES=4,5 python -u train.py \
                --batchSize=2 \
                --testBatchSize=2 \
                --startEpoch=$START_EPOCH \
                --crop_height=600 \
                --crop_width=960 \
                --max_disp=192 \
                --thread=16 \
                --training_list='lists/airsim_wb_sample_training_00.list' \
                --val_list='lists/airsim_wb_sample_validation_00.list' \
                --save_path=$OUTPUT_PATH \
                --resume=$PRETRAINED_MODEL \
                --model='GANet_deep' \
                --airsim=1 \
                --scale_factor=0.5 \
                --subsample_factor_train=1.0 \
                --subsample_factor_val=0.2 \
                --nEpochs=100 2>&1 | tee -a  $LOG_FILE
exit


#Fine tuning for kitti 2015
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train.py --batchSize=16 \
#                 --crop_height=240 \
#                 --crop_width=528 \
#                 --max_disp=192 \
#                 --thread=16 \
#                 --data_path='/robodata/public_datasets/kitti/stereo_2015/training/' \
#                 --training_list='lists/kitti2015_train.list' \
#                 --save_path='./checkpoint/finetune_kitti2015' \
#                 --kitti2015=1 \
#                 --shift=3 \
#                 --resume='./checkpoint/sceneflow_epoch_10.pth' \
#                 --nEpochs=800 
# exit


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --batchSize=16 \
#                 --crop_height=240 \
#                 --crop_width=528 \
#                 --max_disp=192 \
#                 --thread=16 \
#                 --data_path='/ssd1/zhangfeihu/data/stereo/' \
#                 --training_list='lists/sceneflow_train.list' \
#                 --save_path='./checkpoint/sceneflow' \
#                 --resume='' \
#                 --model='GANet_deep' \
#                 --nEpochs=11 2>&1 |tee logs/log_train_sceneflow.txt


# #Fine tuning for kitti 2015
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --batchSize=16 \
#                 --crop_height=240 \
#                 --crop_width=528 \
#                 --max_disp=192 \
#                 --thread=16 \
#                 --data_path='/media/feihu/Storage/stereo/data_scene_flow/training/' \
#                 --training_list='lists/kitti2015_train.list' \
#                 --save_path='./checkpoint/finetune_kitti2015' \
#                 --kitti2015=1 \
#                 --shift=3 \
#                 --resume='./checkpoint/sceneflow_epoch_10.pth' \
#                 --nEpochs=800 2>&1 |tee logs/log_finetune_kitti2015.txt

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --batchSize=8 \
#                 --crop_height=240 \
#                 --crop_width=1248 \
#                 --max_disp=192 \
#                 --thread=16 \
#                 --data_path='/media/feihu/Storage/stereo/data_scene_flow/training/' \
#                 --training_list='lists/kitti2015_train.list' \
#                 --save_path='./checkpoint/finetune2_kitti2015' \
#                 --kitti2015=1 \
#                 --shift=3 \
#                 --lr=0.0001 \
#                 --resume='./checkpoint/finetune_kitti2015_epoch_800.pth' \
#                 --nEpochs=8 2>&1 |tee logs/log_finetune_kitti2015.txt

# #Fine tuning for kitti 2012

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --batchSize=16 \
#                 --crop_height=240 \
#                 --crop_width=528 \
#                 --max_disp=192 \
#                 --thread=16 \
#                 --data_path='/media/feihu/Storage/stereo/kitti/training/' \
#                 --training_list='lists/kitti2012_train.list' \
#                 --save_path='./checkpoint/finetune_kitti' \
#                 --kitti=1 \
#                 --shift=3 \
#                 --resume='./checkpoint/sceneflow_epoch_10.pth' \
#                 --nEpochs=800 2>&1 |tee logs/log_finetune2_kitti.txt

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --batchSize=8 \
#                 --crop_height=240 \
#                 --crop_width=1248 \
#                 --max_disp=192 \
#                 --thread=16 \
#                 --data_path='/media/feihu/Storage/stereo/kitti/training/' \
#                 --training_list='lists/kitti2012_train.list' \
#                 --save_path='./checkpoint/finetune2_kitti' \
#                 --kitti=1 \
#                 --shift=3 \
#                 --lr=0.0001 \
#                 --resume='./checkpoint/finetune_kitti_epoch_800.pth' \
#                 --nEpochs=8 2>&1 |tee logs/log_finetune2_kitti.txt




