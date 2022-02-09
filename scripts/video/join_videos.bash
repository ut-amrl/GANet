#!/bin/bash

# This script joins a number of videos into a single video given a grid size and a list of base directories for the videos. It is assumed that each base directory contains a number of sessions with the name format of %05d. A video is generated for each session. The list of provided videos are arranged in the grid in a row-major order.

# NOTE: It is assumed that all input videos are of the same size and have the same frame rate.


# --------------
# Adjust these parameters
# --------------

# SESSIONS="ALL"
# SESSIONS="1007"
# airsim_wb_test_01.list 
SESSIONS=" 01007 01012 01017 01022 01027 01032 02007 02012 02017 02022 02027 02032 "

OUTPUT_DIR="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01/model_ensemble_epoch_50_e012/cityenv_wb/"

# BASE_DIRS=(
# "/robodata/srabiee/AirSim_IVSLAM/cityenv_wb/"
# "/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_sample4_00/model_ensemble_epoch_32_e034/cityenv_wb/" 
# "/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_sample4_00/model_ensemble_epoch_32_e034/cityenv_wb/" 
# "/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_sample4_00/model_ensemble_epoch_32_e034/cityenv_wb/"
# )  
# VIDEO_DIRS=(
# "img_left"
# "disp_pred"
# "err_vis"
# "uncertainty_vis"
# )
# VIDEO_SUFFIX=""

BASE_DIRS=(
"/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01/model_ensemble_epoch_50_e012/cityenv_wb/"
"/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01/model_ensemble_epoch_50_e012/cityenv_wb/"
"/robodata/srabiee/scratch/results/IVOA/evaluation/GANET/ganet_deep_airsim_01_ensemble_epoch_50_e012_v0_p36_rg30_errTh1.0_0.1_NoGP_ds/evaluation_multi_class_test_per_image/"
"/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01/model_ensemble_epoch_50_e012/cityenv_wb/"
"/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01/model_ensemble_epoch_50_e012/cityenv_wb/"
"/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01/model_ensemble_epoch_50_e012/cityenv_wb/"
)  
VIDEO_DIRS=(
"disp_pred"
"depth_pred_err_binary_vis_abs1.0_rel_0.1_maxRange30"
"failure_pred_patch_vis"
"uncertainty_vis"
"failure_pred_vis"
"failure_pred_patch_vis"
)
VIDEO_SUFFIX="_patch_err_pred_comp"


# BASE_DIRS=(
# "/robodata/srabiee/AirSim_IVSLAM/cityenv_wb/"
# "/robodata/srabiee/AirSim_IVSLAM/cityenv_wb/"
# "/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_sample4_00/model_ensemble_epoch_32_e034/cityenv_wb/" 
# "/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_sample4_00/model_ensemble_epoch_32_e034/cityenv_wb/" 
# )  
# VIDEO_DIRS=(
# "img_left"
# "img_disp"
# "err_vis"
# "disp_pred"
# )
# VIDEO_SUFFIX="_pred_gt_comp"

# Number of cells in the grid
GRID_X=3
GRID_Y=2
 
VIDEO_FMT="avi" # mp4, avi


# -----------------
MAIN_BASE_DIR=${BASE_DIRS[0]}

if [ $SESSIONS == "ALL" ]; then
  pushd $MAIN_BASE_DIR
  folders=$(ls -d *)
  popd
  SESSIONS_LIST=$folders
else
  SESSIONS_LIST="$SESSIONS"
fi

echo "Sessions to be processed:"
echo $SESSIONS_LIST



for session in $SESSIONS_LIST; do
  printf -v SESSION_NUM_STR '%05d' "$((10#$session))"

  echo "*********************************"
  echo "Joining Videos from : $SESSION_NUM_STR"
  echo "*********************************"
  
  # Assert that the number of videos are smaller than the grid size
  if (( $GRID_X * $GRID_Y < ${#VIDEO_DIRS[@]} )); then
    echo "ERROR: Number of videos (${#VIDEO_DIRS[@]}) is larger than the grid size ($(($GRID_X * $GRID_Y)))"
    exit 1
  fi

  # Error if there is only one video stream
  if (( ${#VIDEO_DIRS[@]} == 1 )); then
    echo "ERROR: There is only one video stream"
    exit 1
  fi

  # Go through BASE_DIRS and VIDEO_DIRS
  filter_cmd=""
  video_stream_cmd=""
  for ((i=0; i<${#BASE_DIRS[@]}; i++)); do
    BASE_DIR=${BASE_DIRS[$i]}
    VIDEO_DIR=${VIDEO_DIRS[$i]}
    VIDEO_NAME=${SESSION_NUM_STR}_${VIDEO_DIR}"."${VIDEO_FMT}

    video_stream_cmd+=" -i $BASE_DIR/$SESSION_NUM_STR/$VIDEO_DIR/$VIDEO_NAME"

    # Check if file exists
    if [ ! -f $BASE_DIR/$SESSION_NUM_STR/$VIDEO_DIR/$VIDEO_NAME ]; then
      echo "ERROR: File $BASE_DIR/$SESSION_NUM_STR/$VIDEO_DIR/$VIDEO_NAME does not exist"
      exit 1
    fi

    curr_x=$((i % GRID_X))
    curr_y=$((i / GRID_X))

    if (( i==0 )); then
      echo "First video"
      filter_cmd+="[0:v]pad=iw*$GRID_X:$GRID_Y*ih[int];"
    elif (( i==$((${#BASE_DIRS[@]} - 1)) )); then
      echo "Last video"
      filter_cmd+="[int][$i:v]overlay=shortest=1:x=$curr_x*W/$GRID_X:y=$curr_y*main_h/$GRID_Y[vid]"
    else 
      echo "# $i video"
      filter_cmd+="[int][$i:v]overlay=shortest=1:x=$curr_x*W/$GRID_X:y=$curr_y*main_h/$GRID_Y[int];"
    fi
  done

  # echo "filter_cmd: $filter_cmd"
  # echo "video_stream_cmd: $video_stream_cmd"

  OUTPUT_VIDEO_PATH=${OUTPUT_DIR}/${SESSION_NUM_STR}/${SESSION_NUM_STR}${VIDEO_SUFFIX}"_joined_vid."${VIDEO_FMT}

  echo "Output video: $OUTPUT_VIDEO_PATH"

  # Check desired video format
  if [ $VIDEO_FMT == "mp4" ]; then
    # -crf sets the video quality (15-25), the lower the better
    echo "Generating MP4 video ..."
    ffmpeg \
      -y \
      $video_stream_cmd \
      -filter_complex $filter_cmd \
      -map [vid] \
      -vcodec libx264 \
      -crf 20 \
      -loglevel 4 \
      $OUTPUT_VIDEO_PATH
  elif [ $VIDEO_FMT == "avi" ]; then
    # -qscale:v sets the video quality (1-31), the lower the better
    echo "Generating AVI video ..."
    ffmpeg \
      -y \
      $video_stream_cmd \
      -filter_complex $filter_cmd \
      -map [vid] \
      -c:v mpeg4 \
      -qscale:v 3 \
      -loglevel 0 \
      $OUTPUT_VIDEO_PATH
  else
    echo "Unknown video format: $VIDEO_FMT"
    exit
  fi


  
done





