#!/bin/bash

# This script joins a number of videos into a single video given a grid size and a list of base directories for the videos. It is assumed that each base directory contains a number of sessions with the name format of %05d. A video is generated for each session.

# NOTE: It is assumed that all input videos are of the same size and have the same frame rate.


# --------------
# Adjust these parameters
# --------------

# SESSIONS="ALL"
SESSIONS="1007"

OUTPUT_DIR="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_sample4_00/model_ensemble_epoch_32_e034/cityenv_wb/"

BASE_DIRS=(
"/robodata/srabiee/AirSim_IVSLAM/cityenv_wb/"
"/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_sample4_00/model_ensemble_epoch_32_e034/cityenv_wb/" 
"/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_sample4_00/model_ensemble_epoch_32_e034/cityenv_wb/" 
"/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_sample4_00/model_ensemble_epoch_32_e034/cityenv_wb/"
)  

VIDEO_DIRS=(
"img_left"
"disp_pred"
"err_vis"
"uncertainty_vis"
)

# Number of cells in the grid
GRID_X=2
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

    curr_x=$((i % GRID_X))
    curr_y=$((i / GRID_Y))

    if (( i==0 )); then
      echo "First video"
      filter_cmd+="[0:v]pad=iw*$GRID_X:$GRID_Y*ih[int];"
    elif (( i==$((${#BASE_DIRS[@]} - 1)) )); then
      echo "Last video"
      filter_cmd+="[int][$i:v]overlay=$curr_x*W/2:$curr_y*main_h/2[vid]"
    else 
      echo "# $i video"
      filter_cmd+="[int][$i:v]overlay=$curr_x*W/2:$curr_y*main_h/2[int];"
    fi
  done

  # echo "filter_cmd: $filter_cmd"
  # echo "video_stream_cmd: $video_stream_cmd"

  OUTPUT_VIDEO_PATH=${OUTPUT_DIR}/${SESSION_NUM_STR}/${SESSION_NUM_STR}"_joined_vid."${VIDEO_FMT}


  # Check desired video format
  if [ $VIDEO_FMT == "mp4" ]; then
    # -crf sets the video quality (15-25), the lower the better
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
    ffmpeg \
      -y \
      $video_stream_cmd \
      -filter_complex $filter_cmd \
      -map [vid] \
      -c:v mpeg4 \
      -qscale:v 3 \
      -loglevel 4 \
      $OUTPUT_VIDEO_PATH
  else
    echo "Unknown video format: $VIDEO_FMT"
    exit
  fi


  
done





