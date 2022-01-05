#!/bin/bash

# This script goes through a number of given directories and generates videos from static image files. It is assumed that the base directory includes a number of session folders the name of which follows the format of %05d. A video is generated for each session.


# --------------
# Adjust these parameters
# --------------

SESSIONS="ALL"
# SESSIONS="1007"

# SOURCE_BASE_DIR=\
# "/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_sample4_00/model_ensemble_epoch_32_e034/cityenv_wb/"
# DIRECTORIES_OF_INTEREST=(
#    "disp_pred"
#    "err_vis"
#    "uncertainty_vis" )
# USE_EVERY_N_IMG="1" # 1, 5

SOURCE_BASE_DIR=\
"/robodata/srabiee/AirSim_IVSLAM/cityenv_wb/"
DIRECTORIES_OF_INTEREST=(
   "img_left" )
USE_EVERY_N_IMG="5" # 1, 5

# This frame rate is applied after skipping the unwanted frames. So, if you have recorded at 30 fps, and you use every 10th frame, then to keep the output video real-time you should set the OUTPUT_FRAME_RATE to 30/10=3.
OUTPUT_FRAME_RATE="6"
OUTPUT_SIZE="480x300"
VIDEO_FMT="avi" # mp4, avi

# -----------------

if [ $SESSIONS == "ALL" ]; then
  pushd $SOURCE_BASE_DIR
  folders=$(ls -d *)
  popd
  SESSIONS_LIST=$folders
else
  SESSIONS_LIST="$SESSIONS"
fi

echo "Sessions to be processed:"
echo $SESSIONS_LIST

OUTPUT_FRAME_RATE=$(($OUTPUT_FRAME_RATE * $USE_EVERY_N_IMG))
echo "Output frame rate: " $OUTPUT_FRAME_RATE 

SEQUENCE_PATH=$SOURCE_BASE_DIR

for session in $SESSIONS_LIST; do
  printf -v SESSION_NUM_STR '%05d' "$((10#$session))"
  
  echo "*********************************"
  echo "*********************************"
  echo "Running on $SESSION_NUM_STR"
  echo "*********************************"
  echo "*********************************"
  
  for dir in ${DIRECTORIES_OF_INTEREST[@]}; do
    pushd $SOURCE_BASE_DIR/$SESSION_NUM_STR/$dir
    
    echo "*********************************"
    echo "Generating Video from $dir : $SESSION_NUM_STR"
    echo "*********************************"
    
    # Check desired video format
    if [ $VIDEO_FMT == "mp4" ]; then
      FILE_NAME=${SESSION_NUM_STR}_${dir}".mp4"
      # -crf sets the video quality (15-25), the lower the better
      ffmpeg -f image2 \
            -y \
            -framerate $OUTPUT_FRAME_RATE \
            -pattern_type glob \
            -start_number 0 \
            -r $OUTPUT_FRAME_RATE \
            -i '*.png' \
            -vcodec libx264 \
            -vf "select='not(mod(n,$USE_EVERY_N_IMG))'" \
            -crf 20 \
            -s $OUTPUT_SIZE $FILE_NAME \
            -loglevel 4

    elif [ $VIDEO_FMT == "avi" ]; then
      # -qscale:v sets the video quality (1-31), the lower the better
      FILE_NAME=${SESSION_NUM_STR}_${dir}".avi"
      ffmpeg -f image2 \
            -y \
            -framerate $OUTPUT_FRAME_RATE \
            -pattern_type glob \
            -start_number 0 \
            -r $OUTPUT_FRAME_RATE \
            -i '*.png' \
            -c:v mpeg4 \
            -vf "select='not(mod(n,$USE_EVERY_N_IMG))'" \
            -qscale:v 3 \
            -s $OUTPUT_SIZE $FILE_NAME \
            -loglevel 4

    else
      echo "Unknown video format: $VIDEO_FMT"
      exit
    fi

    popd
  done
done





