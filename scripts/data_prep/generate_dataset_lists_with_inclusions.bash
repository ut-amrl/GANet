#!/bin/bash

# This script is used to generate training and testing dataset "session lists".
# Each line in a session list file contains the path to a folder that contains # diffeerent folders of RGB, depth and disparity images.

# The input to this script are the path to a few base directories as well as 
# the session numbers to be INCLUDED in the output dataset.


# ------------ TEST DATA OOD 00 ------------
# OUTPUT_SESSION_LIST_FILE_PATH="/robodata/srabiee/scratch/My_Repos/GANet/lists/airsim_wb_testing_OOD_00.list"
# BASE_SOURCE_DIR_LIST=(
# "/robodata/srabiee/AirSim_IVSLAM/cityenv_wb/" 
# )
# INCLUDED_SEQ_0=$(seq 4000 4040)
# for id in $INCLUDED_SEQ_0; do
#     INCLUDED_SEQ_FMT_0+=$( printf '%05d' $id )
#     INCLUDED_SEQ_FMT_0+=" "
# done
# INCLUDED_SESSION_NUMBERS_LIST=( "$INCLUDED_SEQ_FMT_0" )

# ------------ TEST DATA 00 ------------
# OUTPUT_SESSION_LIST_FILE_PATH="/robodata/srabiee/scratch/My_Repos/GANet/lists/airsim_wb_test_00.list"
# BASE_SOURCE_DIR_LIST=(
# "/robodata/srabiee/AirSim_IVSLAM/cityenv_wb/" 
# )
# INCLUDED_SEQ_0=" 1002 1007 1012 1017 1022 1027 1032 
#                  2002 2007 2012 2017 2022 2027 2032 
#                  3002 3007 3012 3017 3022 3027 3032 "
# for id in $INCLUDED_SEQ_0; do
#     INCLUDED_SEQ_FMT_0+=$( printf '%05d' $id )
#     INCLUDED_SEQ_FMT_0+=" "
# done
# INCLUDED_SESSION_NUMBERS_LIST=( "$INCLUDED_SEQ_FMT_0" )

# ------------ VALIDATION DATA 00 ------------
# OUTPUT_SESSION_LIST_FILE_PATH="/robodata/srabiee/scratch/My_Repos/GANet/lists/airsim_wb_validation_00.list"
# BASE_SOURCE_DIR_LIST=(
# "/robodata/srabiee/AirSim_IVSLAM/cityenv_wb/" 
# )
# INCLUDED_SEQ_0=" 1005 1010 1015 1020 
#                  2005 2010 2015 2020 
#                  3005 3010 3015 3020 "
# for id in $INCLUDED_SEQ_0; do
#     INCLUDED_SEQ_FMT_0+=$( printf '%05d' $id )
#     INCLUDED_SEQ_FMT_0+=" "
# done
# INCLUDED_SESSION_NUMBERS_LIST=( "$INCLUDED_SEQ_FMT_0" )

# ------------ SAMPLE TRAINING DATA 00 ------------
OUTPUT_SESSION_LIST_FILE_PATH="/robodata/srabiee/scratch/My_Repos/GANet/lists/airsim_wb_sample_training_00.list"
BASE_SOURCE_DIR_LIST=(
"/robodata/srabiee/AirSim_IVSLAM/cityenv_wb/" 
)
INCLUDED_SEQ_0=" 1008 "
for id in $INCLUDED_SEQ_0; do
    INCLUDED_SEQ_FMT_0+=$( printf '%05d' $id )
    INCLUDED_SEQ_FMT_0+=" "
done
INCLUDED_SESSION_NUMBERS_LIST=( "$INCLUDED_SEQ_FMT_0" )


# Remove the output files they already exists.
if [ -f $OUTPUT_SESSION_LIST_FILE_PATH ]; then
  rm $OUTPUT_SESSION_LIST_FILE_PATH
fi



# Go through the list of base directories and their corresponding excluded session numbers.
for i in ${!BASE_SOURCE_DIR_LIST[@]}; do
  BASE_SOURCE_DIR=${BASE_SOURCE_DIR_LIST[$i]}
  INCLUDED_SESSION_NUMBERS=${INCLUDED_SESSION_NUMBERS_LIST[$i]}
  echo "Processing base source directory: $BASE_SOURCE_DIR"
  # echo "Included session numbers: $INCLUDED_SESSION_NUMBERS"


  # Go through all the folders in the base directory and add to the list the path to those that are not excluded.
  for folder in $(ls $BASE_SOURCE_DIR); do
      # Skip if it is not a directory.
      if [ ! -d "$BASE_SOURCE_DIR/$folder" ]; then
          continue
      fi

      # Check if the folder is in the excluded list.
      if [[ $INCLUDED_SESSION_NUMBERS =~ $folder ]]; then
        echo "$BASE_SOURCE_DIR/$folder" >> $OUTPUT_SESSION_LIST_FILE_PATH
      fi
  done
done

