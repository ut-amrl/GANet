#!/bin/bash

# This script is used to generate disparity map from depth map.

BASELINE="0.6"
FX="480.0"

SOURCE_DISPARITY_FOLDER="/robodata/srabiee/AirSim_IVSLAM/cityenv_wb/04023/img_disp"
TARGET_DEPTH_FOLDER="/robodata/srabiee/AirSim_IVSLAM/cityenv_wb/04023/img_depth_from_disp"


python ../generate_depth_from_disparity.py \
--baseline=$BASELINE \
--fx=$FX \
--depth_folder=$TARGET_DEPTH_FOLDER \
--disparity_folder=$SOURCE_DISPARITY_FOLDER 
