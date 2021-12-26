#!/bin/bash

# This script is used to generate disparity map from depth map.

BASELINE="0.6"
FX="480.0"

SOURCE_DEPTH_FOLDER="/robodata/srabiee/AirSim_IVSLAM/cityenv_wb/04023/img_depth"
TARGET_DISPARITY_FOLDER="/robodata/srabiee/AirSim_IVSLAM/cityenv_wb/04023/img_disp"
TARGET_DISPARITY_COLOR_FOLDER="/robodata/srabiee/AirSim_IVSLAM/cityenv_wb/04023/img_disp_color"

# SOURCE_DEPTH_FOLDER="/robodata/srabiee/AirSim_IVSLAM/cityenv_wb/04023/img_depth_from_disp"
# TARGET_DISPARITY_FOLDER="/robodata/srabiee/AirSim_IVSLAM/cityenv_wb/04023/img_disp_from_depth_tmp"


python ../generate_disparity_from_depth.py \
--baseline=$BASELINE \
--fx=$FX \
--depth_folder=$SOURCE_DEPTH_FOLDER \
--disparity_folder=$TARGET_DISPARITY_FOLDER \
--disparity_color_folder=$TARGET_DISPARITY_COLOR_FOLDER
