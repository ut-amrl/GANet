#!/bin/python

# ========================================================================
# Copyright 2021 srabiee@cs.utexas.edu
# Department of Computer Science,
# University of Texas at Austin


# This software is free: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License Version 3,
# as published by the Free Software Foundation.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# Version 3 in the file COPYING that came with this distribution.
# If not, see <http://www.gnu.org/licenses/>.
# ========================================================================

"""
Given a folder of depth images, this script will generate a folder of disparity images
"""

__author__ = "Sadegh Rabiee"
__license__ = "MIT"
__version__ = "1.0.0"


import numpy as np
import os
import sys
import argparse
import cv2
from depth_utilities import *
import matplotlib.pyplot as plt

# Check command line arguments
parser = argparse.ArgumentParser(
    description='Generate disparity images from depth images')
parser.add_argument('--depth_folder', type=str,
                    help='Folder containing depth images', required=True)
parser.add_argument('--disparity_folder', type=str,
                    help='Folder to save disparity images', required=True)
parser.add_argument('--disparity_color_folder', type=str,
                    help='Folder to save colorized disparity images', required=False, default=None)
parser.add_argument('--baseline', type=float,
                    help='Baseline of the stereo camera', required=True)
parser.add_argument('--fx', type=float,
                    help='Focal length of the stereo camera', required=True)
parser.add_argument('--max_disparity', type=float,
                    help='Maximum disparity value - used for normalization.', required=False, default=192.0)


def main():
  args = parser.parse_args()
  depth_folder = args.depth_folder
  disparity_folder = args.disparity_folder
  disparity_color_folder = args.disparity_color_folder
  baseline = args.baseline
  fx = args.fx
  max_disp = args.max_disparity

  # print(depth_folder)
  # print(disparity_folder)

  if not os.path.exists(disparity_folder):
    os.makedirs(disparity_folder)
  for filename in os.listdir(depth_folder):
    if filename.endswith(".pfm"):

      depth_img, scale = read_pfm(os.path.join(depth_folder, filename))
      disp_img = convert_depth_to_disparity(depth_img, baseline, fx, max_disp)

      # Save the disparity image to file
      # Remove the .pfm extension
      filename = filename[:-4]
      cv2.imwrite(os.path.join(disparity_folder, filename + ".png"), disp_img)

      # Save colorized disparity image
      if disparity_color_folder is not None:
        if not os.path.exists(disparity_color_folder):
          os.makedirs(disparity_color_folder)
        disp_img_color = colorize(
            disp_img, plt.get_cmap('viridis'), 0, 256)
        cv2.imwrite(os.path.join(disparity_color_folder,
                    filename + ".png"), disp_img_color * 256)


if __name__ == "__main__":
  main()
