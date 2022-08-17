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
Given a folder of disparity images, this script will generate a folder of depth images
"""

__author__ = "Sadegh Rabiee"
__license__ = "MIT"
__version__ = "1.0.0"


import numpy as np
import os
import sys
import argparse
import cv2
from tqdm import tqdm
from depth_utilities import *

# Check command line arguments
parser = argparse.ArgumentParser(
    description='Generate depth images from disparity images.')
parser.add_argument('--disparity_folder', type=str,
                    help='Folder to load disparity images from', required=True)
parser.add_argument('--depth_folder', type=str,
                    help='Output folder to save depth images', required=True)
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
  baseline = args.baseline
  fx = args.fx
  max_disp = args.max_disparity

  if not os.path.exists(depth_folder):
    os.makedirs(depth_folder)
  for filename in tqdm(os.listdir(disparity_folder)):
    if filename.endswith(".png"):

      # Load the image
      disparity_img = cv2.imread(os.path.join(
          disparity_folder, filename), cv2.IMREAD_GRAYSCALE)
      depth_img = convert_disparity_to_depth(
          disparity_img, baseline, fx, max_disp)

      # Remove the .png extension
      filename = filename[:-4]
      write_pfm(os.path.join(depth_folder, filename + ".pfm"), depth_img)


if __name__ == "__main__":
  main()
