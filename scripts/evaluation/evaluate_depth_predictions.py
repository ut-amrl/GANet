#!/bin/python

# ========================================================================
# Copyright 2022 srabiee@cs.utexas.edu
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
Given a folder of predicted depth images, a folder of ground truth depth images, and evaluation parameters and thresholds this script will evaluate the predicted depth images and save visualizations of predictions errors to file.
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
parser.add_argument('--rgb_img_folder', type=str,
                    help='Folder containing the RGB images', required=True)
parser.add_argument('--pred_depth_folder', type=str,
                    help='Folder containing predicted depth images', required=True)
parser.add_argument('--gt_depth_folder', type=str,
                    help='Folder containing ground truth depth images', required=True)
parser.add_argument('--output_err_vis_folder', type=str, required=True)
parser.add_argument('--max_range', type=float,
                    help='Max depth range to be considered for evaluation.', required=True, default=35.0)
parser.add_argument('--min_range', type=float,
                    help='Minimum depth range to be considered for evaluation.', required=True, default=1.0)
parser.add_argument('--depth_err_thresh_abs', type=float,
                    help='Minimum depth range to be considered for evaluation.', default=1.5, required=True)
parser.add_argument('--depth_err_thresh_relative', type=float,
                    help='Minimum depth range to be considered for evaluation.',
                    default=0.3, required=True)
parser.add_argument('--image_height', type=int, required=True)
parser.add_argument('--image_width', type=int, required=True)


def evaluate_depth_prediction(depth_img_pred, depth_img_gt, max_range, min_range, err_thresh_abs, err_thresh_rel):

  # Check if the images are of the same size
  if depth_img_gt.shape != depth_img_pred.shape:
    print("Error: Images are not of the same size")
    exit()

  # Compute the valid pixel mask
  valid_pixel_mask = np.logical_and(
      depth_img_gt > min_range, depth_img_gt < max_range)

  # Compute the depth error
  depth_err = depth_img_pred - depth_img_gt
  depth_err_relative = depth_err / depth_img_gt

  # Label the depth error into false positives, false negatives, and true positives
  false_positives = np.logical_and(
      depth_err < -err_thresh_abs, depth_err_relative < -err_thresh_rel)
  false_negatives = np.logical_and(
      depth_err > err_thresh_abs, depth_err_relative > err_thresh_rel)
  true_positives = np.logical_and(np.logical_not(
      false_negatives), np.logical_not(false_positives))

  evaluation_results = {
      'false_positives': false_positives,
      'false_negatives': false_negatives,
      'true_positives': true_positives,
      'valid_pixel_mask': valid_pixel_mask
  }
  return evaluation_results


def visualize_depth_prediction_errors(rgb_img, evaluation_results, output_path):
  alpha = 0.65
  fn_color = (0, 0, 255, 255)  # BGRA red
  fp_color = (0, 255, 253, 255)  # BGRA yellow
  tp_color = (0, 255, 0, 255)  # BGRA green
  invalid_color = (255, 0, 0, 255)  # BGRA blue

  fp_mask = evaluation_results['false_positives']
  fn_mask = evaluation_results['false_negatives']
  tp_mask = evaluation_results['true_positives']
  valid_pixel_mask = evaluation_results['valid_pixel_mask']

  rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2BGRA)

  # Assert that the input RGB image has an alpha channel
  assert rgb_img.shape[2] == 4, "Error: RGB image does not have an alpha channel. RGB image size: {}".format(
      rgb_img.shape)

  # Create a partially transparent flat colored image for false positives
  fp_img = np.zeros(rgb_img.shape, dtype=np.uint8)
  fp_img[:, :, 0] = fp_color[0] * fp_mask
  fp_img[:, :, 1] = fp_color[1] * fp_mask
  fp_img[:, :, 2] = fp_color[2] * fp_mask
  fp_img[:, :, 3] = 255 * fp_mask * 1.0

  # Create a partially transparent flat colored image for false negatives
  fn_img = np.zeros(rgb_img.shape, dtype=np.uint8)
  fn_img[:, :, 0] = fn_color[0] * fn_mask
  fn_img[:, :, 1] = fn_color[1] * fn_mask
  fn_img[:, :, 2] = fn_color[2] * fn_mask
  fn_img[:, :, 3] = 255 * fn_mask * 1.0

  # Create a partially transparent flat colored image for true positives
  tp_img = np.zeros(rgb_img.shape, dtype=np.uint8)
  tp_img[:, :, 0] = tp_color[0] * tp_mask
  tp_img[:, :, 1] = tp_color[1] * tp_mask
  tp_img[:, :, 2] = tp_color[2] * tp_mask
  tp_img[:, :, 3] = 255 * tp_mask * 1.0

  invalid_pixel_mask = np.logical_not(valid_pixel_mask)

  # label images overlaid
  label_img = np.zeros(rgb_img.shape, dtype=np.uint8)
  # label_img
  label_img = tp_img + fn_img + fp_img

  # label_img[invalid_pixel_mask_4d] = invalid_color
  label_img[:, :, 0][invalid_pixel_mask] = invalid_color[0]
  label_img[:, :, 1][invalid_pixel_mask] = invalid_color[1]
  label_img[:, :, 2][invalid_pixel_mask] = invalid_color[2]
  label_img[:, :, 3][invalid_pixel_mask] = 255

  # Overlay the false positives, false negatives, and true positives on a grayscale version of the RGB image
  gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
  gray_img_overlaid = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGBA)

  cv2.addWeighted(label_img, alpha, gray_img_overlaid, 1 - alpha,
                  0, gray_img_overlaid)

  cv2.imwrite(output_path, gray_img_overlaid)


def main():
  args = parser.parse_args()
  depth_folder_gt = args.gt_depth_folder
  depth_folder_pred = args.pred_depth_folder

  if not os.path.exists(args.output_err_vis_folder):
    os.makedirs(args.output_err_vis_folder)
  for filename in os.listdir(depth_folder_pred):
    if filename.endswith(".pfm"):

      # Remove the .pfm extension
      base_filename = filename[:-4]
      # Check if the corresponding RGB image exists
      rgb_img_path = os.path.join(args.rgb_img_folder, base_filename + ".png")
      if not os.path.exists(rgb_img_path):
        print("Error: RGB image not found for depth image: " + rgb_img_path)
        exit()

      # Check if the corresponding ground truth depth image exists
      depth_img_gt_path = os.path.join(depth_folder_gt, filename)
      if not os.path.exists(depth_img_gt_path):
        print("Error: Ground truth depth image does not exist: " + depth_img_gt_path)
        exit()

      depth_img_gt, scale = read_pfm(depth_img_gt_path)
      depth_img_pred, scale = read_pfm(
          os.path.join(depth_folder_pred, filename))
      rgb_img = cv2.imread(rgb_img_path)

      # Resize the images to the given size
      depth_img_gt = cv2.resize(
          depth_img_gt, (args.image_width, args.image_height))
      depth_img_pred = cv2.resize(
          depth_img_pred, (args.image_width, args.image_height))
      rgb_img = cv2.resize(rgb_img, (args.image_width, args.image_height))

      # Evalute the predicted depth image
      evaluation_results = evaluate_depth_prediction(
          depth_img_pred, depth_img_gt, args.max_range, args.min_range, args.depth_err_thresh_abs, args.depth_err_thresh_relative)

      # Remove the .pfm extension
      output_file_path = os.path.join(
          args.output_err_vis_folder, base_filename + ".png")

      # Visualize the depth prediction errors
      visualize_depth_prediction_errors(
          rgb_img, evaluation_results, output_file_path)


if __name__ == "__main__":
  main()
