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
Dataset for loading a dataset of predicted depth images, ground truth depth images, and the predicted uncalibrated uncertainty values.
"""

import os
from torch.utils.data import Dataset
from depth_utilities import *
import cv2


class UncCalibDataset(Dataset):
  def __init__(self, gt_data_path, prediction_data_path, session_list, image_size, max_range=30.0, subsample_factor=1):
    """
    Initialize the dataset.
    """
    super(UncCalibDataset, self).__init__()
    self.gt_data_path = gt_data_path
    f = open(session_list, 'r')
    self.session_list = f.readlines()
    self.image_size = image_size
    self.max_range = max_range

    self.gt_depth_folder_name = "img_depth"
    self.file_name_list = []
    self.gt_data_base_path_list = []
    self.pred_data_base_path_list = []

    # Go through all sessions and get the list of files
    for session in self.session_list:
      session = session.strip()
      # Get the list of all files in the session directory
      left_img_dir_path = os.path.join(session, self.gt_depth_folder_name)
      files = [f for f in os.listdir(
          left_img_dir_path) if os.path.isfile(os.path.join(left_img_dir_path, f)) and f.endswith('.pfm')]

      # Subsample files
      if subsample_factor < 1.0:
        # Sort the files in ascending order
        files.sort()
        sample_distance = int(1.0 / subsample_factor)
        files = files[::sample_distance]

      gt_data_base_paths = [session for f in files]

      # Concatenate the session name suffix from each entry in gt_data_base_paths to the prediction_data_path
      pred_data_base_path = os.path.join(
          prediction_data_path, os.path.basename(os.path.normpath(session)))
      # The base path is the same for all files in the same session. Repeat to the number of files in the session.
      pred_data_base_paths = [pred_data_base_path for f in files]

      self.file_name_list += files
      self.gt_data_base_path_list += gt_data_base_paths
      self.pred_data_base_path_list += pred_data_base_paths

    # Assert that prediction images exist for each ground truth image
    for i in range(len(self.file_name_list)):
      assert os.path.exists(os.path.join(
          self.pred_data_base_path_list[i], 'img_depth_pred', self.file_name_list[i])), "Prediction depth image does not exist: " + os.path.join(
          self.pred_data_base_path_list[i], 'img_depth_pred', self.file_name_list[i])
      assert os.path.exists(os.path.join(
          self.pred_data_base_path_list[i], 'depth_uncertainty_pred', self.file_name_list[i])), "Prediction uncertainty image does not exist: " + os.path.join(
          self.pred_data_base_path_list[i], 'depth_uncertainty_pred', self.file_name_list[i])

  def __len__(self):
    """
    Get the length of the dataset.
    """
    return len(self.file_name_list)

  def __getitem__(self, index):

    gt_depth_file_path = os.path.join(self.gt_data_base_path_list[index],
                                      'img_depth', self.file_name_list[index])
    pred_depth_file_path = os.path.join(self.pred_data_base_path_list[index],
                                        'img_depth_pred', self.file_name_list[index])
    pred_unc_file_path = os.path.join(self.pred_data_base_path_list[index],
                                      'depth_uncertainty_pred', self.file_name_list[index])

    depth_img_gt, scale = read_pfm(gt_depth_file_path)
    depth_img_pred, scale = read_pfm(pred_depth_file_path)
    depth_img_pred_unc, scale = read_pfm(pred_unc_file_path)

    # Entries of depth_unc must be non-negative
    if np.any(depth_img_pred_unc < 0):
      print(depth_img_pred_unc)
      raise ValueError("depth_img_pred_unc has negative entry/entries")

    depth_img_gt = cv2.resize(
        depth_img_gt, (self.image_size[1], self.image_size[0]))
    depth_img_pred = cv2.resize(
        depth_img_pred, (self.image_size[1], self.image_size[0]))
    depth_img_pred_unc = cv2.resize(
        depth_img_pred_unc, (self.image_size[1], self.image_size[0]))
    mask = np.logical_and(depth_img_gt > 0, depth_img_gt < self.max_range)

    sample = {'gt_depth': depth_img_gt,
              'pred_depth': depth_img_pred, 'pred_unc': depth_img_pred_unc, 'mask': mask}

    return sample
