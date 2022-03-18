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
This script loads the ground truth depth images along with predicted depth images and predicted depth uncertainty that is output by a GANet model. It then computes pixel-wise error predictions given the predicted uncertainty distribution. The error predictions are evaluated by comparing against the ground truth pixel-wise error labels.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import argparse
from torchvision import transforms
import torch
import torch.nn as nn
from scipy import special
from sklearn.metrics import confusion_matrix
import cv2
from math import floor
from collections import OrderedDict
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
from matplotlib.backends.backend_pdf import PdfPages

from depth_utilities import read_pfm
from depth_utilities import colorize
from remote_monitor import send_notification_to_phone
from pixel_level_failure_detection.data_loader.load_images import DepthErrorDataset
from models.GANet_unc_calib import GANet_unc_calib_linear
from models.GANet_unc_calib import MyGaussianNLLLoss


# TODO:
# replace the name of the args.patch_dataset_path with args.pixelwise_dataset_path (path to the base directory with pixel-wise error images, e.g. mcdropout_off...)
# Extend the cmd line arguments
# Support batch_size of larger than 1

# Helper function to compute the NLL loss
def compute_nll(depth_img_pred, depth_img_gt, unc_img_pred, mask, loss_func):

  loss = loss_func(
      torch.from_numpy(depth_img_pred[mask]), torch.from_numpy(depth_img_gt[mask]), torch.from_numpy(np.power(unc_img_pred[mask], 2)))
  loss_vis = np.zeros(depth_img_pred.shape, dtype=float)
  loss_vis[mask] = loss.detach().cpu().numpy()

  return loss.mean(), loss_vis


# Helper function to draw the confusion matrix
def plot_confusion_matrix(cm, classes, file_name, file_path,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')

  print(cm)

  plt.figure("confusion_mat")
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()
  if not os.path.exists(file_path):
    os.makedirs(file_path)
  pp = PdfPages(file_path + '/' + file_name + '.pdf')
  plt.savefig(pp, format='pdf')
  pp.close()
  plt.close("confusion_mat")


def compute_image_scale_factor(first_img, second_img):
  """
  Given two input images, computes the scale factor by which the first image should be scaled to match the size of the second image, i.e. first_image.shape * scale_factor = second_image.shape.
  :return: same_aspect_ratio, scale factor
  """

  if len(first_img.shape) == 3:
    x_idx_first_img = 2
    y_idx_first_img = 1
  elif len(first_img.shape) == 2:
    x_idx_first_img = 1
    y_idx_first_img = 0
  else:
    print("Error: unexpected image shape: ", first_img.shape)
    exit

  if len(second_img.shape) == 3:
    x_idx_second_img = 2
    y_idx_second_img = 1
  elif len(second_img.shape) == 2:
    x_idx_second_img = 1
    y_idx_second_img = 0
  else:
    print("Error: unexpected image shape: ", first_img.shape)
    exit

  scale_factor = 1.0
  x_scale = second_img.shape[x_idx_second_img] / \
      first_img.shape[x_idx_first_img]
  y_scale = second_img.shape[y_idx_second_img] / \
      first_img.shape[y_idx_first_img]

  if x_scale != y_scale:
    same_aspect_ratio = False
    scale_factor = x_scale
  else:
    same_aspect_ratio = True
    scale_factor = x_scale

  return same_aspect_ratio, scale_factor


def visualize_patch_failure_predictions(patch_failure_pred_labels,
                                        gt_patch_multi_class_labels,
                                        patch_coords,
                                        rgb_image,
                                        output_folder_path,
                                        image_idx):
  """
  Draws a visualization of the predicted and ground truth labels for a patch on
  the original image. The predicted labels are drawn as circles on the center
  of the patch. The ground truth labels are drawn as outer rings around the
  prediction.
  :return:
  """
  failure_color = (0, 0, 255)  # red (BGR)
  non_failure_color = (0, 255, 0)  # green (BGR)

  prediction_circle_radius = 2  # 5, 2
  gt_circle_radius = 5  # 10, 5
  thickness = -1

  gt_binary_labels = np.logical_or(
      gt_patch_multi_class_labels == 2, gt_patch_multi_class_labels == 3)

  i = 0
  image = rgb_image.copy()
  image = np.moveaxis(image, [0, 1, 2], [2, 0, 1]) * 255.0
  image = np.ascontiguousarray(image, dtype=np.uint8)
  for i in range(patch_coords.shape[0]):
    gt_color = non_failure_color if gt_binary_labels[i] == 0 else failure_color
    pred_color = non_failure_color if patch_failure_pred_labels[i] == 0 else failure_color

    center_coordinates = (patch_coords[i, 1], patch_coords[i, 0])
    image = cv2.circle(image, center_coordinates,
                       gt_circle_radius, gt_color, thickness)
    image = cv2.circle(image, center_coordinates,
                       prediction_circle_radius, pred_color, thickness)
    i += 1

  # Create output folder if it does not exist
  if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
  output_image_path = os.path.join(output_folder_path, image_idx + ".png")
  cv2.imwrite(output_image_path,
              image)


def compute_patch_wise_failure_predictions(pixel_wise_failure_pred, patch_coords_tensor, patch_size, patch_scale_factor, min_err_pixel_ratio):
  """
  Given the predicted failure label for each pixel in the image, computes the failure prediction for each patch in a list of given patch coordinates.
  :param patch_scale_factor: the scale factor by which the patch size is multiplied to obtain the size of the patch in the pixel_wise_failure_pred/predicted depth image.
  """
  patch_coords = patch_coords_tensor.numpy()

  scaled_patch_size = patch_size * patch_scale_factor
  half_patch_size = floor(scaled_patch_size / 2)
  patch_failure_pred_labels = np.zeros(
      patch_coords.shape[0], dtype=np.uint8)

  i = 0

  for i in range(patch_coords.shape[0]):
    patch_coord = patch_coords[i, :]
    patch_x_start = floor(
        patch_coord[1] * patch_scale_factor - half_patch_size)
    patch_y_start = floor(
        patch_coord[0] * patch_scale_factor - half_patch_size)
    patch_x_end = floor(patch_x_start + scaled_patch_size)
    patch_y_end = floor(patch_y_start + scaled_patch_size)

    assert patch_x_start >= 0 and patch_y_start >= 0 and patch_x_end < pixel_wise_failure_pred.shape[
        1] and patch_y_end < pixel_wise_failure_pred.shape[0], "Error: patch coordinates are out of bounds. Patch coords: {}:{}, {}:{} \n".format(patch_x_start, patch_x_end, patch_y_start, patch_y_end) + "patch coords: {} \n \n".format(patch_coords) + "patch coords TENSOR: {}".format(patch_coords_tensor)

    patch_pred_failure_count = np.sum(
        pixel_wise_failure_pred[patch_y_start:patch_y_end, patch_x_start:patch_x_end])
    patch_pred_failure_ratio = patch_pred_failure_count / (
        scaled_patch_size * scaled_patch_size)
    patch_failure_pred_labels[i] = 1 if patch_pred_failure_ratio > min_err_pixel_ratio else 0

    i += 1

  return patch_failure_pred_labels


def visualize_failure_predictions(failure_prediction, rgb_image, mask, output_folder_path, image_idx):
  """
  Colors the failure prediction for each pixel in the image and saves the resulting color image to file.
  :param failure_prediction: pixel-wise failure prediction
  :param rgb_image: the input rgb image
  :return:
  """
  alpha = 0.65

  tmp = np.logical_and(failure_prediction <= 1, failure_prediction >= 0)
  assert np.all(tmp), "Error: failure prediction is not in [0, 1]. failure_prediction = {}".format(
      failure_prediction)

  rgb_image = np.moveaxis(rgb_image, [0, 1, 2], [2, 0, 1])
  rgb_image = (cv2.resize(
      rgb_image,
      (failure_prediction.shape[1], failure_prediction.shape[0]))
      * 255).astype(np.uint8)

  assert failure_prediction.shape[0] == rgb_image.shape[0], "Failure prediction and rgb image must have the same shape."
  assert failure_prediction.shape[1] == rgb_image.shape[1], "Failure prediction and rgb image must have the same shape."

  gray_img = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
  gray_img_overlaid = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGBA)
  gray_img_overlaid_masked = gray_img_overlaid.copy()

  failure_prediction_color = np.zeros(gray_img_overlaid.shape, dtype=np.uint8)
  failure_prediction_color[failure_prediction] = [0, 0, 255, 255]
  failure_prediction_color[np.logical_not(failure_prediction)] = [
      0, 255, 0, 255]

  # Generate the image with failure prediction overlaid
  cv2.addWeighted(failure_prediction_color, alpha, gray_img_overlaid, 1 - alpha,
                  0, gray_img_overlaid)

  # Generate the same image but color the masked area differently
  failure_prediction_color_masked = failure_prediction_color
  failure_prediction_color_masked[np.logical_not(mask)] = [255, 0, 0, 255]
  cv2.addWeighted(failure_prediction_color_masked, alpha,
                  gray_img_overlaid_masked, 1 - alpha, 0, gray_img_overlaid_masked)

  # Generate the output path for the masked version of the image:
  output_base_path = os.path.basename(output_folder_path)
  output_dir_name = os.path.dirname(output_folder_path)
  output_folder_path_masked = os.path.join(
      output_dir_name, output_base_path + "_masked")

  # Create output folder if it does not exist
  if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
  if not os.path.exists(output_folder_path_masked):
    os.makedirs(output_folder_path_masked)
  output_image_path = os.path.join(output_folder_path, image_idx + ".png")
  output_image_path_masked = os.path.join(
      output_folder_path_masked, image_idx + ".png")

  cv2.imwrite(output_image_path,
              gray_img_overlaid)
  cv2.imwrite(output_image_path_masked,
              gray_img_overlaid_masked)


def visualize_scalar_img_on_rgb(scalar_img, rgb_image, output_folder_path, image_idx, max_unc_threshold=10.0):
  """
  Visualizes the magnitude of the input scalar image for each pixel in the image.
  :param scalar_img: An HXW image of scalar values.
  :param rgb_image: the input rgb image
  :return:
  """
  alpha = 0.5
  MAX_UNC_THRESH_VISUALIZATION = max_unc_threshold

  rgb_image = np.moveaxis(rgb_image, [0, 1, 2], [2, 0, 1])
  rgb_image = (cv2.resize(
      rgb_image,
      (scalar_img.shape[1], scalar_img.shape[0]))
      * 255).astype(np.uint8)

  assert scalar_img.shape[0] == rgb_image.shape[0], "Scalar img and rgb image must have the same shape."
  assert scalar_img.shape[1] == rgb_image.shape[1], "Scalar img and rgb image must have the same shape."

  gray_img = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
  gray_img_overlaid = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGBA)

  unc_img = colorize(
      scalar_img, plt.get_cmap('viridis'), 0, MAX_UNC_THRESH_VISUALIZATION)
  unc_img = np.uint8(unc_img * 255)
  unc_img = cv2.cvtColor(unc_img, cv2.COLOR_RGBA2BGRA)

  gray_img_overlaid = gray_img_overlaid.astype(
      np.uint8)
  cv2.addWeighted(unc_img,
                  alpha, gray_img_overlaid, 1 - alpha,
                  0, gray_img_overlaid)

  # Create output folder if it does not exist
  if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
  output_image_path = os.path.join(output_folder_path, image_idx + ".png")
  cv2.imwrite(output_image_path,
              gray_img_overlaid)


def convert_unc_to_failure_prediction(unc_image, pred_depth_image, depth_err_thresh_abs, depth_err_thresh_relative):
  """
  Converts the predicted depth uncertainty for each pixel in the image
  to a failure prediction class label (binary failure/success). To do this, the
  predicted PDF for the depth of each pixel is converted to a PMF (probability mass function). Then the outcome (failure/success) with the highest probability is selected.
  :param unc_image: predicted depth uncertainty
  :return: pixel-wise failure prediction
  """
  assert unc_image.shape == pred_depth_image.shape, "Predicted depth uncertainty and predicted depth images must have the same shape."

  abs_err_thresh = np.ones(unc_image.shape) * depth_err_thresh_abs
  rel_err_thresh = pred_depth_image * depth_err_thresh_relative
  per_pixel_failure_err_thresh = np.max(
      np.stack((abs_err_thresh, rel_err_thresh), axis=0), axis=0)

  # Integrate the PDF of the depth of each pixel to obtain the probability of the error being within the threshold bounds [-per_pixel_failure_err_thresh, per_pixel_failure_err_thresh].
  success_prob = special.erf(
      per_pixel_failure_err_thresh / (unc_image * np.sqrt(2)))
  failure_prob = 1 - success_prob

  # (2 X W X H)
  failure_prediction_prob = np.stack((success_prob, failure_prob), axis=0)

  # (W X H)
  failure_prediction_class = failure_prob > 0.5
  return failure_prediction_class, failure_prediction_prob


def load_and_scale_gt_depth(gt_base_path, gt_depth_folder, target_img_size, img_idx, session_name):
  """
  Loads the ground-truth depth image and resizes it.
  """
  depth_image_name = img_idx + ".pfm"
  depth_path = os.path.join(
      gt_base_path, session_name, gt_depth_folder, depth_image_name)

  depth_image, scale = read_pfm(depth_path)
  depth_image = cv2.resize(
      depth_image, (target_img_size[1], target_img_size[0]))

  return depth_image


def load_predicted_uncertainty(pred_unc_path, pred_depth_unc_folder, pred_depth_folder, img_idx, session_name):
  """
  Loads the predicted depth uncertainty given the name of the input RGB image.
  :param pred_unc_path: Path to the directory that contains the GANet predictions.
  :param pred_depth_unc_folder: Name of the folder containing predicted depth uncertainty images
  :param image_name
  :param session_name
  :return: predicted_depth_uncertainty_image, predicted_depth_image
  """

  unc_image_name = img_idx + ".pfm"
  depth_image_name = img_idx + ".pfm"

  depth_unc_path = os.path.join(
      pred_unc_path, session_name, pred_depth_unc_folder, unc_image_name)
  depth_path = os.path.join(
      pred_unc_path, session_name, pred_depth_folder, depth_image_name)

  unc_image, scale = read_pfm(depth_unc_path)
  depth_image, scale = read_pfm(depth_path)

  return unc_image, depth_image


def convert_image_name_to_index(image_name):
  """
  Converts the image name to an index.
  :param image_name
  :return: index
  """
  image_index = image_name.rstrip(".png")

  return image_index


def main():
  parser = argparse.ArgumentParser(
      description='This script loads the ground truth depth images along with predicted depth images and predicted depth uncertainty that is output by a GANet model evaluates the performance of the model at both pixel level and patch level.')
  parser.add_argument('--gt_data_path', type=str,
                      help='Path to the base directory containing ground truth depth images', required=True)
  parser.add_argument('--gt_depth_folder', type=str,
                      help='Name of the folder containing ground-truth depth images', required=False, default="img_depth")
  parser.add_argument('--predictions_base_path', type=str,
                      help='Path to the base directory that includes the GANet predictions.', required=True)
  parser.add_argument('--pred_depth_unc_folder', type=str,
                      help='Name of the folder containing predicted depth uncertainty images', required=True, default="depth_uncertainty_pred")
  parser.add_argument('--pred_depth_folder', type=str,
                      help='Name of folder containing predicted depth images', required=True, default="img_depth_pred")
  parser.add_argument('--output_pred_failure_vis_folder', type=str,
                      help='Name of folder to save visualization of predicted failures in. The folder will be created under predictions_base_path. ', required=True, default="failure_pred_vis")
  parser.add_argument('--output_pred_failure_patch_vis_folder', type=str,
                      help='Name of folder to save visualization of predicted failures for image patches in. The folder will be created under predictions_base_path. ', required=False, default="failure_pred_patch_vis")
  parser.add_argument('--patch_dataset_name', type=str,
                      help='Name of the image patch dataset to evaluate the predictions on.', required=True)
  parser.add_argument('--patch_dataset_path', type=str,
                      help='Path to the image patch dataset (generated from running IVOA data pre-processing on images to generate error labels for image patches.)', required=True)

  parser.add_argument('--depth_err_thresh_abs', type=float,
                      help='Absolute depth error threshold.', default=1.5, required=True)
  parser.add_argument('--depth_err_thresh_relative', type=float,
                      help='Relative depth error threshold.',
                      default=0.3, required=True)
  parser.add_argument('--min_err_pixel_ratio', type=float, default=0.1,
                      help='The minimum ratio of pixels with depth estimation error in an image patch in order to label the patch as an instance of depth error')
  parser.add_argument('--save_failure_pred_vis_images', default=False,
                      type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                      help='Whether to save visualization images of predicted failures.',
                      required=True)
  parser.add_argument('--unc_calibration_model', type=str,
                      help='Path to the calibration model for the uncertainty.', required=False, default=None)

  parser.add_argument('--patch_size', type=int, required=True)
  parser.add_argument('--num_workers', type=int, default=4)
  parser.add_argument('--max_range', type=float,
                      help='Max depth range to be considered for evaluation.', required=False, default=30.0)
  parser.add_argument('--min_range', type=float,
                      help='Minimum depth range to be considered for evaluation.', required=False, default=1.0)
  parser.add_argument('--batch_size', type=int, default=1)

  args = parser.parse_args()

  # TODO: Support batch-size of larger than 1.
  if args.batch_size > 1:
    print("ERROR - Batches of size > 1 are not supported.")

  # Save visualization of depth uncertainty (as opposed to disparity on the original images)
  VISUALIZE_DEPTH_UNCERTAINTY = False
  VISUALIZE_NLL = False
  loss_func = MyGaussianNLLLoss(eps=1e-06, reduction="none")
  loss_func_classification = nn.NLLLoss(ignore_index=-1)

  # TODO: Add these to cmd line args
  session_prefix_length = 5
  IMG_HEIGHT = 300
  IMG_WIDTH = 480
  CONFUSION_MATRIX_COMPUTATION_BATCH_SIZE = 300
  # OUTPUT_CNF_NAME_SUFFIX = "_manual_calibration_scalar_120_PIXELWISE"
  OUTPUT_CNF_NAME_SUFFIX = "_auto_calibration_PIXELWISE"

  # Retrieve the session number list give the dataset name.
  test_set_dict = {
      "test_tmp": [1007],
      "test_01_ganet_v0": [1007, 1012, 1017, 1022, 1027, 1032, 2007, 2012, 2017, 2022, 2027, 2032]
  }
  assert args.patch_dataset_name in test_set_dict, "Invalid dataset name."
  session_num_list = test_set_dict[args.patch_dataset_name]

  data_transform_input = transforms.Compose([
      transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
      transforms.ToTensor()
  ])
  data_transform_target = data_transform_input
  test_dataset = DepthErrorDataset(args.patch_dataset_path,
                                   args.gt_data_path,
                                   session_num_list,
                                   loaded_image_color=True,
                                   output_image_color=True,
                                   session_prefix_length=session_prefix_length,
                                   raw_img_folder="img_left",
                                   label_img_folder="depth_err_labels_abs1.0_rel_0.2_maxRange30_ds",
                                   mask_img_folder="valid_pixel_mask_abs1.0_rel_0.2_maxRange30_ds",
                                   transform_input=data_transform_input,
                                   transform_target=data_transform_target,
                                   load_masks=True,
                                   regression_mode=False,
                                   binarize_target=True,
                                   no_meta_data_available=True,
                                   load_only_with_labels=True)

  data_loaders = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size, num_workers=args.num_workers)

  unc_calib_model = None
  if args.unc_calibration_model is not None:
    unc_calib_model = GANet_unc_calib_linear()

    # Map to CPU as you load the model
    state_dict = torch.load(args.unc_calibration_model,
                            map_location=lambda storage, loc: storage)
    was_trained_with_multi_gpu = False
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
      if k.startswith('module'):
        was_trained_with_multi_gpu = True
        name = k[7:]  # remove 'module.'
        new_state_dict[name] = v
        print(name)

    # # TODO: temporary testing. Remove this
    # # tmp_bias = 0.0
    # tmp_scalar = 120.0  # 20.16
    # print("Original calib model weights:", state_dict)
    # if(was_trained_with_multi_gpu):
    #   #   new_state_dict['weights'][0] = tmp_bias
    #   new_state_dict['weights'][1] = tmp_scalar
    #   print("Modified calib model weights:", new_state_dict)
    # else:
    #   #   state_dict['weights'][0] = tmp_bias
    #   state_dict['weights'][1] = tmp_scalar
    #   print("Modified calib model weights:", state_dict)

    if(was_trained_with_multi_gpu):
      unc_calib_model.load_state_dict(new_state_dict)
    else:
      unc_calib_model.load_state_dict(state_dict)

    print("State dict of loaded model: ")
    print(unc_calib_model.state_dict())
    # unc_calib_model = unc_calib_model.cuda()

  all_failure_predictions = np.array([], dtype=np.int_)
  all_binary_labels = np.array([], dtype=np.int_)
  cnf_matrix = np.zeros((2, 2), dtype=np.int_)
  valid_data_points_count = 0

  print("Dataset size: ", len(data_loaders))
  dataset_size = len(data_loaders)

  # The length of the session name prefix
  if (session_prefix_length == 2):
    session_name_format = "{0:02d}"
  elif (session_prefix_length == 5):
    session_name_format = "{0:05d}"
  else:
    print("ERROR: Session prefix length of %d is not supported.",
          session_prefix_length)
    exit()

  running_loss_classification = 0.0
  # Go through the dataset
  for iteration, batch in enumerate(tqdm(data_loaders), 0):

    valid_pixel_mask = torch.squeeze(batch["mask_img"][0]).numpy()
    session_num = batch["session"][0].item()
    session_name = session_name_format.format(session_num)
    img_idx = convert_image_name_to_index(
        batch["img_name"][0])

    # Load the predicted depth uncertainty
    unc_image, pred_depth_image = load_predicted_uncertainty(
        args.predictions_base_path, args.pred_depth_unc_folder, args.pred_depth_folder, img_idx, session_name)

    if VISUALIZE_DEPTH_UNCERTAINTY:
      visualize_scalar_img_on_rgb(
          unc_image, batch["img"][0].numpy(), os.path.join(args.predictions_base_path, session_name, 'depth_uncertainty_vis'), img_idx)
      visualize_scalar_img_on_rgb(
          unc_image / pred_depth_image, batch["img"][0].numpy(), os.path.join(args.predictions_base_path, session_name, 'depth_uncertainty_rel_vis'), img_idx, max_unc_threshold=1.0)

    # Use the learned calibration model to calibrate the predicted depth uncertainty
    if unc_calib_model is not None:
      unc_image = torch.sqrt(unc_calib_model(
          torch.pow(torch.tensor(unc_image), 2)))
      unc_image = unc_image.detach().cpu().numpy()

    # Compute NLL given the predicted depth and the associated uncertainty
    if VISUALIZE_NLL and (unc_calib_model is not None):
      gt_depth_img = load_and_scale_gt_depth(
          args.gt_data_path, args.gt_depth_folder, pred_depth_image.shape, img_idx, session_name)

      loss, loss_img = compute_nll(pred_depth_image, gt_depth_img,
                                   unc_image, valid_pixel_mask, loss_func)
      visualize_scalar_img_on_rgb(
          loss_img, batch["img"][0].numpy(), os.path.join(args.predictions_base_path, session_name, 'loss_after_calib_vis'), img_idx, max_unc_threshold=500.0)

    # Convert the predicted depth uncertainty to per pixel binary failure predictions as well as predicted failure probabilities
    failure_prediction, failure_prediction_prob = convert_unc_to_failure_prediction(
        unc_image, pred_depth_image, args.depth_err_thresh_abs, args.depth_err_thresh_relative)
    if args.save_failure_pred_vis_images:
      visualize_failure_predictions(
          failure_prediction, batch["img"][0].numpy(), valid_pixel_mask, os.path.join(args.predictions_base_path, session_name, args.output_pred_failure_vis_folder), img_idx)
      if VISUALIZE_DEPTH_UNCERTAINTY and (unc_calib_model is not None):
        visualize_scalar_img_on_rgb(
            unc_image, batch["img"][0].numpy(), os.path.join(args.predictions_base_path, session_name, 'depth_uncertainty_calib_vis'), img_idx)
        visualize_scalar_img_on_rgb(
            unc_image / pred_depth_image, batch["img"][0].numpy(), os.path.join(args.predictions_base_path, session_name, 'depth_uncertainty_rel_calib_vis'), img_idx, max_unc_threshold=1.0)

    # Compute the NLL for failure predictions (as opposed to depth estimates)
    failure_prediction_prob_tensor = torch.tensor(
        failure_prediction_prob, dtype=float)
    failure_prediction_prob_tensor = torch.unsqueeze(
        failure_prediction_prob_tensor, dim=0)
    loss_classification = loss_func_classification(
        failure_prediction_prob_tensor, batch['labels'])
    loss_classification = loss_classification.item()
    running_loss_classification += loss_classification

    same_aspect_ratio, rgb_to_depth_scale_factor = compute_image_scale_factor(
        batch["img"][0].numpy(), unc_image)
    assert same_aspect_ratio, "The RGB image and the depth uncertainty image do not have the same aspect ratio."

    mask = torch.squeeze(batch['mask_img'], 1)
    curr_valid_labels = batch["labels"][mask
                                        ].cpu().numpy().astype(np.int_)
    curr_valid_predictions = failure_prediction[mask[
        0].cpu().numpy()].astype(np.int_)

    all_failure_predictions = np.concatenate(
        (all_failure_predictions, curr_valid_predictions), 0)
    all_binary_labels = np.concatenate(
        (all_binary_labels, curr_valid_labels), 0)

    if iteration % CONFUSION_MATRIX_COMPUTATION_BATCH_SIZE == 0 and iteration > 0:
      # Compute the confusion matrix for current batch and add it to the total confusion matrix
      cnf_matrix = cnf_matrix + confusion_matrix(
          all_binary_labels, all_failure_predictions)
      print("Current confusion matrix:")
      print(cnf_matrix)

      # Reset the arrays
      valid_data_points_count += all_binary_labels.size
      all_failure_predictions = np.array([], dtype=np.int_)
      all_binary_labels = np.array([], dtype=np.int_)

  # Evalute the failure predictions
  if all_binary_labels.size > 0:
    cnf_matrix = cnf_matrix + confusion_matrix(all_binary_labels,
                                               all_failure_predictions)
    valid_data_points_count += all_binary_labels.size

  print('Total loss (classification): ',
        running_loss_classification / iteration)
  print('Total iteration', iteration)

  print("All valid data points count:", valid_data_points_count)

  print("Confusion matrix:")
  print(cnf_matrix)

  plot_confusion_matrix(
      cnf_matrix,
      ['NF', 'F'],
      "cnfMat_binary" + args.patch_dataset_name +
      OUTPUT_CNF_NAME_SUFFIX, args.predictions_base_path,
      normalize=True)

  msg = "Running evaluate_uncertainty_predictions.py finished. Results saved to {}".format(
      args.predictions_base_path)
  send_notification_to_phone(msg, 'Unc. Eval. Job Finished')


if __name__ == '__main__':
  main()
