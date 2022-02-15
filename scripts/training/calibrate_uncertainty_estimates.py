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
Given a dataset of ground truth depth images as well as predicted depth images, and predicted (uncalibrated) uncertainty estimates, this script learns a linear scaling of the uncertainty estimates to minimize the negative log-likelihood of the predicted depth images on the training data.
"""

__author__ = "Sadegh Rabiee"
__version__ = "1.0.0"


import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
import argparse
from depth_utilities import *
from dataloader.unc_calibration_dataset import UncCalibDataset
from models.GANet_unc_calib import GANet_unc_calib_linear
from models.GANet_unc_calib import MyGaussianNLLLoss
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


def train_model(
        model, training_data_loader, criterion, optimizer, use_gpu, epoch_num):
  """
  Trains the model for a single epoch
  """
  model.train()  # Sets the module in training mode.
  epoch_loss = 0

  for batch_idx, batch in enumerate(training_data_loader):
    if use_gpu:
      depth_img_gt, depth_img_pred, depth_img_pred_unc, mask = batch['gt_depth'].cuda(
      ), batch['pred_depth'].cuda(), batch['pred_unc'].cuda(), batch['mask'].cuda()
    else:
      depth_img_gt, depth_img_pred, depth_img_pred_unc, mask = batch[
          'gt_depth'], batch['pred_depth'], batch['pred_unc'], batch['mask']

    optimizer.zero_grad()
    with torch.set_grad_enabled(True):
      calibrated_unc = model(depth_img_pred_unc)
      loss = criterion(depth_img_pred[mask],
                       depth_img_gt[mask], torch.pow(calibrated_unc[mask], 2))
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()
      # with torch.no_grad():
      #   for p in model.parameters():
      #     p.data.clamp_(min=0)

    if batch_idx % 10 == 0:
      print("===> Epoch[{}]({}/{}): Loss: {:.6f})".format(epoch_num,
            batch_idx, len(training_data_loader), loss.item()))

  return epoch_loss / (batch_idx + 1)


def validate_model(model,
                   validation_data_loader,
                   criterion,
                   use_gpu,
                   epoch_num):
  """
  Validates the model
  """
  model.eval()
  val_loss = 0
  for batch_idx, batch in enumerate(validation_data_loader):
    if use_gpu:
      depth_img_gt, depth_img_pred, depth_img_pred_unc, mask = batch['gt_depth'].cuda(
      ), batch['pred_depth'].cuda(), batch['pred_unc'].cuda(), batch['mask'].cuda()
    else:
      depth_img_gt, depth_img_pred, depth_img_pred_unc, mask = batch[
          'gt_depth'], batch['pred_depth'], batch['pred_unc'], batch['mask']

    with torch.set_grad_enabled(False):
      calibrated_unc = model(depth_img_pred_unc)
      loss = criterion(depth_img_pred[mask],
                       depth_img_gt[mask], torch.pow(calibrated_unc[mask], 2))
      val_loss += loss.item()

  return val_loss / (batch_idx + 1)


def train(model, training_data_loader, validation_data_loader, criterion, optimizer, save_path, use_gpu, epochs=100):
  """
  Trains the model for a given number of epochs
  """
  for epoch in range(epochs):
    # Train the model
    train_loss = train_model(
        model, training_data_loader, criterion, optimizer, use_gpu, epoch)
    # Validate the model
    val_loss = validate_model(
        model, validation_data_loader, criterion, use_gpu, epoch)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, val_loss))
    print('Model weights:', model.state_dict())

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/validation', val_loss, epoch)

    # Save the model
    torch.save(model.state_dict(), os.path.join(
        save_path, 'calib_unc_epoch_{}.pth'.format(epoch)))


def main():
  # Check command line arguments
  parser = argparse.ArgumentParser(
      description='Learns a linear scaling of the predicted uncertainty estimates given the actual depth prediction errors in the training data.')
  parser.add_argument('--pred_data_path', type=str,
                      help='Path to the base directory containing predicted depth images', required=True)
  parser.add_argument('--gt_data_path', type=str,
                      help='Path to the base directory containing ground truth depth images', required=True)
  parser.add_argument('--training_list', type=str,
                      default='./lists/sceneflow_train.list', help="training list", required=True)
  parser.add_argument('--val_list', type=str,
                      default='./lists/sceneflow_test_select.list', help="validation list", required=True)
  parser.add_argument('--save_path', type=str,
                      default='./checkpoint/', help="location to save models")
  parser.add_argument('--image_height', type=int, required=True)
  parser.add_argument('--image_width', type=int, required=True)
  parser.add_argument('--use_gpu',
                      type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                      help='Whether to use GPU', default=True,
                      required=False)
  parser.add_argument('--batch_size', type=int, default=1)
  parser.add_argument('--num_epochs', type=int, default=100)
  parser.add_argument('--num_workers', type=int, default=4)
  parser.add_argument('--subsample_factor', type=float, default=1.0)
  parser.add_argument(
      '--max_range', help='Only points with their depth smaller than this threshold will be used to learn the calibration parameters.', type=float, default=30.0)

  args = parser.parse_args()
  if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

  print(args)

  # torch.set_num_threads(2)
  # print("number of threads: ", torch.get_num_threads())

  # Load the datasets
  training_data = UncCalibDataset(gt_data_path=args.gt_data_path,
                                  prediction_data_path=args.pred_data_path,
                                  session_list=args.training_list,
                                  image_size=(args.image_height,
                                              args.image_width),
                                  max_range=args.max_range,
                                  subsample_factor=args.subsample_factor)
  validation_data = UncCalibDataset(gt_data_path=args.gt_data_path,
                                    prediction_data_path=args.pred_data_path,
                                    session_list=args.val_list,
                                    image_size=(args.image_height,
                                                args.image_width),
                                    max_range=args.max_range,
                                    subsample_factor=args.subsample_factor)
  training_data_loader = torch.utils.data.DataLoader(
      dataset=training_data, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
  validation_data_loader = torch.utils.data.DataLoader(
      dataset=validation_data, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

  model = GANet_unc_calib_linear()
  if args.use_gpu:
    model = torch.nn.DataParallel(model).cuda()

  criterion = MyGaussianNLLLoss(eps=1e-06)
  optimizer = torch.optim.Adam(
      model.parameters(), lr=0.001, betas=(0.9, 0.999))

  # Train the model
  train(model, training_data_loader, validation_data_loader,
        criterion, optimizer, args.save_path, args.use_gpu, args.num_epochs)
  writer.flush()
  writer.close()


if __name__ == "__main__":
  main()
