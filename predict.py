from __future__ import print_function
import argparse
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from math import log10
# from GCNet.modules.GCNet import L1Loss
import sys
import shutil
import os
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from models.GANet_deep import GANet
from dataloader.data import get_test_set
import numpy as np
import cv2
from depth_utilities import colorize
from depth_utilities import convert_disparity_unc_to_depth_unc
from depth_utilities import write_pfm
from remote_monitor import send_notification_to_phone
import matplotlib.pyplot as plt
import time


def has_dropout(model):
  """
  Helper function to check if the model has nay dropout layers.
  NOTE: This is not a fool-proof way to check though, because if the initialized model architecture includes dropouts, but the trained
  model whose parameters are loaded does not, then this check still passes.
  """
  for mod in model.modules():
    if mod.__class__.__name__.startswith('Dropout'):
      return True
  return False


def set_dropout_to_train(model):
  """
  Helper function to set all dropout layers to train mode
  """
  for mod in model.modules():
    if mod.__class__.__name__.startswith('Dropout'):
      mod.train()


def find_least_multiple_larger_than(thresh, divisor):
  return int(math.ceil(thresh / divisor) * divisor)


# Training settings
parser = argparse.ArgumentParser(description='PyTorch GANet Example')
parser.add_argument('--crop_height', type=int,
                    required=True, help="crop height")
parser.add_argument('--crop_width', type=int, required=True, help="crop width")
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--baseline', type=float,
                    help='Baseline of the stereo camera in meters. Only used for computing depth uncertainty given disparity uncertainty in the ensemble mode.', required=True)
parser.add_argument('--fx', type=float,
                    help='Focal length of the stereo camera. Only used for computing depth uncertainty given disparity uncertainty in the ensemble mode.', required=True)

# parser.add_argument('--model_paths', type=str, default='',
# help="model_paths from saved model")
parser.add_argument(
    "--model_paths",
    nargs="+",
    default=[""],
    required=True,
    help="The path to the model to be loaded or the list of paths to an ensemble of models")
parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=1,
                    help='number of threads for data loader to use')
parser.add_argument('--kitti', type=int, default=0,
                    help='kitti dataset? Default=False')
parser.add_argument('--kitti2015', type=int, default=0,
                    help='kitti 2015? Default=False')
parser.add_argument('--airsim_fmt', type=int, default=0,
                    help='Is it AirSim dataset? Default=False')
parser.add_argument('--data_path', type=str, required=True, help="data root")
parser.add_argument('--test_list', type=str,
                    required=True, help="training list")
parser.add_argument('--save_path', type=str,
                    default='./result/', help="location to save result")
parser.add_argument('--model', type=str,
                    default='GANet_deep', help="model to train")
parser.add_argument('--left_right', type=int, default=0,
                    help="use the right view. Default=False")
parser.add_argument('--batchSize', type=int, default=1,
                    help='batch size')
parser.add_argument('--scale_factor', type=float, default=1.0,
                    help='scaling factor for the images')
parser.add_argument('--subsample_factor', type=float, default=1.0,
                    help='Subsampling factor for the images. Note: It is only applied to the airsim formatted dataset.')
parser.add_argument('--is_mc_dropout',
                    type=lambda s: s.lower() in ['true', '1', 't', 'yes'], default=False,
                    required=False, help='Whether to use Monte Carlo dropout (assuming the model is trained with MC dropout)')
parser.add_argument('--num_mc_dropout_samples', type=int, default=1,
                    required=False,
                    help="number of Monte Carlo samples for dropout if the model is trained with dropout.")
parser.add_argument('--dropout_rate', type=float, default=0.1,
                    help='Dropout rate. Only used if is_mc_dropout is True.')

opt = parser.parse_args()

# Adjust the max disparity value based on the scaling factor
adjusted_max_disp = int(opt.scale_factor * opt.max_disp)
adjusted_max_disp = find_least_multiple_larger_than(adjusted_max_disp, 12)
opt.max_disp = adjusted_max_disp
print("Adjusted max disp: ", opt.max_disp)

# Adjust the height and width cropping values based on the scaling factor
adjusted_crop_height = int(opt.scale_factor * opt.crop_height)
adjusted_crop_width = int(opt.scale_factor * opt.crop_width)
adjusted_crop_height = find_least_multiple_larger_than(
    adjusted_crop_height, 48)
adjusted_crop_width = find_least_multiple_larger_than(adjusted_crop_width, 48)
opt.crop_height = adjusted_crop_height
opt.crop_width = adjusted_crop_width
print("Adjusted crop height: ", opt.crop_height)
print("Adjusted crop width: ", opt.crop_width)

# Adjust based on scaling factor
opt.fx = opt.fx * opt.scale_factor
print("Adjusted fx: ", opt.fx)

print(opt)
if opt.model == 'GANet11':
  from models.GANet11 import GANet
elif opt.model == 'GANet_deep':
  from models.GANet_deep import GANet
  from models.GANet_deep import GANetDropOut
else:
  raise Exception("No suitable model found ...")

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
  raise Exception("No GPU found, please run without --cuda")

print('===> Loading datasets')
return_info = True
test_set = get_test_set(opt.data_path, opt.test_list, [
                        opt.crop_height, opt.crop_width], opt.left_right, opt.kitti, opt.kitti2015, opt.airsim_fmt, opt.scale_factor, opt.subsample_factor, return_info)
testing_data_loader = DataLoader(
    dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)

models = []
if len(opt.model_paths) >= 1:
  for model_path in opt.model_paths:
    print('===> Building model')
    if opt.is_mc_dropout:
      model = GANetDropOut(opt.max_disp, dropout_rate=opt.dropout_rate)
    else:
      model = GANet(opt.max_disp)

    if cuda:
      model = torch.nn.DataParallel(model).cuda()

    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    models.append(model)

else:
  print("No model path was provided. Please provide a model path.")
  exit()


def load_and_scale_img(img_path, scale_factor):
  rgb_img = Image.open(img_path)
  if opt.scale_factor < 1.0:
    rgb_img = rgb_img.resize(
        (int(rgb_img.size[0] * scale_factor), int(rgb_img.size[1] * scale_factor)), Image.ANTIALIAS)
  rgb_img = np.asarray(rgb_img)
  return rgb_img


def save_predicted_depth_uncertainty(predicted_disp_img, predicted_disp_unc_img, data_path, file_names, output_path, original_image_size):
  """
  Converts the predicted disparity uncertainty to depth uncertainty and saves it to a file.
  :param predicted_disp_img: The predicted disparity image (normalized between 0 and 1)
  :param predicted_disp_unc_img: The predicted disparity uncertainty image. (standard deviation of disparity divided by max_disp)
  :param data_path: The list of paths to each input image. This will be used to infer the session ID (output folder name)
  :param file_names: The list of file names for each input image.
  :param output_path: The path to save the depth uncertainty image to.
  :param original_image_size: The list of original image sizes. It is used to crop the padded predicted disparity image.
  """

  # Loop through images in the batch and save them to file
  for i in range(prediction.shape[0]):
    width = original_image_size[1][i]
    height = original_image_size[0][i]

    # Crop the disparity image to the original input image size
    pred_disp = predicted_disp_img[i, :, :]
    unc_img = predicted_disp_unc_img[i, :, :]
    if height <= opt.crop_height and width <= opt.crop_width:
      pred_disp = pred_disp[opt.crop_height - height: opt.crop_height,
                            opt.crop_width - width: opt.crop_width]
      unc_img = unc_img[
          opt.crop_height - height: opt.crop_height,
          opt.crop_width - width: opt.crop_width]

    depth_unc = convert_disparity_unc_to_depth_unc(
        disp_img=256.0 * pred_disp, disp_unc_img=unc_img, baseline=opt.baseline, fx=opt.fx, max_disp=opt.max_disp)

    # Entries of depth_unc must be non-negative
    if np.any(depth_unc < 0):
      print(depth_unc)
      raise ValueError("depth_unc has negative entry/entries")

    # Save images to file
    folder_name = os.path.basename(data_path[i])
    output_dir = os.path.join(
        output_path, folder_name, 'depth_uncertainty_pred')
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    output_depth_unc_file_name = os.path.join(
        output_dir, file_names[i].rstrip('.png') + ".pfm")
    write_pfm(output_depth_unc_file_name, depth_unc)


def save_prediction_images(prediction, target, input, mask, data_path, file_name, output_path, original_image_size, uncertainty_img=None):
  # prediction = prediction.cpu().detach().numpy()
  # target = target.cpu().detach().numpy()
  input = input.cpu().detach().numpy()
  # mask = mask.cpu().detach().numpy()
  prediction[prediction < 0] = 0

  # Loop through images in the batch and save them to file
  for i in range(prediction.shape[0]):
    width = original_image_size[1][i]
    height = original_image_size[0][i]

    # Get the target
    gt_disp = target[i, :, :]
    # Get the mask
    mask_img = mask[i, :, :]

    # Load the input image from file (since the one loaded by data loader is normalized)
    rgb_img_path = data_path[i] + '/img_left/' + file_name[i]
    rgb_img = load_and_scale_img(rgb_img_path, opt.scale_factor)

    # Crop the disparity image to the original input image size
    pred_disp = prediction[i, :, :]
    if height <= opt.crop_height and width <= opt.crop_width:
      pred_disp = pred_disp[opt.crop_height - height: opt.crop_height,
                            opt.crop_width - width: opt.crop_width]
      gt_disp = gt_disp[opt.crop_height - height: opt.crop_height,
                        opt.crop_width - width: opt.crop_width]
      mask_img = mask_img[opt.crop_height - height: opt.crop_height,
                          opt.crop_width - width: opt.crop_width]
      if uncertainty_img is not None:
        unc_img = uncertainty_img[i, :, :]
        unc_img = unc_img[
            opt.crop_height - height: opt.crop_height,
            opt.crop_width - width: opt.crop_width]

    # -------
    # Overlay the error image on the input RGB image
    MAX_ERR_THRESH_VISUALIZATION = 0.1  # def: 0.1
    err_img = np.abs(gt_disp - pred_disp)
    err_img = colorize(
        err_img, plt.get_cmap('viridis'), 0, MAX_ERR_THRESH_VISUALIZATION)
    err_img = np.uint8(err_img * 255)
    input_img_gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    input_img_err_overlaid = cv2.cvtColor(input_img_gray, cv2.COLOR_GRAY2RGBA)

    alpha = 0.5
    assert rgb_img.shape == (
        height, width, 4), "rgb_img.shape: {}".format(rgb_img.shape)

    input_img_err_overlaid = input_img_err_overlaid.astype(np.uint8)
    cv2.addWeighted(err_img, alpha, input_img_err_overlaid, 1 - alpha,
                    0, input_img_err_overlaid)

    # -------
    # Overlay the uncertainty image on the input RGB image
    MAX_UNC_THRESH_VISUALIZATION = 0.1  # def: 0.1
    if uncertainty_img is not None:
      unc_img = colorize(
          unc_img, plt.get_cmap('viridis'), 0, MAX_UNC_THRESH_VISUALIZATION)
      unc_img = np.uint8(unc_img * 255)
      input_img_uncertainty_overlaid = cv2.cvtColor(
          input_img_gray, cv2.COLOR_GRAY2RGBA)

      alpha = 0.5
      input_img_uncertainty_overlaid = input_img_uncertainty_overlaid.astype(
          np.uint8)
      cv2.addWeighted(unc_img,
                      alpha, input_img_uncertainty_overlaid, 1 - alpha,
                      0, input_img_uncertainty_overlaid)

    # TODO: mask out the pixels in input_img_err_overlaid that are not in the mask

    # Save images to file

    # Extract the folder name from the data path
    folder_name = os.path.basename(data_path[i])
    output_dir_disp_pred = os.path.join(output_path, folder_name, 'disp_pred')
    output_dir_err_overlaid = os.path.join(
        output_path, folder_name, 'err_vis')
    output_dir_unc_overlaid = os.path.join(
        output_path, folder_name, 'uncertainty_vis')
    if not os.path.exists(output_dir_disp_pred):
      os.makedirs(output_dir_disp_pred)
    if not os.path.exists(output_dir_err_overlaid):
      os.makedirs(output_dir_err_overlaid)
    if not os.path.exists(output_dir_unc_overlaid) and uncertainty_img is not None:
      os.makedirs(output_dir_unc_overlaid)
    output_path_disp_pred = os.path.join(output_dir_disp_pred, file_name[i])
    output_path_err_overlaid = os.path.join(
        output_dir_err_overlaid, file_name[i])
    output_path_unc_overlaid = os.path.join(
        output_dir_unc_overlaid, file_name[i])

    skimage.io.imsave(output_path_disp_pred,
                      (np.minimum(pred_disp * 256, 255)).astype('uint8'), check_contrast=False)
    skimage.io.imsave(output_path_err_overlaid,
                      input_img_err_overlaid, check_contrast=False)
    if uncertainty_img is not None:
      skimage.io.imsave(output_path_unc_overlaid,
                        input_img_uncertainty_overlaid, check_contrast=False)


if __name__ == "__main__":
  SAVE_PREDICTION_IMAGES = True
  COMPUTE_AND_SAVE_DEPTH_UNCERTAINTY = True
  MEASURE_INFERENCE_TIME = False

  if MEASURE_INFERENCE_TIME:
    starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)

  is_ensemble = False
  if len(models) > 1:
    is_ensemble = True
    print("Running ensemble model")

  for model in models:
    model.eval()

  epoch_error = 0
  valid_iteration = 0

  assert not (
      opt.is_mc_dropout and is_ensemble), "Cannot be both ensemble and mc-dropout"

  assert has_dropout(
      model) if opt.is_mc_dropout else True, "Model must have dropout units if is_mc_dropout is True"
  if opt.is_mc_dropout:
    print("Running in MC-dropout mode with {} samples".format(opt.num_mc_dropout_samples))

  # Samples of inference time of a single model in the ensemble on batch_size number of datapoints
  timings_individual_all = []
  # Inference time of each of the models in the ensemble on the latest data batch
  timings_individual_per_batch = []
  # Samples of the total inference time of all models in the ensemble on batch_size number of datapoints
  timings_ensemble_all = []
  # Iterations to run before starting timing
  warm_up_iterations = 20

  for iteration, batch in enumerate(testing_data_loader):
    timings_individual_per_batch = []
    input1, input2, target = Variable(batch['input1'], requires_grad=False), Variable(
        batch['input2'], requires_grad=False), Variable(batch['target'], requires_grad=False)
    if cuda:
      input1 = input1.cuda()
      input2 = input2.cuda()
      target = target.cuda()

    target = torch.squeeze(target, 1)
    mask = target < opt.max_disp
    mask.detach_()
    valid = target[mask].size()[0]

    if opt.is_mc_dropout:
      mc_dropout_pred_size = tuple(
          [opt.num_mc_dropout_samples] + list(target.size()))
      mc_dropout_prediction = np.zeros(mc_dropout_pred_size, dtype=np.float32)

    ensemble_pred_size = tuple([len(models)] + list(target.size()))
    ensemble_prediction = np.zeros(ensemble_pred_size, dtype=np.float32)
    if valid > 0:
      with torch.no_grad():
        if is_ensemble:
          target = target.cpu().detach().numpy()
          mask = mask.cpu().detach().numpy()
          i = 0
          for model in models:
            if MEASURE_INFERENCE_TIME and iteration > warm_up_iterations:
              starter.record()
            disp2 = model(input1, input2)
            if MEASURE_INFERENCE_TIME and iteration > warm_up_iterations:
              ender.record()
              torch.cuda.synchronize()
              timings_individual_per_batch += [starter.elapsed_time(ender)]
              timings_individual_all += [starter.elapsed_time(ender)]

            disp2 = disp2.cpu().detach().numpy()
            ensemble_prediction[i, :, :, :] = disp2
            i += 1
          prediction = np.mean(ensemble_prediction, axis=0)
          error2 = np.mean(np.abs(prediction[mask] - target[mask]))
          unc_img = np.std(ensemble_prediction, axis=0)

        elif opt.is_mc_dropout:
          set_dropout_to_train(model)
          target = target.cpu().detach().numpy()
          mask = mask.cpu().detach().numpy()
          i = 0
          for i in range(opt.num_mc_dropout_samples):
            if MEASURE_INFERENCE_TIME and iteration > warm_up_iterations:
              starter.record()
            disp2 = model(input1, input2)
            if MEASURE_INFERENCE_TIME and iteration > warm_up_iterations:
              ender.record()
              torch.cuda.synchronize()
              timings_individual_per_batch += [starter.elapsed_time(ender)]
              timings_individual_all += [starter.elapsed_time(ender)]

            disp2 = disp2.cpu().detach().numpy()
            mc_dropout_prediction[i, :, :, :] = disp2
            i += 1
          prediction = np.mean(mc_dropout_prediction, axis=0)
          error2 = np.mean(np.abs(prediction[mask] - target[mask]))
          unc_img = np.std(mc_dropout_prediction, axis=0)

        else:
          prediction = model(input1, input2)
          error2 = torch.mean(torch.abs(prediction[mask] - target[mask]))
          epoch_error += error2.item()
          prediction = prediction.cpu().detach().numpy()
          target = target.cpu().detach().numpy()
          mask = mask.cpu().detach().numpy()
          unc_img = None

        valid_iteration += 1

        print("===> Test({}/{}): Error: ({:.4f})".format(iteration,
              len(testing_data_loader), error2.item()))

    if MEASURE_INFERENCE_TIME and iteration > warm_up_iterations:
      timings_ensemble_all += [sum(timings_individual_per_batch)]
      mean_inf_time = np.mean(np.array(timings_individual_per_batch))
      std_inf_time = np.std(np.array(timings_individual_per_batch))
      print("Mean inference time across the ensemble models on current batch: {:.4f} ms, Std: {:.4f} ms".format(
          mean_inf_time, std_inf_time))

    if SAVE_PREDICTION_IMAGES:
      save_prediction_images(
          prediction, target, input1, mask, batch['data_path'], batch['file_name'], opt.save_path, batch['image_size'], unc_img)
    if COMPUTE_AND_SAVE_DEPTH_UNCERTAINTY and (is_ensemble or opt.is_mc_dropout):
      save_predicted_depth_uncertainty(
          prediction, unc_img, batch['data_path'], batch['file_name'], opt.save_path, batch['image_size'])

  if MEASURE_INFERENCE_TIME:
    mean_inf_time = np.mean(np.array(timings_individual_all))
    std_inf_time = np.std(np.array(timings_individual_all))
    print("Mean inference time for individual models: {:.4f} ms, Std: {:.4f} ms".format(
        mean_inf_time, std_inf_time))

    mean_inf_time = np.mean(np.array(timings_ensemble_all))
    std_inf_time = np.std(np.array(timings_ensemble_all))
    print("Mean inference time for the ensemble: {:.4f} ms, Std: {:.4f} ms".format(
        mean_inf_time, std_inf_time))

  print("===> Test: Avg. Error: ({:.4f})".format(
      epoch_error / valid_iteration))

  msg = "Running predict.py finished. Average error: {:.4f} \n Results are saved to {}".format(
      epoch_error / valid_iteration, opt.save_path)
  send_notification_to_phone(msg, 'Job Finished')
