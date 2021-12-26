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
A set of helper functions for converting disparity and depth images to each other
"""

__author__ = "Sadegh Rabiee"
__license__ = "MIT"
__version__ = "1.0.0"

import numpy as np
import re
import sys
# import colour
from matplotlib import colors
from matplotlib import cm


def convert_depth_to_disparity(depth_img, baseline, fx, max_disp):
  """
  Convert depth image to disparity image
  :param depth_img: depth image
  :param baseline: baseline of the stereo camera
  :param fx: focal length of the stereo camera
  :param max_disp: maximum disparity
  :return: disparity image
  """
  disp_img = np.zeros(depth_img.shape, dtype=np.int)

  idx = np.where(depth_img != 0)
  depth_img[idx] = (baseline * fx / depth_img[idx]) / max_disp
  disp_img[idx] = (np.minimum(depth_img[idx], 1) * 256.0).astype(int)
  return disp_img


def convert_disparity_to_depth(disp_img, baseline, fx, max_disp):
  """
  Convert disparity image to depth image
  :param disp_img: disparity image
  :param baseline: baseline of the stereo camera
  :param fx: focal length of the stereo camera
  :param max_disp: maximum disparity
  :return: depth image
  """
  depth_img = np.zeros(disp_img.shape, dtype=np.float32)
  idx = np.where(disp_img != 0)
  depth_img[idx] = baseline * fx / (max_disp * disp_img[idx] / 256.0)
  # for i in range(disp_img.shape[0]):
  #   for j in range(disp_img.shape[1]):
  #     if disp_img[i][j] != 0:
  #       depth_img[i][j] = baseline * fx / (disp_img[i][j] * max_disp / 256.0)
  return depth_img


def read_pfm(file):
  """ Read a pfm file """
  file = open(file, 'rb')

  color = None
  width = None
  height = None
  scale = None
  endian = None

  header = file.readline().rstrip()
  header = str(bytes.decode(header, encoding='utf-8'))
  if header == 'PF':
    color = True
  elif header == 'Pf':
    color = False
  else:
    raise Exception('Not a PFM file.')

  pattern = r'^(\d+)\s(\d+)\s$'
  temp_str = str(bytes.decode(file.readline(), encoding='utf-8'))
  dim_match = re.match(pattern, temp_str)
  if dim_match:
    width, height = map(int, dim_match.groups())
  else:
    temp_str += str(bytes.decode(file.readline(), encoding='utf-8'))
    dim_match = re.match(pattern, temp_str)
    if dim_match:
      width, height = map(int, dim_match.groups())
    else:
      raise Exception('Malformed PFM header: width, height cannot be found')

  scale = float(file.readline().rstrip())
  if scale < 0:  # little-endian
    endian = '<'
    scale = -scale
  else:
    endian = '>'  # big-endian

  data = np.fromfile(file, endian + 'f')
  shape = (height, width, 3) if color else (height, width)

  data = np.reshape(data, shape)
  file.close()

  return data, scale


def write_pfm(file, image, scale=1):
  """ Write a pfm file """
  file = open(file, 'wb')

  color = None

  if image.dtype.name != 'float32':
    raise Exception('Image dtype must be float32.')

  if len(image.shape) == 3 and image.shape[2] == 3:  # color image
    color = True
  # greyscale
  elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:
    color = False
  else:
    raise Exception(
        'Image must have H x W x 3, H x W x 1 or H x W dimensions.')

  file.write(bytes('PF\n', 'UTF-8') if color else bytes('Pf\n', 'UTF-8'))
  temp_str = '%d %d\n' % (image.shape[1], image.shape[0])
  file.write(bytes(temp_str, 'UTF-8'))

  endian = image.dtype.byteorder

  if endian == '<' or endian == '=' and sys.byteorder == 'little':
    scale = -scale

  temp_str = '%f\n' % scale
  file.write(bytes(temp_str, 'UTF-8'))

  image.tofile(file)


# Colorize a grayscale image given a color map and a min and max value
def colorize(img, cmap, vmin=None, vmax=None):
  """
  Colorize a grayscale image given a color map and a min and max value
  :param img: grayscale image
  :param cmap: color map
  :param vmin: minimum value
  :param vmax: maximum value
  :return: color image
  """
  if vmin is None:
    vmin = img.min()
  if vmax is None:
    vmax = img.max()
  norm = colors.Normalize(vmin=vmin, vmax=vmax)
  normed_img = norm(img)
  return cmap(normed_img)
  # return cmap(img)
