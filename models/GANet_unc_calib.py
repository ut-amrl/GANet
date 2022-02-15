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
This file contains the implementation of the GANet_unc_calib class --- a module for calibrating the predicted uncertainty of an ensemble of GANet models.
"""

import torch
import torch.nn as nn
from torch import Tensor
import math


class MyLoss(nn.Module):
  """
  The code for this class is taken out from pytorch v1.10.0 (because the default environment for this codebase is pytorch v1.4.0 due to dependencies of the GANet model.)
  """
  reduction: str

  def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
    super(MyLoss, self).__init__()
    self.reduction = reduction


class MyGaussianNLLLoss(MyLoss):
  """
  The code for this class is taken out from pytorch v1.10.0 (because the default environment for this codebase is pytorch v1.4.0 due to dependencies of the GANet model.)
  """
  __constants__ = ['full', 'eps', 'reduction']
  full: bool
  eps: float

  def __init__(self, *, full: bool = False, eps: float = 1e-6, reduction: str = 'mean') -> None:
    super(MyGaussianNLLLoss, self).__init__(None, None, reduction)
    self.full = full
    self.eps = eps

  def forward(self, input: Tensor, target: Tensor, var: Tensor) -> Tensor:
    return gaussian_nll_loss(input, target, var, full=self.full, eps=self.eps, reduction=self.reduction)


def gaussian_nll_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    var: torch.Tensor,
    full: bool = False,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
  """
  The code for this function is taken out from pytorch v1.10.0 (because the default environment for this codebase is pytorch v1.4.0 due to dependencies of the GANet model.)
  """

  # Check var size
  # If var.size == input.size, the case is heteroscedastic and no further checks are needed.
  # Otherwise:
  if var.size() != input.size():

    # If var is one dimension short of input, but the sizes match otherwise, then this is a homoscedastic case.
    # e.g. input.size = (10, 2, 3), var.size = (10, 2)
    # -> unsqueeze var so that var.shape = (10, 2, 1)
    # this is done so that broadcasting can happen in the loss calculation
    if input.size()[:-1] == var.size():
      var = torch.unsqueeze(var, -1)

    # This checks if the sizes match up to the final dimension, and the final dimension of var is of size 1.
    # This is also a homoscedastic case.
    # e.g. input.size = (10, 2, 3), var.size = (10, 2, 1)
    elif input.size()[:-1] == var.size()[:-1] and var.size(-1) == 1:  # Heteroscedastic case
      pass

    # If none of the above pass, then the size of var is incorrect.
    else:
      raise ValueError("var is of incorrect size")

  # Check validity of reduction mode
  if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
    raise ValueError(reduction + " is not valid")

  # Entries of var must be non-negative
  if torch.any(var < 0):
    print(var)
    print("var type: ", type(var))
    print("var.dtype: ", var.dtype)
    raise ValueError("var has negative entry/entries")

  # Clamp for stability
  var = var.clone()
  with torch.no_grad():
    var.clamp_(min=eps)

  # Calculate the loss
  loss = 0.5 * (torch.log(var) + (input - target)**2 / var)
  if full:
    loss += 0.5 * math.log(2 * math.pi)

  if reduction == 'mean':
    return loss.mean()
  elif reduction == 'sum':
    return loss.sum()
  else:
    return loss


class GANet_unc_calib_linear(nn.Module):
  """
  This a linear model for scaling the predicted uncertainty of an ensemble of  GANet models. scaled_sigma2 = bias + alpha * sigma2, where alpha and bias are the parameters of the model. bias is a positive constant that models the inherent uncertainty of each of the models in the ensemble. alpha is a positive constant that scales the empirically computer uncertainty of the ensemble.
  """

  def __init__(self):
    """
    Initialize the linear model.
    """
    super(GANet_unc_calib_linear, self).__init__()
    weights = torch.distributions.Uniform(0, 0.1).sample((2,))
    self.weights = nn.Parameter(weights)

  def forward(self, sigma2):
    """
    Forward pass of the linear model.
    :param sigma2: the predicted and unscaled uncertainty of the GANet model.
    :return: the scaled uncertainty of the GANet model.
    """

    bias, alpha = self.weights

    # return torch.nn.functional.relu(10.0 * torch.nn.functional.sigmoid(bias) + alpha * sigma2)
    return torch.nn.functional.relu(bias + alpha * sigma2)


class GANet_unc_calib_bias(nn.Module):
  """
  This a simple linear model for scaling the predicted uncertainty of an ensemble of  GANet models. scaled_sigma2 = bias + sigma2, where bias are the parameters of the model. bias is a positive constant that models the inherent uncertainty of each of the models in the ensemble.
  """

  def __init__(self):
    """
    Initialize the linear model.
    """
    super(GANet_unc_calib_bias, self).__init__()
    weights = torch.distributions.Uniform(0, 0.1).sample((1,))
    self.weights = nn.Parameter(weights)

  def forward(self, sigma2):
    """
    Forward pass of the linear model.
    :param sigma2: the predicted and unscaled uncertainty of the GANet model.
    :return: the scaled uncertainty of the GANet model.
    """

    bias = self.weights

    return torch.nn.functional.relu(bias + sigma2)
    # return torch.nn.functional.relu(10.0 * torch.nn.functional.sigmoid(bias) + sigma2)
