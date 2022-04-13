#
# common.py
#
# This file is part of TSCNN.
#
# Copyright (c) 2020 Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2021-01-20     Hengxin.Li   the first version

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def default_conv(in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, groups=groups,
        padding=(kernel_size//2), bias=bias)

