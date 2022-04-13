#
# utils.py
#
# This file is part of TSCNN.
#
# Copyright (c) 2020 Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2021-01-20     Hengxin.Li   the first version

import sys
import torch
import numpy

from skimage.metrics import structural_similarity
from skimage.color import rgb2ycbcr
from importlib import import_module

from argument.argument import args


def RGGB2RGB(img):
    rggb = numpy.zeros([img.shape[0], img.shape[1], 3], dtype=float)
    rggb[:, :, 0] = img[:, :, 0]
    rggb[:, :, 1] = (img[:, :, 1] + img[:, :, 2])/2
    rggb[:, :, 2] = img[:, :, 3]
    return rggb


def psnr(input, target, rgb_range):
    r_input, g_input, b_input = input.split(1, 1)
    if target.shape[1] == 3:
        r_target, g_target, b_target = target.split(1, 1)
    if target.shape[1] == 4:
        r_target, g_target1, g_target2, b_target = target.split(1, 1)
        g_target = (g_target1 + g_target2)/2

    mse_r = (r_input - r_target).pow(2).mean()
    mse_g = (g_input - g_target).pow(2).mean()
    mse_b = (b_input - b_target).pow(2).mean()

    cpsnr = 10 * (rgb_range * rgb_range / ((mse_r + mse_g + mse_b) / 3)).log10()

    psnr = torch.tensor([[10 * (rgb_range * rgb_range / mse_r).log10(),
                         10 * (rgb_range * rgb_range / mse_g).log10(),
                         10 * (rgb_range * rgb_range / mse_b).log10(),
                         cpsnr]]).float()

    return psnr


def ssim(input, target, rgb_range):
    y1 = rgb2ycbcr(input)[:, :, 0]
    if target.shape[2] == 3:
        y2 = rgb2ycbcr(target)[:, :, 0]
    if target.shape[2] == 4:
        y2 = rgb2ycbcr(RGGB2RGB(target))[:, :, 0]

    c_s = structural_similarity(y1, y2, data_range=rgb_range, sigma=1.5, gaussian_weights=True,
                                use_sample_covariance=False)

    return torch.tensor([[c_s]]).float()


def calc_para(net):
    num_params = 0
    f_params = 0
    m_params = 0
    l_params = 0
    stage = 1
    total_str = 'The number of parameters for each sub-block:\n'

    for param in net.parameters():
        num_params += param.numel()

    for body in net.named_children():
        res_params = 0
        res_str = []
        for param in body[1].parameters():
            res_params += param.numel()
        res_str = '[{:s}] parameters: {}\n'.format(body[0], res_params)
        total_str = total_str + res_str
        if stage == 1:
            f_params = f_params + res_params
        elif stage == 2:
            m_params = m_params + res_params
        elif stage == 3:
            l_params = l_params + res_params
        if 'anchor' in body[0]:
            stage += 1

    total_str = total_str + '[total] parameters: {}\n\n'.format(num_params) + \
                '[first_net]\tparameters: {:.3f} M\n'.format(f_params/1e6) + \
                '[middle_net]parameters: {:.3f} M\n'.format(m_params/1e6) + \
                '[last_net]\tparameters: {:.3f} M\n'.format(l_params/1e6) + \
                '[total_net]\tparameters: {:.3f} M\n'.format(num_params/1e6) + \
                '**'
    return total_str


def import_fun(fun_dir, module):
    fun = module.split('.')
    m = import_module(fun_dir + '.' + fun[0])
    return getattr(m, fun[1])


def catch_exception(exception):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print('{}: {}.'.format(exc_type, exception), exc_tb.tb_frame.f_code.co_filename, exc_tb.tb_lineno)


