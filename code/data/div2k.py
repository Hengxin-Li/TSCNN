#
# rdsrn.py
#
# This file is part of TSCNN.
#
# Copyright (c) 2020 Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2021-01-20     Hengxin.Li   the first version

import os
import glob
import torch
import numpy as np
import cv2
import pickle

import common
from data import data_base


class DIV2K(data_base.DataBase):
    def __init__(self, args, b_train):
        dataset_name = 'div2k'
        if args.n_patch_size == 96:
            dataset_range = '1-236000/1-100'
        if args.n_patch_size == 144:
            dataset_range = '1-114000/1-100'
        if args.n_patch_size == 192:
            dataset_range = '1-62000/1-100'
        if args.n_patch_size == 192 and 'ps' in args.s_train_dataset:
            dataset_range = '1-72000/1-100'

        self.n_random_train = 57000  # number of random training data blocks
        dataset_ext = ('bmp',)

        super(DIV2K, self).__init__(args, dataset_name, dataset_range, dataset_ext, b_train)

    def __getitem__(self, index):
        # global batch_index
        res_epoch = common.get_epoch()
        index = (index * 349 + (res_epoch - 1) * 307) % 62000
        im_data = im_x_dem = im_label = 0
        if self.data_pack == 'packet':
            im_data = self.bin_data[index]
            im_x_dem = self.bin_x_dem[index]
            im_label = self.bin_label[index]
        elif self.data_pack == 'bin':
            im_data = np.load(self.bin_data[index])
            im_x_dem = self.bin_x_dem[index]
            im_label = np.load(self.bin_label[index])
        else:
            pass
        return torch.from_numpy(im_data).float(), torch.from_numpy(im_x_dem).float(), torch.from_numpy(im_label).float()

