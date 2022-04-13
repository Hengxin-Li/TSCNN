#
# Urban100.py
#
# This file is part of TSCNN.
#
# Copyright (c) 2020 Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2021-01-20     Hengxin.Li   the first version

import torch

import numpy as np

from data import data_base


class Urban100(data_base.DataBase):
    def __init__(self, args, b_train):
        dataset_name = 'Urban100'
        dataset_range = '0-0/1-100'
        dataset_ext = ('png',)

        super(Urban100, self).__init__(args, dataset_name, dataset_range, dataset_ext, b_train)

    def __getitem__(self, index):
        im_data = 0
        im_label = 0
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
