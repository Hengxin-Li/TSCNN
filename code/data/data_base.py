#
# data_base.py
#
# This file is part of TSCNN.
#
# Copyright (c) 2020 Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2021-01-20     Hengxin.Li   the first version

import os
import glob
import pickle
import numpy as np
import torch.utils.data as data
import utils.filename_sort as filename_sort


class DataBase(data.Dataset):
    def __init__(self, args, dataset_name, dataset_range, dataset_ext, b_train):
        super(DataBase, self).__init__()

        self.args = args

        self.name = dataset_name
        self.data_range = dataset_range
        self.ext = dataset_ext
        self.b_train = b_train

        self.bin_label = []
        self.bin_data = []
        self.n_data_paris = 0

        self.file_sort = filename_sort.FileNameSort()

        a_range = [r.split('-') for r in self.data_range.split('/')]
        _range = a_range[0] if self.b_train else a_range[1]
        self.n_data_begin = _range[0]
        self.n_data_end = _range[1]
        self.n_data_paris = int(self.n_data_end) - int(self.n_data_begin) + 1

        data_pack = self.args.data_pack.split('/')
        self.data_pack = data_pack[0] if self.b_train else data_pack[1]

        self.dir_root = os.path.join(self.args.dir_dataset, self.name)

        if self.data_pack == 'packet':
            if self.b_train:
                bin_pack = np.load(os.path.join(self.dir_root, 'bin', 'train', self.name + '_train.npz'))
            else:
                load_file = open(os.path.join(self.dir_root, 'bin', 'test', self.name + '_test.bin'), 'rb')
                bin_pack = pickle.load(load_file)

            self.bin_label = bin_pack['label'][int(self.n_data_begin) - 1:int(self.n_data_end)]
            self.bin_x_dem = bin_pack['x_dem'][int(self.n_data_begin) - 1:int(self.n_data_end)]
            self.bin_data = bin_pack['data'][int(self.n_data_begin) - 1:int(self.n_data_end)]
            # del bin_pack
        elif self.data_pack == 'bin':
            dir_label = os.path.join(self.dir_root, 'bin',
                                     'train' if self.b_train else 'test',
                                     'label')
            dir_data = os.path.join(self.dir_root, 'bin',
                                    'train' if self.b_train else 'test',
                                    'data')

            self.bin_label = self.file_sort.sort((glob.glob(os.path.join(dir_label, '*.npy'))))
            for i in self.bin_label:
                filename, _ = os.path.splitext(os.path.basename(i))
                self.bin_data.append(os.path.join(dir_data, '{}{}'.format(filename, _)))

            self.bin_label = self.bin_label[int(self.n_data_begin) - 1:int(self.n_data_end)]
            self.bin_data = self.bin_data[int(self.n_data_begin) - 1:int(self.n_data_end)]
        else:
            pass

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return self.n_data_paris
