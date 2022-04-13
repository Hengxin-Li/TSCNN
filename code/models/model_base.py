#
# model_base.py
#
# This file is part of TSCNN.
#
# Copyright (c) 2020 Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2021-01-20     Hengxin.Li   the first version

import torch
import torch.nn as nn


class ModelBase(nn.Module):
    def __init__(self, args):
        super(ModelBase, self).__init__()

        self.args = args

        self.device = torch.device('cpu' if args.b_cpu else 'cuda')
        self.n_gpu = args.n_gpu
        self.b_save_all_models = args.b_save_all_models

    def forward(self, x):
        pass

