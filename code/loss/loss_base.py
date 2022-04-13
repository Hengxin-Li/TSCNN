#
# loss_base.py
#
# This file is part of TSCNN.
#
# Copyright (c) 2020 Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2021-01-20     Hengxin.Li   the first version

import torch
import torch.nn as nn


class LossBase(nn.Module):
    def __init__(self, args):
        super(LossBase, self).__init__()

        self.args = args
        self.device = torch.device('cpu' if args.b_cpu else 'cuda')

        self.loss = []

        for loss in self.args.s_loss.strip().split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            else:
                loss_function = None

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function,
                'value': 0.0
            })

        self.loss.append({'type': 'Total', 'weight': 0.0, 'function': None, 'value': 0.0})

    def forward(self, model_out, target):
        losses = torch.zeros(len(self.loss), device=self.device)
        loss_sum = torch.zeros(1, device=self.device)
        for i, l in enumerate(self.loss):
            if l.get('function') is not None:
                loss = l.get('function')(model_out[i], target[i])

                effective_loss = l.get('weight') * loss
                losses[i] = effective_loss
                l['value'] = losses[i].item()
            elif l.get('type') == 'Total':
                loss_sum = losses.sum()
                l['value'] = loss_sum.item()

        return loss_sum
