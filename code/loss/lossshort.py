#
# lossshort.py
#
# This file is part of TSCNN.
#
# Copyright (c) 2020 Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2021-01-20     Hengxin.Li   the first version

from loss import loss_base


class LossShort(loss_base.LossBase):
    def __init__(self, args):
        super(LossShort, self).__init__(
            args
        )

    def get_loss(self):
        return self.loss

    def forward(self, model_out, target):
        return super(LossShort, self).forward(model_out, target)
