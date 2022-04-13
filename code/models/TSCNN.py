# This code is inspired by EDVR, RCAN and RDN 
# Residual Dense Network for Image Super-Resolution, add CA attention, add long-term residual with pre-demosaic

# TSCNN.py
#
# This file is part of TSCNN.
#
# Copyright (c) 2021 Hengxin.Li<LiHengxin_gxu@163.com>
#
# Change History:
# 2021-01-20     Hengxin.Li   the first version

from models import common
from models import model_base
import torch
import torch.nn as nn
import math


def pixel_shuffle(scale):
    return nn.PixelShuffle(upscale_factor=scale)


def convBlock_dr(out_channels):
    return nn.Conv2d(in_channels=64,
                     kernel_size=1, out_channels=out_channels,
                     stride=1, padding=0, bias=True)


def MASK(input, pattern):
    num = torch.zeros(len(pattern), dtype=torch.uint8)
    p = [i for (i, val) in enumerate(pattern) if ((val == 'r') + (val == 'R'))]
    num[p] = 0
    p = [i for (i, val) in enumerate(pattern) if ((val == 'g') + (val == 'G'))]
    num[p] = 1
    p = [i for (i, val) in enumerate(pattern) if ((val == 'b') + (val == 'B'))]
    num[p] = 2
    mask = torch.zeros(input.shape, dtype=torch.float32)
    mask[:, 0, 0::2, 0::2] = 1
    mask[:, 1, 0::2, 1::2] = 1
    mask[:, 1, 1::2, 0::2] = 1
    mask[:, 2, 1::2, 1::2] = 1
    return mask


# [ECCV 2018] Image Super-Resolution Using Very Deep Residual Channel Attention Networks
# https://github.com/yulunzhang/RCAN
# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, 4, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(4, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# [ECCV 2018] Image Inpainting for Irregular Holes Using Partial Convolutions
# https://github.com/NVIDIA/partialconv
# weights_init  PartialConv PCBActiv
def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


# ** Sub_network of the first stage **
class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        return output


class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',
                 conv_bias=False):
        super().__init__()
        self.conv = PartialConv(in_ch, out_ch, 5, 1, 2, bias=conv_bias)
        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h


class RES(nn.Module):
    def __init__(self,  conv=common.default_conv, act=nn.ReLU(True)):
        super(RES, self).__init__()
        self.AT = nn.Sequential(*[
            conv(32, 32, kernel_size=3, padding=1, bias=True),
            act,
            conv(32, 32, kernel_size=3, padding=1, bias=True),
        ])

    def forward(self, x):
        out = self.AT(x) + x
        return out


class RES2(nn.Module):
    def __init__(self,  conv=common.default_conv, act=nn.ReLU(True)):
        super(RES2, self).__init__()
        self.AT = nn.Sequential(*[
            conv(32, 32, kernel_size=3, padding=1, bias=True),
            act,
            conv(32, 32, kernel_size=3, padding=1, bias=True),
        ])
        self.AT2 = nn.Sequential(*[
            conv(32, 32, kernel_size=3, padding=1, bias=True),
            act,
            conv(32, 32, kernel_size=3, padding=1, bias=True),
        ])

    def forward(self, x):
        out = self.AT(x) + x
        out = self.AT2(out) + out
        return out


class pre_demosaic(nn.Module):
    def __init__(self, conv=common.default_conv, act=nn.ReLU(True)):
        super(pre_demosaic, self).__init__()
        self.PCR = PCBActiv(1, 32)
        self.PCG = PCBActiv(1, 64)
        self.PCB = PCBActiv(1, 32)
        self.feature_R = nn.Sequential(*[
            conv(32, 32, kernel_size=3, padding=1, bias=True),
        ])
        self.feature_B = nn.Sequential(*[
            conv(32, 32, kernel_size=3, padding=1, bias=True),
        ])
        self.feature_G = nn.Sequential(*[
            conv(64, 64, kernel_size=3, padding=1, bias=True),
        ])
        self.body_R = nn.Sequential(*[
            conv(96, 32, kernel_size=1, padding=0, bias=True),
            act,
            RES(),
            conv(32, 1, kernel_size=1, padding=0, bias=True)
        ])
        self.body_B = nn.Sequential(*[
            conv(96, 32, kernel_size=1, padding=0, bias=True),
            act,
            RES(),
            conv(32, 1, kernel_size=1, padding=0, bias=True)
        ])
        self.body_G = nn.Sequential(*[
            conv(128, 32, kernel_size=1, padding=0, bias=True),
            act,
            RES2()
        ])
        self.bodyR = nn.Sequential(*[
            conv(32, 64, kernel_size=1, padding=0, bias=True)
        ])
        self.bodyB = nn.Sequential(*[
            conv(32, 64, kernel_size=1, padding=0, bias=True)
        ])
        self.finalG = nn.Sequential(*[
            conv(32, 1, kernel_size=1, padding=0, bias=True)
        ])

    def forward(self, x):
        mask_rgb = MASK(x, 'rggb').cuda()
        R_f = self.PCR(x[:, 0:1, :, :], mask_rgb[:, 0:1, :, :])
        G_f = self.PCG(x[:, 1:2, :, :], mask_rgb[:, 1:2, :, :])
        B_f = self.PCB(x[:, 2:, :, :], mask_rgb[:, 2:, :, :])
        R_f = self.feature_R(R_f)
        G_f = self.feature_G(G_f)
        B_f = self.feature_B(B_f)

        G_f = torch.cat([R_f, G_f, B_f], dim=1)
        G_f = self.body_G(G_f)

        R_f = torch.cat([R_f, self.bodyR(G_f)], dim=1)
        B_f = torch.cat([B_f, self.bodyB(G_f)], dim=1)

        R_f = self.body_R(R_f)
        B_f = self.body_B(B_f)

        out = torch.cat([R_f, self.finalG(G_f), B_f], dim=1)
        return out


# ** Sub_network of the second stage **
class AT(nn.Module):
    def __init__(self, inc,  conv=common.default_conv, act=nn.ReLU(True)):
        super(AT, self).__init__()
        self.body1 = nn.Sequential(*[
            conv(inc, inc//2, kernel_size=1, padding=0, bias=True),
            act,
            conv(inc//2, inc//2, kernel_size=3, padding=1, bias=True),
            act,
            conv(inc//2, inc, kernel_size=1, padding=0, bias=True)
        ])
        self.body2 = nn.Sequential(*[
            conv(inc, inc//2, kernel_size=1, padding=0, bias=True),
            act,
            conv(inc//2, inc//2, kernel_size=3, padding=1, bias=True),
            act,
            conv(inc//2, inc, kernel_size=1, padding=0, bias=True)
        ])

    def forward(self, x):
        out1 = self.body1(x)
        out2 = self.body2(x)
        out = out1 + out2
        return out


class AEB(nn.Module):
    def __init__(self,  n_feat=64, reduction=16):
        super(AEB, self).__init__()
        self.AT = nn.Sequential(*[AT(n_feat)])
        self.CA = nn.Sequential(*[
            CALayer(channel=n_feat, reduction=reduction)
        ])

    def forward(self, x):
        out = self.AT(x)
        out = self.CA(out)
        out = out + x
        return out


class AEBs(nn.Module):
    def __init__(self,  n_feat=64, reduction=16):
        super(AEBs, self).__init__()
        modules_body = []
        self.nums = 3
        for i in range(1, self.nums+1):
            modules_body.append(AEB(n_feat, reduction=n_feat//4))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        out = self.body(x)
        return out


class AEBsBlock(nn.Module):
    def __init__(self, inc, outc, conv=common.default_conv, act=nn.ReLU(True)):
        super(AEBsBlock, self).__init__()
        self.body = nn.Sequential(*[
            conv(inc, outc, kernel_size=1, padding=0, bias=True),
            AEBs(outc)
        ])

    def forward(self, x):
        out = self.body(x)
        return out


class hDenseAEBs(nn.Module):     # long skip concat
    def __init__(self, args, conv=common.default_conv, act=nn.ReLU(True)):
        super(hDenseAEBs, self).__init__()
        nf = args.hrc
        self.nums = args.h_blocks

        for i in range(1, self.nums + 1):
            if i == 1:
                self.__setattr__('AEBs_{}'.format(i), AEBs(nf))
            else:
                self.__setattr__('AEBs_{}'.format(i), AEBsBlock(inc=i*nf, outc=nf))
        self.conv = conv(nf * self.nums + nf, nf, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        tem = x
        # res = x
        for i in range(1, self.nums + 1):
            res = self.__getattr__('AEBs_{}'.format(i))(tem)
            tem = torch.cat([tem, res], dim=1)
        out = self.conv(tem) + x
        return out


class lDenseAEBs(nn.Module):     # long skip concat
    def __init__(self, args, conv=common.default_conv, act=nn.ReLU(True)):
        super(lDenseAEBs, self).__init__()
        nf = args.lrc
        self.nums = args.l_blocks

        for i in range(1, self.nums + 1):
            if i == 1:
                self.__setattr__('AEBs_{}'.format(i), AEBs(nf))
            else:
                self.__setattr__('AEBs_{}'.format(i), AEBsBlock(inc=i * nf, outc=nf))
        self.conv = conv(nf * self.nums + nf, nf, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        tem = x
        for i in range(1, self.nums + 1):
            res = self.__getattr__('AEBs_{}'.format(i))(tem)
            tem = torch.cat([tem, res], dim=1)
        out = self.conv(tem) + x
        return out


class OctRDB(nn.Module):
    def __init__(self, args, conv=common.default_conv, act=nn.ReLU(True)):
        super(OctRDB, self).__init__()
        self.conv1 = conv(64, args.lrc, kernel_size=3, padding=1, stride=2)
        self.conv2 = conv(64, args.hrc, kernel_size=3, padding=1, stride=1)
        for i in range(1, 3):
            self.__setattr__('hDenseAEBs{}'.format(i), hDenseAEBs(args))
        for i in range(1, 3):
            self.__setattr__('lDenseAEBs{}'.format(i), lDenseAEBs(args))
        self.dr = convBlock_dr(args.hrc)
        self.pre_up1 = conv(args.lrc, 4*args.lrc, kernel_size=1)
        self.pre_up2 = conv(args.lrc, 4*args.lrc, kernel_size=1)
        self.upsample = pixel_shuffle(2)
        self.conv3 = conv(64, 64, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x_l = self.conv1(x)
        x_h = self.conv2(x)
        x_l = self.lDenseAEBs1(x_l)
        x_h = self.hDenseAEBs1(x_h)
        x_h = self.dr(torch.cat([self.upsample(self.pre_up1(x_l)), x_h], dim=1))
        x_l = self.lDenseAEBs2(x_l)
        x_h = self.hDenseAEBs2(x_h)
        out = torch.cat([self.upsample(self.pre_up2(x_l)), x_h], 1)
        out = self.conv3(out)
        return out


class RDSRN(model_base.ModelBase):
    def __init__(self, args):
        super(RDSRN, self).__init__(args)
        r = self.args.n_patch_size // 48
        G0 = args.G0
        kSize = args.RDNkSize

        # number of RDB blocks, conv layers, out channels
        self.D, L, H, lc, hc = {
            'A': (3, 2, 3, 16, 48),     # TSCNN-L   1.428 M
            'B': (4, 4, 6, 16, 48),     # TSCNN-H   3.392 M
        }[args.RDNconfig]
        args.hrc = hc       # hr channels
        args.lrc = lc       # lr channels
        args.h_blocks = H   # blocks in hr's dense
        args.l_blocks = L   # blocks in lr's dense

        # RGB initial fusion
        self.pre_demosaic = pre_demosaic()

        # the second stage
        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        self.OctRDBs = nn.ModuleList()
        for i in range(self.D):
            self.OctRDBs.append(
                OctRDB(args)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        # Up-sampling net
        if r == 2 or r == 3:
            self.output = nn.Sequential(*[
                nn.Conv2d(64, 3 * r * r, kernel_size=3, padding=1, bias=True),
                pixel_shuffle(r)
            ])
        elif r == 4 or 8:
            self.output = nn.Sequential(*[
                nn.Conv2d(64, 3 * r * r, kernel_size=3, padding=1, bias=True),
                pixel_shuffle(r),
            ])

    def forward(self, x, x_dem):  # x is mosaiced patch, x_dem is pre-demosaiced patch get from MLRI
        # the first stage
        x_input = self.pre_demosaic(x) + x_dem

        # the second stage
        f__1 = self.SFENet1(x_input)
        x_input = self.SFENet2(f__1)
        RDBs_out = []
        for i in range(self.D):
            out = self.OctRDBs[i](x_input)
            RDBs_out.append(out)

        out = self.GFF(torch.cat(RDBs_out, 1))
        out += x_input

        return self.output(out)
