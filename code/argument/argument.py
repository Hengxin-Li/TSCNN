#
# argument.py
#
# This file is part of TSCNN.
#
# Copyright (c) 2020 Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2021-01-20     Hengxin.Li   the first version

import argparse

parser = argparse.ArgumentParser(description='template of demosaick')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=0,
                    help='number of threads for data loading')
parser.add_argument('--b_cpu', type=bool, default=False,
                    help='use cpu only')
parser.add_argument('--n_gpu', type=int, default=1,
                    help='number of GPU')
parser.add_argument('--b_cudnn', type=bool, default=True,
                    help='use cudnn')
parser.add_argument('--n_seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0',
                    help='CUDA_VISIBLE_DEVICES')

# Model specifications
parser.add_argument('--s_model', '-m', default='TSCNN.RDSRN',
                    help='model name')
parser.add_argument('--b_save_all_models', default=False,
                    help='save all intermediate models')

# Data specifications
parser.add_argument('--dir_dataset', type=str, default='../DATA',
                    help='dataset directory')
parser.add_argument('--n_patch_size', type=int, default=96,
                    choices=[96, 144, 192],
                    help='output patch size')
parser.add_argument('--n_rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--data_pack', type=str, default='packet/packet',  # train/test
                    choices=('packet', 'bin', 'ori'),
                    help='make binary data')

# Optimization specifications
parser.add_argument('--lr', type=float, default=2e-4,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='200,300,400',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--s_loss_model', '-l', default='lossshort.LossShort',
                    help='loss model name')
parser.add_argument('--s_loss', type=str, default='1*MSE',
                    help='loss function configuration')

# Training specifications
parser.add_argument('--s_train_dataset', '-t', default='div2k.DIV2K',
                    help='training data model name')
parser.add_argument('--n_epochs', type=int, default=500,
                    help='number of epochs to train')
parser.add_argument('--n_batch_size', type=int, default=64,
                    help='input batch size for training')

# Evaluation specifications
parser.add_argument('--s_eval_dataset', default='mcm.Mcm+kodak.Kodak',
                    help='evaluation dataset')
parser.add_argument('--b_test_only', type=bool, default=False,
                    help='set this option to test the model')
parser.add_argument('--pre_train', type=str, default='TSCNN_Lx2.pth',
                    help='pre-trained model directory')
parser.add_argument('--n_epochs_per_evaluation', type=int, default=1,
                    help='how many batches to wait before logging training status')
parser.add_argument('--n_epochs_per_save', type=int, default=1,
                    help='how many batches to ')

# Log specifications
parser.add_argument('--s_experiment_name', type=str, default='test',
                    help='file name to save')
parser.add_argument('--b_save_results', type=bool, default=False,
                    help='save output results')
parser.add_argument('--n_batches_per_print', type=int, default=256,
                    help='how many batches to wait before logging training status')

# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='A',
                    help='')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')

args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
