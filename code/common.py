#
# common.py
#
# This file is part of TSCNN.
#
# Copyright (c) 2020 Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2021-01-20     Hengxin.Li   the first version

import os
import torch
import numpy as np
import cv2
from scipy.io import loadmat
from scipy.io import savemat
from torch.nn.functional import interpolate as F
from argument.argument import args
from utils import log, timer, utils


epochs = 0


def get_epoch():
    global epochs
    return epochs


def train(model, data_loader, criterion, optimizer):
    global epochs
    epochs = epochs + 1

    info_log = log.Log(os.path.join('./experiments', args.s_experiment_name), '%(message)s')
    device = torch.device('cpu' if args.b_cpu else 'cuda')

    model.train()

    timer_train = timer.Timer()
    timer_load_data = timer.Timer(True)
    display_losses = torch.zeros(len(criterion.get_loss()), device=device, requires_grad=False)
    load_data_elapsed_ticks = 0
    timer_train_elapsed_ticks = 0

    trained_num = 0
    for iteration_batch, (data, x_dem, target) in enumerate(data_loader):
        if data_loader.dataset.n_random_train - trained_num > 0:
            trained_num = trained_num + data_loader.batch_sampler.batch_size
            data, x_dem, target = data.to(device, non_blocking=True), x_dem.to(device, non_blocking=True), target.to(device, non_blocking=True)
            timer_load_data.stop()
            load_data_elapsed_ticks += timer_load_data.elapsed_ticks()
            timer_train.restart()

            optimizer.zero_grad()
            model_out = model(data, x_dem)
            epoch_loss = criterion(model_out, target)
            epoch_loss.backward()
            if args.gclip > 0:
                torch.nn.utils.clip_grad_value_(model.parameters(), args.gclip)
            optimizer.step()

            timer_train.stop()
            timer_train_elapsed_ticks += timer_train.elapsed_ticks()

            for i, l in enumerate(criterion.get_loss()):
                display_losses[i] += l.get('value')

            if (iteration_batch + 1) % args.n_batches_per_print == 0 or (iteration_batch + 1) * len(data) == len(data_loader.dataset):
                display_loss = ''
                for i, l in enumerate(criterion.get_loss()):
                    display_loss = display_loss + '[{}: {:.4f}]'.format(
                        l.get('type'), display_losses[i] / (iteration_batch + 1))

                info_log.write('[{}/{}]\t{}\t[tr/ld: {:.1f}s/{:.1f}s]'.format(
                    (iteration_batch + 1) * len(data),
                    len(data_loader.dataset),
                    display_loss,
                    timer_train_elapsed_ticks,
                    load_data_elapsed_ticks
                ))
                load_data_elapsed_ticks = 0
                timer_train_elapsed_ticks = 0

            timer_load_data.restart()
        else:
            break
    return display_losses / (len(data_loader.dataset) / args.n_batch_size)


def test(model, data_loader):
    info_log = log.Log(os.path.join('./experiments', args.s_experiment_name), '%(message)s')
    device = torch.device('cpu' if args.b_cpu else 'cuda')

    model.eval()

    with torch.no_grad():
        im_psnr = torch.Tensor().to(device)
        im_ssim = torch.Tensor().to(device)
        timer_test_elapsed_ticks = 0

        timer_test = timer.Timer()
        for d_index, d in enumerate(data_loader):
            t_psnr = torch.Tensor().to(device)
            t_ssim = torch.Tensor().to(device)
            for batch_index, (data, x_dem, target) in enumerate(d):
                data, x_dem, target = data.to(device, non_blocking=True), x_dem.to(device, non_blocking=True), target.to(device, non_blocking=True)
                try:
                    timer_test.restart()
                    model_out = model(data, x_dem)
                    timer_test.stop()

                    timer_test_elapsed_ticks += timer_test.elapsed_ticks()
                    model_rgb = model_out if len(model_out) == 1 else model_out[0]
                    model_rgb = model_rgb.mul(1.0).clamp(0, args.n_rgb_range)

                    all_psnr = utils.psnr(model_rgb, target, args.n_rgb_range).to(device)
                    im_psnr = torch.cat((im_psnr, all_psnr))
                    t_psnr = torch.cat((t_psnr, all_psnr))

                    out_data = model_rgb[0, :].permute(1, 2, 0).cpu().numpy()
                    out_label = target[0, :].permute(1, 2, 0).cpu().numpy()

                    if args.n_rgb_range == 255:
                        out_data = np.uint8(out_data)
                        out_label = np.uint8(out_label)
                    elif args.n_rgb_range == 65535:
                        out_data = np.uint16(out_data)
                        out_label = np.uint16(out_label)

                    all_ssim = utils.ssim(out_data, out_label, args.n_rgb_range).to(device)
                    im_ssim = torch.cat((im_ssim, all_ssim))
                    t_ssim = torch.cat((t_ssim, all_ssim))

                    info_log.write('{}_{}:\t{:.3f}({:.3f}, {:.3f}, {:.3f}), {:.5f}'.format(
                        d.dataset.name,
                        batch_index,
                        all_psnr[:, -1].item(),
                        all_psnr[:, 0].item(),
                        all_psnr[:, 1].item(),
                        all_psnr[:, 2].item(),
                        all_ssim.item(),
                    ))

                    if args.b_save_results:
                        path = os.path.join('./experiments', args.s_experiment_name, d.dataset.name,
                                            'result_' + str(batch_index) + '.bmp')
                        cv2.imwrite(path, cv2.cvtColor(model_out[0, :].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
                except Exception as e:
                    utils.catch_exception(e)

            t_psnr = t_psnr.mean(dim=0, keepdim=True)
            t_ssim = t_ssim.mean(dim=0, keepdim=True)

            info_log.write('{}:\t{:.3f}({:.3f}, {:.3f}, {:.3f}), {:.5f}'.format(
                d.dataset.name,
                t_psnr[:, -1].item(),
                t_psnr[:, 0].item(),
                t_psnr[:, 1].item(),
                t_psnr[:, 2].item(),
                t_ssim.item(),
            ))

        im_psnr = im_psnr.mean(dim=0, keepdim=True)
        im_ssim = im_ssim.mean(dim=0, keepdim=True)

    return im_psnr, im_ssim, timer_test_elapsed_ticks
