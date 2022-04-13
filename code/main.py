#
# main.py
#
# This file is part of TSCNN.
#
# Copyright (c) 2020 Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2021-01-20     Hengxin.Li   the first version

import os
import time
import random
import numpy as np

import torch
import torch.nn.parallel.data_parallel
import torch.backends.cudnn
import torch.optim.lr_scheduler
import torch.distributed
import torch.multiprocessing
from torch.utils.data import DataLoader
from argument import argument
import common
from argument.argument import args
from utils import log, timer, check_point, utils

os.environ['CUDA_VISIBLE_DEVICES'] = argument.args.CUDA_VISIBLE_DEVICES


def main():
    torch.cuda.empty_cache()
    random.seed(args.n_seed)
    np.random.seed(args.n_seed)
    torch.manual_seed(args.n_seed)
    torch.backends.cudnn.enabled = args.b_cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    device = torch.device('cpu' if args.b_cpu else 'cuda')

    info_log = log.Log(os.path.join('./experiments', args.s_experiment_name), '%(message)s')
    info_log.write('Experiment: {} ({})'.format(
        args.s_experiment_name,
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    ))

    model = utils.import_fun('models', args.s_model.strip())(args)
    model = torch.nn.DataParallel(model, device_ids=list(range(args.n_gpu))) if args.n_gpu > 1 and not args.b_cpu else model
    model = model.to(device)

    total_params = utils.calc_para(model)
    info_log.write('Total number of parameters: {}'.format(total_params))

    loss = utils.import_fun('loss', args.s_loss_model.strip())(args).to(device)
    ckp = check_point.CheckPoint('./experiments', args.s_experiment_name)

    global data_train
    data_train = utils.import_fun('data', args.s_train_dataset.strip())(args, b_train=True)
    train_sampler = None
    data_loader_train = DataLoader(
        dataset=data_train,
        num_workers=args.n_threads,
        batch_size=args.n_batch_size,
        shuffle=False,
        pin_memory=not args.b_cpu,
        sampler=train_sampler)

    data_loader_test = [(DataLoader(
                            dataset=utils.import_fun('data', dataset)(args, b_train=False),
                            num_workers=0,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=not args.b_cpu
                        )) for dataset in args.s_eval_dataset.strip().split('+')]

    ckp.save_config(args, model)

    optimizer = None
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay,
                                    momentum=args.momentum)
    elif args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     betas=args.betas,
                                     eps=args.epsilon,
                                     weight_decay=args.weight_decay)

    milestones = [int(i) for i in args.decay.strip().split(',')]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)

    load_epoch = 0
    timer_epoch = timer.Timer()

    if args.b_test_only:
        pth = ckp.load(device)
        model.load_state_dict(pth.get('model'))
        info_log.write('Resume model for testing')
        for ds in data_loader_test:
            info_log.write('Testing database: {}({})'.format(ds.dataset.name, len(ds)))
        psnr, ssim, test_time = common.test(model, data_loader_test)

        info_log.write('[Testing: {:.3f}({:.3f}, {:.3f}, {:.3f}), {:.5f}]'.format(
            psnr[:, -1].item(),
            psnr[:, 0].item(), psnr[:, 1].item(), psnr[:, 2].item(),
            ssim[-1].item(),
        ))
        info_log.write('Testing elapsed: {:.3f}s'.format(test_time))
    else:
        str_loss = 'Loss function: '
        for l in loss.get_loss():
            if l.get('function') is not None:
                str_loss = str_loss + '[{:.3f} * {}]'.format(l.get('weight'), l.get('type'))
        info_log.write(str_loss)

        info_log.write('Training database: {}({})'.format(data_train.name, len(data_train)))
        for ds in data_loader_test:
            info_log.write('Testing database: {}({})'.format(ds.dataset.name, len(ds)))

        best_cpsnr = {'psnr': torch.zeros(3, device=device, requires_grad=False), 'cpsnr': torch.zeros(1, device=device, requires_grad=False), 'epoch': 0}
        for epoch in range(load_epoch, args.n_epochs):
            timer_epoch.start()

            info_log.write('\n[Epoch: {}/{}] [Lr: {:.2e}] ({})'.format(
                epoch + 1,
                args.n_epochs,
                scheduler.get_last_lr()[0],
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            ))

            losses = common.train(model, data_loader_train, loss, optimizer)
            scheduler.step(None)

            test_time = 0
            if (epoch + 1) % args.n_epochs_per_evaluation == 0:
                psnr, ssim, test_time = common.test(model, data_loader_test)
                if psnr[:, -1] > best_cpsnr.get('cpsnr'):
                    best_cpsnr['cpsnr'] = psnr[0, -1]
                    best_cpsnr['psnr'] = psnr[0, :3]
                    best_cpsnr['ssim'] = ssim[-1]
                    best_cpsnr['epoch'] = epoch + 1

                info_log.write('[Epoch: {}/{}, {:.3f}({:.3f}, {:.3f}, {:.3f}), {:.5f}]'.format(
                    epoch + 1, args.n_epochs,
                    psnr[:, -1].item(),
                    psnr[:, 0].item(), psnr[:, 1].item(), psnr[:, 2].item(),
                    ssim[-1].item(),
                ))
                info_log.write('[Best: {:.3f}({:.3f}, {:.3f}, {:.3f}), {:.5f} @epoch {}]'.format(
                    best_cpsnr.get('cpsnr').item(),
                    best_cpsnr.get('psnr')[0].item(), best_cpsnr.get('psnr')[1].item(), best_cpsnr.get('psnr')[2].item(),
                    best_cpsnr.get('ssim').item(),
                    best_cpsnr.get('epoch')
                ))

                if (epoch + 1) % args.n_epochs_per_save == 0:
                    ckp.save(model.module if args.n_gpu > 1 else model, loss, losses, optimizer, scheduler, epoch + 1,
                             is_best=(best_cpsnr.get('epoch') == epoch + 1), result=psnr.cpu())

            timer_epoch.stop()
            info_log.write('[Epoch elapsed: {:.1f}s/{:.1f}s]'.format(test_time, timer_epoch.elapsed_ticks()))

    info_log.write('Completed !!!')


if __name__ == '__main__':
    main()
