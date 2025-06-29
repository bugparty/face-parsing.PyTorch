#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet
from face_dataset import FaceMask
from loss import OhemCELoss
from evaluate import evaluate
from optimizer import Optimizer


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import datetime
import argparse


respth = './res'
os.makedirs(respth, exist_ok=True)
logger = logging.getLogger()


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
            '--local_rank', '--local-rank',
            dest='local_rank',
            type=int,
            default=os.environ.get('LOCAL_RANK', -1),
            )
    return parse.parse_args()


def train():
    args = parse_args()
    
     # 如果 local_rank == -1，说明是单卡模式；否则为分布式多卡模式、
    print('local_rank:', args.local_rank)
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend='nccl'
    )
    device = torch.device('cuda', args.local_rank)

    setup_logger(respth)

    # dataset
    n_classes = 19
    n_img_per_gpu = 16
    n_workers = 8
    cropsize = [448, 448]
    data_root = './data/CelebAMask-HQ/'

    ds = FaceMask(data_root, cropsize=cropsize, mode='train')

    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds,
                    batch_size=n_img_per_gpu,
                    shuffle=False,
                    sampler=sampler,
                    num_workers=n_workers,
                    pin_memory=True,
                    drop_last=True)

    # model
    net = BiSeNet(n_classes=n_classes).to(device)
    net.train()
    net = nn.parallel.DistributedDataParallel(
        net, device_ids=[args.local_rank], output_device=args.local_rank
    )
    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1]//16
    # define ignore label index for loss functions
    ignore_idx = 255

    LossP = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss3 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    ## optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    max_iter = 80000
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5
    optim = Optimizer(
            model = net.module if hasattr(net, 'module') else net,
            lr0 = lr_start,
            momentum = momentum,
            wd = weight_decay,
            warmup_steps = warmup_steps,
            warmup_start_lr = warmup_start_lr,
            max_iter = max_iter,
            power = power)

    ## train loop
    msg_iter = 50
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    for it in range(max_iter):
        try:
            im, lb = next(diter)
            if not im.size()[0] == n_img_per_gpu:
                raise StopIteration
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            diter = iter(dl)
            im, lb = next(diter)
        im, lb = im.to(device), lb.to(device)
        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        out, out16, out32 = net(im)
        lossp = LossP(out, lb)
        loss2 = Loss2(out16, lb)
        loss3 = Loss3(out32, lb)
        loss = lossp + loss2 + loss3
        loss.backward()
        optim.step()

        loss_avg.append(loss.item())

        #  print training log message
        if (it+1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join([
                    'it: {it}/{max_it}',
                    'lr: {lr:4f}',
                    'loss: {loss:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]).format(
                    it = it+1,
                    max_it = max_iter,
                    lr = lr,
                    loss = loss_avg,
                    time = t_intv,
                    eta = eta
                )
            logger.info(msg)
            loss_avg = []
            st = ed
        # 只有 rank 0（或单卡）才保存和评估
        if dist.get_rank() == 0:
            if (it+1) % 5000 == 0:
                state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                if dist.get_rank() == 0:
                    torch.save(state, './res/cp/{}_iter.pth'.format(it))
                evaluate(dspth=data_root+'/test-img', cp='{}_iter.pth'.format(it))

    # 结束后单卡或 rank 0 保存最终模型
    if  dist.get_rank() == 0:
        torch.save(net.state_dict(), osp.join(respth, 'model_final.pth'))
        logger.info('training done, model saved.')


if __name__ == "__main__":
    train()
