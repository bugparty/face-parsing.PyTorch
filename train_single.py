#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp
import time
import datetime
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from logger import setup_logger
from model import BiSeNet
from face_dataset import FaceMask
from loss import OhemCELoss
from evaluate import evaluate, process_state_dict
from optimizer import Optimizer

# --- Configuration ---
respth = './res'
os.makedirs(respth, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Training parameters
n_classes = 19
batch_size = 16
num_workers = 8
cropsize = [448, 448]
max_iter = 120000
msg_interval = 50
save_interval = 1000

# Loss parameters
score_thres = 0.7
n_min = batch_size * cropsize[0] * cropsize[1] // 16
ignore_idx = 255

# Optimizer parameters
lr_start = 1e-2
momentum = 0.9
weight_decay = 5e-4
power = 0.9
warmup_steps = 1000
warmup_start_lr = 1e-5

# Data root
data_root = './data/CelebAMask-HQ/'

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train():
    # Logger
    setup_logger(respth)
    torch.backends.cudnn.benchmark = True 
    
    # Get the max_iter variable we'll use throughout the function
    nonlocal_max_iter = max_iter
    
    # Dataset and DataLoader
    ds = FaceMask(data_root, cropsize=cropsize, mode='train')
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True, drop_last=True)

    # Model
    net = BiSeNet(n_classes=n_classes).to(device)
    net.train()
    
    # Resume from latest checkpoint if available
    import glob, re
    start_it = 1
    ckpt_dir = osp.join(respth, 'cp')
    if osp.isdir(ckpt_dir):
        ckpts = glob.glob(osp.join(ckpt_dir, '*_iter.pth'))
        if ckpts:
            # extract iteration numbers and pick latest
            iters = [(int(re.match(r"(\d+)_iter.pth$", os.path.basename(f)).group(1)), f) for f in ckpts if re.match(r"\d+_iter.pth$", os.path.basename(f))]
            last_it, last_ckpt = max(iters, key=lambda x: x[0])
            checkpoint = torch.load(last_ckpt, map_location=device)
            # 兼容新旧两种checkpoint格式
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                # 新格式: 包含模型和优化器状态的字典
                # Process state dict to remove _orig_mod prefix if it exists
                model_state_dict = process_state_dict(checkpoint['model'])
                net.load_state_dict(model_state_dict)
                has_optim_state = 'optim' in checkpoint
                # Update max_iter if we're resuming from a point beyond it
                if 'it' in checkpoint and checkpoint['it'] >= nonlocal_max_iter:
                    # Extend max_iter if we're already beyond it
                    original_max_iter = nonlocal_max_iter
                    nonlocal_max_iter = checkpoint['it'] + 20000  # Add more iterations
                    logger.info(f"Extended maximum iterations from {original_max_iter} to {nonlocal_max_iter}")
            else:
                # 旧格式: 直接是模型状态字典
                model_state_dict = process_state_dict(checkpoint)
                net.load_state_dict(model_state_dict)
                has_optim_state = False
            
            start_it = last_it + 1
            logger.info(f"Resuming from checkpoint {last_ckpt} at iteration {start_it}")

    # Compile model after loading checkpoint to avoid prefix issues
    net = torch.compile(net)

    # Loss functions
    loss_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx).to(device)

    loss2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx).to(device)
    loss3 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx).to(device)

    # Optimizer
    optim = Optimizer(model=net, lr0=lr_start, momentum=momentum,
                      wd=weight_decay, warmup_steps=warmup_steps,
                      warmup_start_lr=warmup_start_lr,
                      max_iter=nonlocal_max_iter, power=power)
    
    # Load optimizer state if resuming
    if 'checkpoint' in locals() and has_optim_state:
        optim.optim.load_state_dict(checkpoint['optim'])
        optim.it = start_it - 1  # 恢复优化器内部的迭代计数器

    # Training loop
    st = time.time()
    glob_st = st
    iters = iter(dl)
    for it in range(1, nonlocal_max_iter + 1):
        it = it if start_it == 1 else it + start_it - 1
        try:
            im, lb = next(iters)
        except StopIteration:
            iters = iter(dl)
            im, lb = next(iters)
        except FileNotFoundError as e:
            logger.warning(f"FileNotFoundError encountered: {e}, skipping this batch")
            continue

        im, lb = im.to(device), lb.to(device)
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        out, out16, out32 = net(im)
        l1 = loss_p(out, lb)
        l2 = loss2(out16, lb)
        l3 = loss3(out32, lb)
        loss = l1 + l2 + l3
        loss.backward()
        optim.step()

        # Logging
        if it % msg_interval == 0:
            elapsed = time.time() - st
            glob_elapsed = time.time() - glob_st
            eta = int((nonlocal_max_iter - it) * (glob_elapsed / it))
            eta_str = str(datetime.timedelta(seconds=eta))
            logger.info(f"it: {it}/{nonlocal_max_iter}, lr: {optim.lr:.6f}, loss: {loss.item():.4f}, eta: {eta_str}, time: {elapsed:.3f}s")
            st = time.time()

        # Save and evaluate
        if it % save_interval == 0:
            # Save checkpoint
            ckpt_path = osp.join(respth, 'cp', f'{it}_iter.pth')
            os.makedirs(osp.dirname(ckpt_path), exist_ok=True)
            torch.save({'model': net.state_dict(), 'optim': optim.optim.state_dict(), 'it': it}, ckpt_path)
            print(f'Saved checkpoint to {ckpt_path}')
            # Evaluate
            evaluate(dspth='~/data/CelebAMask-HQ/test-img', cp=f'{it}_iter.pth')

    # Final save
    final_path = osp.join(respth, 'model_final.pth')
    torch.save({'model': net.state_dict(), 'optim': optim.optim.state_dict(), 'it': nonlocal_max_iter}, final_path)
    logger.info('Training completed, model saved.')


if __name__ == '__main__':
    train()
