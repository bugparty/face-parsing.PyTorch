#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet
from face_dataset import FaceMask

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import numpy as np
from tqdm import tqdm
import math
from PIL import Image
import torchvision.transforms as transforms
import cv2

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def process_state_dict(state_dict):
    """Process the state dict to remove '_orig_mod.' prefix if it exists."""
    new_state_dict = {}
    orig_prefix = '_orig_mod.'
    
    # Check if this is a compiled model state dict (has _orig_mod. prefix)
    has_orig_prefix = any(k.startswith(orig_prefix) for k in state_dict.keys())
    
    if has_orig_prefix:
        # Remove the prefix from all keys
        for k, v in state_dict.items():
            if k.startswith(orig_prefix):
                new_key = k[len(orig_prefix):]
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        return new_state_dict
    else:
        # No prefixes to process
        return state_dict

def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):
    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    
    # Load and process state dict
    checkpoint = torch.load(save_pth)
    
    # Check if it's the new format (dict with 'model' key)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        # Process the model state dict to remove _orig_mod prefixes if they exist
        model_state_dict = process_state_dict(checkpoint['model'])
        net.load_state_dict(model_state_dict)
    else:
        # For old format (direct model weights), also check for _orig_mod prefixes
        model_state_dict = process_state_dict(checkpoint)
        net.load_state_dict(model_state_dict)
    
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))







if __name__ == "__main__":
    setup_logger('./res')
    evaluate()
