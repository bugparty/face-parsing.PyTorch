#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json
import cv2

from transform import *



class FaceMask(Dataset):
    def __init__(self, rootpth, cropsize=(640, 480), mode='train', *args, **kwargs):
        super(FaceMask, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.ignore_lb = 255
        self.rootpth = rootpth

        # 获取图片列表
        img_dir = os.path.join(self.rootpth, 'CelebA-HQ-img')
        mask_dir = os.path.join(self.rootpth, 'mask')
        print(f"图像目录: {img_dir}")
        print(f"掩码目录: {mask_dir}")
        
        # 仅获取两个目录中都存在的文件
        all_imgs = os.listdir(img_dir)
        valid_imgs = []
        
        for img in all_imgs:
            mask_name = img[:-3] + 'png'
            if os.path.exists(os.path.join(mask_dir, mask_name)):
                valid_imgs.append(img)
        
        self.imgs = valid_imgs
        print(f"总图像数: {len(all_imgs)}, 有效图像数: {len(valid_imgs)}")

        #  pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5),
            HorizontalFlip(),
            RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomCrop(cropsize)
            ])

    def __getitem__(self, idx):
        try:
            impth = self.imgs[idx]
            img_path = osp.join(self.rootpth, 'CelebA-HQ-img', impth)
            mask_path = osp.join(self.rootpth, 'mask', impth[:-3]+'png')
            
            # 检查文件是否存在
            if not osp.exists(img_path):
                print(f"警告: 图像文件不存在: {img_path}, 跳过索引 {idx}")
                # 递归尝试下一个
                return self.__getitem__((idx + 1) % len(self))
                
            if not osp.exists(mask_path):
                print(f"警告: 掩码文件不存在: {mask_path}, 跳过索引 {idx}")
                # 递归尝试下一个
                return self.__getitem__((idx + 1) % len(self))
            
            img = Image.open(img_path)
            img = img.resize((512, 512), Image.BILINEAR)
            label = Image.open(mask_path).convert('P')
            
            if self.mode == 'train':
                im_lb = dict(im=img, lb=label)
                im_lb = self.trans_train(im_lb)
                img, label = im_lb['im'], im_lb['lb']
            img = self.to_tensor(img)
            label = np.array(label).astype(np.int64)[np.newaxis, :]
            return img, label
            
        except Exception as e:
            print(f"加载图片出错 ({idx}): {e}")
            # 递归尝试下一个索引
            return self.__getitem__((idx + 1) % len(self))

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    face_data = '/home/zll/data/CelebAMask-HQ/CelebA-HQ-img'
    face_sep_mask = '/home/zll/data/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
    mask_path = '/home/zll/data/CelebAMask-HQ/mask'
    counter = 0
    total = 0
    for i in range(15):
        # files = os.listdir(osp.join(face_sep_mask, str(i)))

        atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

        for j in range(i*2000, (i+1)*2000):

            mask = np.zeros((512, 512))

            for l, att in enumerate(atts, 1):
                total += 1
                file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
                path = osp.join(face_sep_mask, str(i), file_name)

                if os.path.exists(path):
                    counter += 1
                    sep_mask = np.array(Image.open(path).convert('P'))
                    # print(np.unique(sep_mask))

                    mask[sep_mask == 225] = l
            cv2.imwrite('{}/{}.png'.format(mask_path, j), mask)
            print(j)

    print(counter, total)














