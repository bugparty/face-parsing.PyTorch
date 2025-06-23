#!/usr/bin/python
"""
This script preprocesses face parsing data by combining individual attribute masks into a single mask for each image.

Modules:
- os.path: Provides path manipulation functions.
- os: Provides functions for interacting with the operating system.
- cv2: OpenCV library for image processing.
- transform: Custom module for transformations (not detailed in this script).
- PIL.Image: Provides image processing capabilities.
- multiprocessing: Supports concurrent processing using multiple CPU cores.
- functools: Provides higher-order functions, in this case, `partial` for argument manipulation.

Global Variables:
- face_data: Path to the directory containing face images.
- face_sep_mask: Path to the directory containing separate attribute masks.
- mask_path: Path to the directory where combined masks will be saved.

Functions:
- process_directory(i, face_sep_mask, mask_path): Processes all images in the `i`-th subdirectory of `face_sep_mask`, combining individual attribute masks into a single mask.

Processing Steps:
1. Defines a list of attributes (`atts`) corresponding to facial features.
2. For each image ID in the range of 2000 images per subdirectory:
    - Initializes a blank mask of size 512x512.
    - Iterates through each attribute and checks if the corresponding mask file exists.
    - If the mask file exists, updates the blank mask with the attribute's label value.
    - Saves the combined mask as a PNG file in `mask_path`.
    - Prints the image ID being processed.
3. Uses multiprocessing to process multiple subdirectories concurrently, improving performance on systems with multiple CPU cores.
4. Outputs the count of processed masks (`counter`) and the total masks attempted (`total`) across all subdirectories.

Notes:
- Attribute masks are expected to have pixel values of 225 for the relevant regions.
- Combined masks use integer values (1 to len(atts)) to represent different attributes.
- Ensure the paths (`face_data`, `face_sep_mask`, `mask_path`) are correctly set before running the script.
"""
# -*- encoding: utf-8 -*-

import os.path as osp
import os
import cv2
from transform import *
from PIL import Image
import multiprocessing
from functools import partial

face_data = '/home/bowman/data/CelebAMask-HQ/CelebA-HQ-img'
face_sep_mask = '/home/bowman/data/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
mask_path = '/home/bowman/data/CelebAMask-HQ/mask'

def process_directory(i, face_sep_mask, mask_path):
    local_counter = 0
    local_total = 0
    
    atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
    
    for j in range(i * 2000, (i + 1) * 2000):
        mask = np.zeros((512, 512))
        
        for l, att in enumerate(atts, 1):
            local_total += 1
            file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
            path = osp.join(face_sep_mask, str(i), file_name)
            
            if os.path.exists(path):
                local_counter += 1
                sep_mask = np.array(Image.open(path).convert('P'))
                mask[sep_mask == 225] = l
                
        cv2.imwrite('{}/{}.png'.format(mask_path, j), mask)
        print(f"Processing directory {i}, image {j}")
    
    return local_counter, local_total

if __name__ == "__main__":
    # Create mask_path directory if it doesn't exist
    os.makedirs(mask_path, exist_ok=True)
    
    # Create a pool of workers
    pool = multiprocessing.Pool(processes=min(15, multiprocessing.cpu_count()))
    
    # Map the process_directory function to each directory index
    process_func = partial(process_directory, face_sep_mask=face_sep_mask, mask_path=mask_path)
    results = pool.map(process_func, range(15))
    
    # Close the pool to free resources
    pool.close()
    pool.join()
    
    # Sum up the counters from all processes
    counter = sum(result[0] for result in results)
    total = sum(result[1] for result in results)
    
    print(counter, total)