#!/bin/bash
export PATH="/opt/nvidia/nsight-compute/2025.2.1:$PATH"
export CUDA_VISIBLE_DEVICES=0

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 ncu --launch-count 1 python  train.py