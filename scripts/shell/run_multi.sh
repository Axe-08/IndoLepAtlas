#!/bin/bash
source ~/miniconda3/bin/activate ./conda_env
CUDA_VISIBLE_DEVICES=5 python generate_annotations.py --dataset butterflies
python verify_bboxes_butterflies.py
