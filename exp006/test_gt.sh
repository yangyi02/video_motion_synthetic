#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main_gt.py --test --init_model=./model/final.pth --test_epoch=10 --batch_size=2 --motion_range=3 --image_size=64 --num_inputs=2 --num_channel=3 --display
