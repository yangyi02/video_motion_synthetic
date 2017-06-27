#!/bin/bash

MODEL_PATH="./model"
if [[ ! -e $MODEL_PATH ]]; then
    echo "$MODEL_PATH not exist!"
    exit
fi

CUDA_VISIBLE_DEVICES=1 python main.py --test_video --init_model=./model/final.pth --image_size=64 --motion_range=2 --num_inputs=3 --num_channel=3 --test_dir=/home/yi/Downloads/youtube-64 --flow_dir=./flow 2>&1 | tee $MODEL_PATH/test_video.log

python visualize_flow_video.py --motion_range=2 --num_inputs=3 --test_dir=/home/yi/Downloads/youtube-64 --flow_dir=./flow --flow_video_dir=./flow_video

cp test_video.sh $MODEL_PATH
