#!/bin/bash

MODEL_PATH="./model"
if [[ ! -e $MODEL_PATH ]]; then
    echo "$MODEL_PATH not exist!"
    exit
fi

python main.py --test --init_model=./model/final.pth --test_epoch=1 --motion_range=1 --image_size=64 --num_channel=3 --num_inputs=2 --display

cp test.sh $MODEL_PATH
