#!/bin/bash

python train.py -n qanet-cnn0311202210/bidaf/nocnn/no-kaiming-init/hs128 --eval_after_epoch true --hidden_size 128 --seed 0 --amp true #--use_char_cnn true --l2_wd 0.0000003
