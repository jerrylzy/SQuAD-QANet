#!/bin/bash

python train.py -n ensemble/bidaf/nocnn/hs100 --eval_after_epoch true --hidden_size 100 --seed 0 --amp true #--use_char_cnn true #--l2_wd 0.0000003
