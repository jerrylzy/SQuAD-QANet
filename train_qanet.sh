#!/bin/bash

python train.py -n qanet-base --lr 0.001 --drop_prob 0.1 --qanet true --project true --use_char_cnn true --hidden_size 128 --batch_size 16 --l2_wd 0.0000003 --eval_after_epoch true --amp true
