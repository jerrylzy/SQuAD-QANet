#!/bin/sh

python train.py -n bidaf-conv-proj --eval_after_epoch true --use_char_cnn true --l2_wd 0.0000003
