#!/bin/sh

python train.py -n bidaf-conv-cnn --eval_after_epoch true --l2_wd 0.0000003 --amp true --use_char_cnn true
