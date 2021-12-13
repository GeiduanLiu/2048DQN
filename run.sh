#!/bin/bash

CUDA_VISIBLE_DEVICES="$1" python train_AI.py \
  --player bootstrap\
  --model_type CNN \
  --embedding_type emd
