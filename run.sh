#!/bin/bash

GpU=$1
CUDA_VISIBLE_DEVICES="$GPU" python train_AI.py \
                        --model_type CNN \
                        --embedding_type emd \
                             
