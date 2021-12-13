#!/bin/bash

CUDA_VISIBLE_DEVICES="$1" python train_AI.py \
                        --model_type CNN \
                        --embedding_type emd \
                             
