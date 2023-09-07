#!/bin/bash 
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64
export PATH=/usr/local/cuda-11.6/bin:$PATH
ESM_MODEL=esm2_t36_3B_UR50D
#ESM_MODEL=esm1b_t33_650M_UR50S   # esm2_t33_650M_UR50D

if [ -n "$1" ]; then 
    ESM_MODEL=$1 
    echo "Evaluate $ESM_MODEL"
fi

if [ "$ESM_MODEL" == "esm1b_t33_650M_UR50S" ]; then
    PRETRAINED_MODEL=sap.pt
else 
    PRETRAINED_MODEL=''
fi

deepspeed classification_evaluate_deepspeed.py \
    --model_location $ESM_MODEL \
    --num_classes 2 \
    --fasta_file d1/d1_fasta_clean \
    --output_dir S2C2_classification_${ESM_MODEL} \
    --include mean per_tok \
    --toks_per_batch 2048 \
    --lr 1e-3 \
    --rank 4 \
    --lr-factor 10 \
    --split_file d1/d1_1_classification.pkl \
    --seed 1 \
    --adv \
    --gamma 1e-6 