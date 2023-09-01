#!/bin/bash

ESM_MODEL=esm2_t36_3B_UR50D
#ESM_MODEL=esm1b_t33_650M_UR50S   # esm2_t33_650M_UR50D

if [ -n "$1" ]; then 
    ESM_MODEL=$1 
    echo "train $ESM_MODEL"
fi

if [ "$ESM_MODEL" == "esm1b_t33_650M_UR50S" ]; then
    PRETRAINED_MODEL=sap.pt
else 
    PRETRAINED_MODEL=''
fi

if [ ! -d "log" ]; then
    mkdir -p "log"
    echo "Directory $dir created."
fi

nohup \
python finetune_sup_head_fst.py \
    $ESM_MODEL \
    S/S_target S_classification_${ESM_MODEL} \
    --num_classes 5 \
    --include mean per_tok \
    --toks_per_batch 2048 \
    --save-freq 30000 \
    --lr 1e-3 \
    --rank 16 \
    --lr-factor 10 \
    --split_file S/S_target_classification.pkl \
    --seed 1 \
    --adv \
    --gamma 1e-6 \
    --load-pretrained "${PRETRAINED_MODEL}" \
    > log/S_classification_${ESM_MODEL}.log 2>&1 &