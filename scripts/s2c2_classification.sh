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
    d1/d1_fasta_clean \
    S2C2_classification_${ESM_MODEL} \
    --num_classes 2 \
    --include mean per_tok \
    --toks_per_batch 2048 \
    --lr 1e-3 \
    --rank 4 \
    --lr-factor 10 \
    --split_file d1/d1_1_classification.pkl \
    --seed 1 \
    --adv \
    --gamma 1e-6 \
    --load-pretrained "${PRETRAINED_MODEL}" \
    >> log/S2C2_classification_${ESM_MODEL}.log 2>&1 &