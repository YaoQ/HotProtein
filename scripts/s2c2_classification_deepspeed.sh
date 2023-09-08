#!/bin/bash 
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64
export PATH=/usr/local/cuda-11.6/bin:$PATH

ESM_MODEL=esm2_t6_8M_UR50D
#ESM_MODEL=esm1b_t33_650M_UR50S   # esm2_t33_650M_UR50D

if [ -n "$1" ]; then 
    ESM_MODEL=$1 
    echo "train $ESM_MODEL"
fi

if [ "$ESM_MODEL" == "esm1b_t33_650M_UR50S" ]; then
    PRETRAINED_MODEL=sap.pt
else 
    PRETRAINED_MODEL=' '
fi


if [ ! -d "log" ]; then
    mkdir -p "log"
    echo "Directory $dir created."
fi

#--adv \
deepspeed finetune_sup_head_fst_deepspeed.py \
    --deepspeed  \
    --model_location $ESM_MODEL \
    --fasta_file d1/d1_fasta_clean \
    --output_dir S2C2_classification_${ESM_MODEL} \
    --num_classes 2 \
    --include mean per_tok \
    --toks_per_batch 2048 \
    --lr 1e-3 \
    --adv \
    --rank 4 \
    --lr-factor 10 \
    --split_file d1/d1_1_classification.pkl \
    --seed 1 \
    --gamma 1e-6 
    #--load-pretrained "${PRETRAINED_MODEL}" 
