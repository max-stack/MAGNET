#!/bin/bash

set -x

DATAHOME=$(dirname "${PWD}")/data
EXEHOME=$PWD

SAVEPATH=${DATAHOME}/models/magnet

mkdir -p ${SAVEPATH}

cd ${EXEHOME}

python train.py \
       -save_path ${SAVEPATH} \
       -log_home ${SAVEPATH} \
       -online_process_data \
       -train_src ${DATAHOME}/train/train.equ.txt \
       -train_tgt ${DATAHOME}/train/train.nl.txt \
       -train_lda ${DATAHOME}/train/train.lda.txt \
       -dev_input_src ${DATAHOME}/dev/dev.equ -dev_ref ${DATAHOME}/dev/dev.nl \
       -layers 1 -enc_rnn_size 512 -brnn -word_vec_size 300 -dropout 0.5 \
       -batch_size 64 -beam_size 3 \
       -epochs 20 \
       -optim adam -learning_rate 0.001 \
       -gpus 0 \
       -curriculum 0 -extra_shuffle \
       -start_eval_batch 200 -eval_per_batch 100 \
       -seed 12345 -cuda_seed 12345 \
       -log_interval 100 \
       -eq_lambda 0

