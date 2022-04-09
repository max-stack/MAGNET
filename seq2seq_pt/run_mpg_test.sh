#!/bin/bash

set -x

DATAHOME=$(dirname "${PWD}")/data
EXEHOME=$PWD

SAVEPATH=${DATAHOME}/models/magnet

mkdir -p ${SAVEPATH}

cd ${EXEHOME}

python translate.py \
       -model ${SAVEPATH}/model_e20.pt \
       -src ${DATAHOME}/test/dev.equ.txt \
       -tgt ${DATAHOME}/test/dev.nl.txt \
       -lda ${DATAHOME}/test/dev.lda.txt \
       -batch_size 1 -beam_size 3 \
       -gpu 0 \
       -output ${DATAHOME}/test/output.txt
