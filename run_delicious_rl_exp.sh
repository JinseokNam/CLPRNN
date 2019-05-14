#!/bin/bash

SEED=`date +%s`
CUDA_VISIBLE_DEVICES=0 python3 rnn_reinforce.py \
  --use-cuda \
  --dataset delicious \
  --reward-function ndcg5 \
  --alpha 0.9 \
  --value-weight 1.0 \
  --gamma 0.3 \
  --entropy-penalty 0.01 \
  --embedding-size 512 \
  --rnn-hidden-size 2048 \
  --hidden-size 512 \
  --minibatch-size 128 \
  --lr 0.0001 \
  --nonlinear-func relu \
  --weight-decay 0.0 \
  --pretrain-grad-norm 1 \
  --grad-norm 10 \
  --input-dropout-prob 0.2 \
  --dropout-prob 0.5 \
  --pretrain-max-epoch 0 \
  --max-epoch 200 \
  --cv-fold 0 \
  --seed ${SEED}
