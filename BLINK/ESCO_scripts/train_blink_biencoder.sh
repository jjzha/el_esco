#!/bin/bash

export PYTHONPATH=.

DATA=data/blink_format/
ENCODERS=( "bert-large-uncased" ) # "bert-base-uncased" 
LEARNING_RATE=( 2e-5 ) #2e-6

for MODEL in "${!ENCODERS[@]}"; do
  for SEED in 276800 381552 497646 624189 884832; do
    echo "Training ${ENCODERS[$MODEL]} on seed $SEED"
    python3 blink/biencoder/train_biencoder.py \
      --data_path $DATA \
      --output_path runs/$SEED/${ENCODERS[$MODEL]} \
      --learning_rate 2e-6  \
      --num_train_epochs 3  \
      --max_context_length 128  \
      --max_cand_length 128 \
      --seed $SEED \
      --train_batch_size 32  \
      --eval_batch_size 8 \
      --warmup_proportion 0.05 \
      --shuffle True \
      --eval_interval 200 \
      --bert_model ${ENCODERS[$MODEL]} \
      
  done
done