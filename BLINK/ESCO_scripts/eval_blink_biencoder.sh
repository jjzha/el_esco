#!/bin/bash

export PYTHONPATH=.

DATA=data/blink_format/
MODEL_TYPE=( "bert-base-uncased" "bert-large-uncased" "bert-large-uncased" )
ENCODER=( "bert-base-uncased" "bert-large-uncased" "bert-large-uncased-blink" )
CKPT=( "epoch_*/pytorch_model.bin" "epoch_*/pytorch_model.bin" "epoch_*/pytorch_model.bin" )

for SEED in 276800 381552 497646 624189 884832; do
  for MODEL in "${!MODEL_TYPE[@]}"; do

    python3 blink/biencoder/eval_biencoder.py \
      --mode test \
      --data_path $DATA \
      --output_path results/$SEED/${ENCODER[$MODEL]} \
      --entity_dict_path data/documents/documents.jsonl \
      --cand_pool_path results/cand_${ENCODER[$MODEL]}.cached \
      --cand_encode_path results/candenc_${ENCODER[$MODEL]}.cached \
      --max_context_length 128  \
      --max_cand_length 128 \
      --top_k 32 \
      --save_topk_result \
      --seed $SEED \
      --bert_model ${MODEL_TYPE[$MODEL]} \
      --path_to_model runs/$SEED/${ENCODER[$MODEL]}/${CKPT[$MODEL]} \

  done
done