#!/bin/bash

export PYTHONPATH=.

# MODEL_TYPE=( "bart_base" "bart_large" "bart_large_genre" ) #bart_large_genre_pretrained
# MODEL_TYPE=( "bart_large_genre_pretrained" )
MODEL_TYPE=( "bart_large_2e6" )

for SEED in 276800 381552 497646 624189 884832; do # 276800 381552 497646 624189 884832
  for MODEL in "${!MODEL_TYPE[@]}"; do

    python3 scripts_genre_esco/evaluate.py \
      --prediction_path datasets/blink_format/test-esco-kilt.jsonl \
      --model_type ${MODEL_TYPE[$MODEL]} \
      --seed $SEED \
      --write_predict \
      # --pretrained \

  done
done
