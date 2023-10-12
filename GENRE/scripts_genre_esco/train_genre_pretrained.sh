#!/bin/bash
DATASET=datasets/blink_format/
ENCODERS=( "bart_large" )

for MODEL in "${!ENCODERS[@]}"; do
  for SEED in 276800 381552 497646 624189 884832; do
    echo "Training ${ENCODERS[$MODEL]} on seed $SEED"

    fairseq-train $DATASET/bin/ \
        --save-dir runs/$SEED/${ENCODERS[$MODEL]}_genre \
        --arch ${ENCODERS[$MODEL]} \
        --task translation  \
        --criterion label_smoothed_cross_entropy  \
        --source-lang source  \
        --target-lang target  \
        --truncate-source  \
        --label-smoothing 0.1  \
        --max-tokens 256  \
        --update-freq 1  \
        --max-epoch 52  \
        --required-batch-size-multiple 1  \
        --batch-size 32 \
        --dropout 0.1  \
        --attention-dropout 0.1  \
        --relu-dropout 0.0  \
        --weight-decay 0.01  \
        --optimizer adam  \
        --adam-betas "(0.9, 0.999)"  \
        --adam-eps 1e-08  \
        --clip-norm 0.1  \
        --lr-scheduler polynomial_decay  \
        --lr 2e-6 \
        --total-num-update 2000000  \
        --warmup-updates 500  \
        --ddp-backend no_c10d  \
        --num-workers 20  \
        --share-all-embeddings \
        --layernorm-embedding \
        --share-decoder-input-output-embed  \
        --skip-invalid-size-inputs-valid-test  \
        --log-format tqdm  \
        --no-epoch-checkpoints \
        --log-interval 10  \
        --patience 200 \
        --seed $SEED \
        --restore-file models/fairseq_entity_disambiguation_blink/model.pt
        # --reset-meters  \
        # --reset-optimizer \
  done
done