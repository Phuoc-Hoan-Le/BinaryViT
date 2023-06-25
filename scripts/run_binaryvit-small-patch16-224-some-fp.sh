
DATA_DIR=/path/to/dataset

torchrun --nproc_per_node=8 main.py \
    --num-workers=32 \
    --batch-size=64 \
    --epochs=300 \
    --model=configs/deit-small-patch16-224 \
    --dropout=0.0 \
    --drop-path=0.0 \
    --opt=adamw \
    --sched=cosine \
    --weight-decay=0.00 \
    --lr=5e-4 \
    --warmup-epochs=0 \
    --color-jitter=0.0 \
    --aa=noaug \
    --reprob=0.0 \
    --mixup=0.0 \
    --cutmix=0.0 \
    --data-path=${DATA_DIR} \
    --output-dir=logs/binaryvit-small-patch16-224-some-fp \
    --teacher-model=configs/deit-small-patch16-224 \
    --teacher-model-file=logs/deit-small-patch16-224/best.pth \
    --model-type=extra-res-pyramid \
    --replace-ln-bn \
    --weight-bits=1 \
    --input-bits=1 \
    --att-prob-quantizer-type=bit \
    --avg-res3 \
    --avg-res5 \
    --some-fp \
    # --resume=logs/binaryvit-small-patch16-224-some-fp/checkpoint.pth \
    # --current-best-model=logs/binaryvit-small-patch16-224-some-fp/best.pth \