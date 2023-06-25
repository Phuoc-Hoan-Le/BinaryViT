
DATA_DIR=/path/to/dataset

torchrun --nproc_per_node=2 main.py \
    --num-workers=16 \
    --batch-size=256 \
    --epochs=300 \
    --model=configs/deit-small-patch16-224 \
    --dropout=0.0 \
    --drop-path=0.1 \
    --opt=adamw \
    --sched=cosine \
    --weight-decay=0.05 \
    --lr=5e-4 \
    --warmup-epochs=5 \
    --color-jitter=0.4 \
    --aa=rand-m9-mstd0.5-inc1 \
    --repeated-aug \
    --reprob=0.25 \
    --mixup=0.8 \
    --cutmix=1.0 \
    --data-path=${DATA_DIR} \
    --output-dir=logs/deit-small-patch16-224 \
    # --resume=logs/deit-small-patch16-224/checkpoint.pth \
    # --current-best-model=logs/deit-small-patch16-224/best.pth \