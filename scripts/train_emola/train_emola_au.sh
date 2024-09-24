#!/bin/bash
export WANDB_PORT=8086
GPU_ID="localhost:4,5,6,7" # 0,1,2,3,4,5,6,7
SEED=1
APPEND_NAME="SEED1"
DATA_PATH='./data/FABAInstruct/train/train_au_anno.json'
PER_DEVICE_TRAIN_BS=16

deepspeed --master_port=29502 --include=$GPU_ID emollava/train/train_mem.py \
    --seed $SEED \
    --model_name_or_path ./checkpoints/llava-v1.5-7b \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --version v1 \
    --data_path $DATA_PATH \
    --image_folder ./data/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_landmark_feature_projector True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./saved_checkpoints/llava-v1.5-7b-lora-emola-$APPEND_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BS \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
