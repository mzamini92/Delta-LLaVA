#!/bin/bash
#SBATCH -N 1                     # Request 2 nodes
#SBATCH --ntasks-per-node=1        # 1 task per node
#SBATCH --cpus-per-task=32         # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:8               # Request 8 GPUs per node (total 16 GPUs across 2 nodes)
#SBATCH --account=reasoning
#SBATCH --partition=mb-h100        # Specify partition
#SBATCH -t 7-00:00:00              # Set the maximum runtime (7 days)
#SBATCH --job-name=144
#SBATCH --mem=512GB                # Memory allocation per node
#SBATCH --output=144.log  # Output file for logs
export PYTHONPATH=${ROOT}:${PATH}

export WANDB_MODE=offline
JSON_FOLDER="./LLM-JSON"
IMAGE_FOLDER="./LLM-IMAGES"
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ${JSON_FOLDER}/blip_laion_cc_sbu_558k.json \
    --image_folder ${IMAGE_FOLDER}/llava_image \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type delatallava \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-pretrain-144 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --scale_factor 2 \
    --report_to wandb
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --image_folder ${IMAGE_FOLDER}/ \
    --data_path ${JSON_FOLDER}/llava_v1_5_mix665k.json \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-pretrain-144/mm_projector.bin \
    --mm_projector_type delatallava \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-144 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --scale_factor 2 \
    --report_to wandb
