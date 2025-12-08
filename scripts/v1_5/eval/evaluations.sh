#!/bin/bash
#SBATCH -N 1                     # Request 2 nodes
#SBATCH --ntasks-per-node=1        # 1 task per node
#SBATCH --cpus-per-task=32         # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:8               # Request 8 GPUs per node (total 16 GPUs across 2 nodes)
#SBATCH --account=reasoning
#SBATCH --partition=mb-h100        # Specify partition
#SBATCH -t 7-00:00:00              # Set the maximum runtime (7 days)
#SBATCH --job-name=evaluations
#SBATCH --mem=512GB                # Memory allocation per node
#SBATCH --output=evaluations.log  # Output file for logs

#=====================EVAL=====================
# Define available models and tasks
AVAILABLE_MODELS=("7b-144")
AVAILABLE_TASKS=("gqa" "mmbench_en_dev" "mme" "pope_random" "pope_pop" "scienceqa_img" "textvqa_val" "textvqa_test" "vqav2_val" "vizwiz_vqa_val" "nocaps" "flickr30k_test")

export PYTHONPATH=${ROOT}:${PATH}
export HF_HUB_ENABLE_HF_TRANSFER="0"

# Set general configurations
conv_template="vicuna_v1"
GPUS=$(nvidia-smi -L | wc -l) # Count all GPUs
master_port=12345
project_name="delta-llava"

# Iterate over each checkpoint (model)
for MODEL in "${AVAILABLE_MODELS[@]}"; do
    ckpt="./checkpoints/llava-v1.5-${MODEL}"
    run_name="pmod-llava-${MODEL}-eval"

    echo "Evaluating Model: $MODEL"
    echo "Checkpoint: $ckpt"
    echo "Conversation Template: $conv_template"
    echo "GPUs: $GPUS"
    echo "Master Port: $master_port"

    # Iterate over each task
    for TASK in "${AVAILABLE_TASKS[@]}"; do
        echo "Running task: $TASK for model: $MODEL"

        # Check if the accelerate config file exists
        if [ ! -f "$HF_HOME/accelerate/default_config.yaml" ]; then
            echo "Accelerate config file does not exist. Using default settings."

            python3 -m accelerate.commands.launch --num_processes=$GPUS --main_process_port=${master_port} \
                -m lmms_eval \
                --model llava \
                --model_args="pretrained=$ckpt,conv_template=$conv_template" \
                --tasks=$TASK \
                --batch_size 1 \
                --log_samples \
                --log_samples_suffix lmms_eval \
                --output_path="$ckpt/logs/$TASK/" \
                --wandb_args="project=$project_name,job_type=eval,name=${run_name}_${TASK}"
        else
            echo "Using accelerate config: $HF_HOME/accelerate/default_config.yaml"

            python3 -m accelerate.commands.launch --config_file "$HF_HOME/accelerate/default_config.yaml" --num_processes=$GPUS --main_process_port=${master_port} \
                -m lmms_eval \
                --model llava \
                --model_args="pretrained=$ckpt,conv_template=$conv_template" \
                --tasks=$TASK \
                --batch_size 1 \
                --log_samples \
                --log_samples_suffix lmms_eval \
                --output_path="$ckpt/logs/$TASK/" \
                --wandb_args="project=$project_name,job_type=eval,name=${run_name}_${TASK}"
        fi

        echo "Finished task: $TASK for model: $MODEL" | tee -a "$log_file"
        echo "--------------------------------------------" | tee -a "$log_file"
    done

    echo "Completed all tasks for model: $MODEL" | tee -a "$log_file"
    echo "=============================================" | tee -a "$log_file"

    # Move the log file to the checkpoint directory
    mv "$log_file" "$ckpt/eval_summary.log"
done

echo "All evaluations completed!"  
