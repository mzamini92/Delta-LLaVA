#!/bin/bash
export PYTHONPATH=${ROOT}:${PATH}

CKPT="./checkpoints/llava-v1.5-7b-1"
FILE="llava_v1_5_mix1k.jsonl"

python -m llava.eval.analyze_speed_multi \
    --model-path ${CKPT} \
    --question-file ./LLM-JSON/${FILE} \
    --image-folder ./LLM-IMAGES \
    --temperature 0 \
    --conv-mode vicuna_v1 \
