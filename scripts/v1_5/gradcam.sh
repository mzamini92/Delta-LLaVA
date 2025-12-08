#!/bin/bash
python -m llava.eval.analyze_attn_dispersion \
  --model-path ./checkpoints/llava-v1.5-7b-144 \
  --image ./assets/extreme_ironing.jpg \
  --max-new-tokens 128