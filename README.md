

# Delta-LLaVA Hermes-4.3-36B

## Delta-LLaVA: Base-then-Specialize Alignment for Token-Efficient Vision-Language Models [[Paper]](https://openaccess.thecvf.com/content/WACV2026/papers/Zamini_Delta-LLaVA_Base-then-Specialize_Alignment_for_Token-Efficient_Vision-Language_Models_WACV_2026_paper.pdf)

<p align="center">
  <img src="https://github.com/user-attachments/assets/5d217aa5-4e00-4327-9584-4d37018588fe" width="400"/>
</p>

<p align="justify">
Multimodal Large Language Models (MLLMs) combine visual and textual representations to enable rich reasoning capabilities. However, the high computational cost of processing dense visual tokens remains a major bottleneck. A critical component in this pipeline is the visual projector, which bridges the vision encoder and the language model. Standard designs often employ a simple multi-layer perceptron for direct token mapping, but this approach scales poorly with high-resolution inputs, introducing significant redundancy. We present Delta-LLaVA, a token-efficient projector that employs a low-rank DeltaProjection to align multi-level vision features into a compact subspace before further interaction. On top of this base alignment, lightweight Transformer blocks act as specialization layers, capturing both global and local structure under constrained token budgets. Extensive experiments and ablations demonstrate that this base-then-specialize design yields consistent gains across multiple benchmarks with only $144$ tokens, highlighting the importance of token formation prior to scaling interaction capacity. With Delta-LLaVA, inference throughput improves by up to 55%, while end-to-end training accelerates by nearly 4-5x in pretraining and over 1.5x in finetuning, highlighting the dual benefits of our design in both efficiency and scalability.
</p>

This model checkpoint is trained with the [Delta-LLaVA](https://github.com/mzamini92/Delta-LLaVA/)
codebase on top of [NousResearch/Hermes-4.3-36B](https://huggingface.co/NousResearch/Hermes-4.3-36B).
It uses the `hermes_43` chat template and a LLaVA-style multimodal stack.

If you use this repository or model, please cite our Delta-LLaVA paper.

## Download 36 B Parameter model from HuggingFace  ([LINK](https://huggingface.co/mzamini/DeltaLLaVA-36B-144))



## Model Details

- **Base language model:** `NousResearch/Hermes-4.3-36B`
- **Model family:** LLaVA-style multimodal instruction model
- **Conversation template:** `hermes_43`
- **Vision encoder:** CLIP ViT-L/14 336px style vision tower
- **Multimodal projector:** `mlp2x_gelu`
- **Context length used in training:** 2048 tokens
- **Precision during training:** `bf16`

## Training Summary

This checkpoint follows a two-stage LLaVA-style training recipe:

1. **Vision-language projector pretraining**
   - Data: `blip_laion_cc_sbu_558k.json`
   - Trainable module: multimodal projector (`tune_mm_mlp_adapter=True`)
   - Learning rate: `1e-3`
   - Epochs: `1`
   - Scheduler: cosine with `warmup_ratio=0.03`
   - Vision layer selection: `mm_vision_select_layer=-2`

2. **Instruction tuning**
   - Data: `eagle-1-sft-1_8M.json` and/or `llava_v1_5_mix665k.json` depending on the run
   - Initialized from the pretrained projector checkpoint
   - Learning rate: `2e-5`
   - Epochs: `1`
   - Image aspect ratio handling: `pad`
   - Grouped by modality length: `True`

## Intended Use

This model is intended for research and prototyping on image-grounded chat, visual question
answering, image description, and multimodal reasoning.

## Limitations and Safety

- The model may hallucinate visual details or produce incorrect reasoning, especially for
  fine-grained OCR, counting, medical, legal, or safety-critical use cases.
- The model inherits biases and limitations from both the Hermes-4.3-36B base model and the
  multimodal training data.
- For high-stakes settings, verify outputs with domain experts and independent tools.

## How to Use

Clone the project code and point `PYTHONPATH` to the repository before loading the model:

```bash
git clone https://github.com/mzamini92/Delta-LLaVA.git
cd Delta-LLaVA
export PYTHONPATH="$PWD:$PYTHONPATH"
```

Example inference script:

```python
import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

model_path = "YOUR_HF_USERNAME/YOUR_MODEL_REPO"
image_path = "example.jpg"
prompt_text = "Describe this image in detail."

disable_torch_init()
model_name = get_model_name_from_path(model_path)

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name,
    device="cuda",
)

messages = [
    {
        "role": "user",
        "content": f"{DEFAULT_IMAGE_TOKEN}\n{prompt_text}",
    }
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image = Image.open(image_path).convert("RGB")
image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0)

input_ids = tokenizer_image_token(
    prompt,
    tokenizer,
    IMAGE_TOKEN_INDEX,
    return_tensors="pt",
).unsqueeze(0).cuda()

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
        do_sample=False,
        max_new_tokens=256,
        use_cache=True,
    )

response = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
print(response.strip())
```

## Training Configuration Reference

The Hermes-4.3-specific training path is implemented in:

- `llava/train/train.py` (`preprocess_hermes_43`, `version=hermes_43`)
- `llava/conversation.py` (`conv_hermes_43` is used as the template selector; actual Hermes
  prompt rendering is done with `tokenizer.apply_chat_template`)
- `llava/model/language_model/llava_seed_oss.py` (`LlavaSeedOssForCausalLM`)
- `scripts/v1_5/org_llava.sh`
- `scripts/v1_5/multinode_4.sh`

## Citation

If you use this repository or checkpoint, please cite our paper:

```bibtex
@inproceedings{zamini2026delta,
  title={Delta-LLaVA: Base-then-Specialize Alignment for Token-Efficient Vision-Language Models},
  author={Zamini, Mohamad and Shukla, Diksha},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={3648--3657},
  year={2026}
}
```
