import argparse
import torch
import os

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from llava.model import *
import torch.nn.functional as F
from functools import partial
from torchvision.transforms import Compose, ToTensor, Normalize

from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer
import time

# ---------------------------
# Helpers for metrics
# ---------------------------
def get_visual_token_count(model, image_tensor):
    """
    Try to measure the number of visual tokens *after* the projector.
    Falls back to vision features length or to config-based estimate.
    """
    try:
        vt = model.get_vision_tower()
        feats = vt(image_tensor.to(device='cuda', dtype=torch.float16))  # [B, N, C] typically
        # Project through mm_projector if present
        proj = None
        if hasattr(model, 'mm_projector') and model.mm_projector is not None:
            proj = model.mm_projector(feats)  # [B, Np, D]
        elif hasattr(model, 'model') and hasattr(model.model, 'mm_projector') and model.model.mm_projector is not None:
            proj = model.model.mm_projector(feats)
        elif hasattr(model, 'projector') and model.projector is not None:
            proj = model.projector(feats)
        out = proj if proj is not None else feats
        if isinstance(out, (tuple, list)):
            # Some projectors return (tokens, aux)
            out = out[0]
        vis_tokens = out.shape[1]
        return int(vis_tokens)
    except Exception:
        # Fallbacks
        try:
            if 'vision_tower' in model.config.to_dict():
                img_size = getattr(getattr(vt, 'config', vt), 'image_size', 336)
                patch = getattr(getattr(vt, 'config', vt), 'patch_size', 14)
                return int((img_size // patch) ** 2)
        except Exception:
            pass
        # Last resort
        return -1  # unknown

def estimate_decoder_flops(
    d_model: int,
    d_ff: int,
    n_layers: int,
    prompt_len: int,
    new_tokens: int
) -> tuple[float, float, float]:
    """
    Returns (prefill_flops, decode_flops, total_flops) in FLOPs.
    Multiply-add counted as 2 FLOPs. Matches the earlier script.
    """
    d = int(d_model)
    L0 = int(prompt_len)
    T  = int(new_tokens)

    # Per-layer prefill (no KV cache):
    # 8*L0*d^2 (Q,K,V,Out) + 4*L0*d*d_ff (MLP) + 4*L0^2*d (attention scores)
    prefill_per_layer = 8*L0*(d**2) + 4*L0*d*d_ff + 4*(L0**2)*d
    prefill = n_layers * prefill_per_layer

    # Decode with KV cache; sum over t=1..T of attention length L0+(t-1)
    # Projections/MLP per token: 8*d^2 + 4*d*d_ff
    decode_proj_mlp_sum = T * (8*(d**2) + 4*d*d_ff)
    decode_attn_sum = 4*d * (T*L0 + T*(T-1)//2)
    decode_per_layer = decode_proj_mlp_sum + decode_attn_sum
    decode = n_layers * decode_per_layer

    total = prefill + decode
    return float(prefill), float(decode), float(total)


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image
# ---------------------------
# NEW: human-friendly explainer for params vs FLOPs
# ---------------------------
def print_flops_explanation(
    model_name: str,
    n_layers: int,
    d_model: int,
    d_ff: int,
    prompt_len: int,
    new_tokens: int,
    prefill_flops: float,
    decode_flops: float,
    total_flops: float,
):
    """
    Explain why FLOPs can be ~1e11 even for a '7B' model.

    Args:
        *_flops are in raw FLOPs (not Tera), as returned by estimate_decoder_flops.
    """
    B = 1e9
    TERA = 1e12

    total_B = total_flops / B
    prefill_B = prefill_flops / B
    decode_B = decode_flops / B

    print("\n[Why do FLOPs look so big for a '7B' model?]")
    print(
        "• '7B' refers to the number of parameters (weights). FLOPs measure the math per inference.\n"
        "• Per-token compute in a Transformer scales with hidden size and number of layers; "
        "attention also scales with sequence length.\n"
        "• With many visual tokens (e.g., 576) + text tokens, the effective sequence length (L_eff) is large, "
        "so prefill (no KV cache) can dominate."
    )
    print(f"• Current run: model={model_name}, LAYERS={n_layers}, d={d_model}, d_ff={d_ff}, "
          f"L_eff={prompt_len}, new_tokens={new_tokens}")
    print(f"• Decoder FLOPs breakdown (this run): prefill ≈ {prefill_B:,.1f} B, "
          f"decode ≈ {decode_B:,.1f} B, total ≈ {total_B:,.1f} B")

    # Show the actual formula used so users can map the numbers:
    print("\n[Formula used (per layer)]")
    print("prefill_per_layer  = 8*L*d^2 + 4*L*d*d_ff + 4*L^2*d")
    print("decode_per_layer   = T*(8*d^2 + 4*d*d_ff) + 4*d*Σ_{t=1..T}(L + t-1)")
    print("Then multiply by number of layers.\n")

    # Quick intuition bullets:
    print("Intuition:")
    print("  1) Prefill has an L^2*d term from attention; large L (many vision tokens) makes it big.")
    print("  2) Decode uses KV cache; attention grows ~linearly with L and T (much cheaper than prefill).")
    print("  3) Hence trimming visual tokens (FastV) slashes prefill FLOPs, often by ~40–80%.\n")
def fastv_reduction_ratio(n: int, R: float, K: int, T: int, d: int, m: int) -> float:
    """
    FastV Eq.(5): FLOPs reduction ratio when pruning image tokens.

    Args:
        n: initial token count (e.g., 576 for CLIP-336)
        R: pruning ratio (0.0–1.0), e.g. 0.75 means keep 25%
        K: number of layers before pruning is applied
        T: total number of layers
        d: hidden size (e.g., 4096)
        m: FFN intermediate size (e.g., 11008)
    Returns:
        Reduction ratio (0–1)
    """
    n_hat = int(round((1 - R) * n))

    def flops(tokens):
        return 4 * tokens * (d**2) + 2 * (tokens**2) * d + 2 * tokens * d * m

    baseline = T * flops(n)
    pruned = K * flops(n) + (T - K) * flops(n_hat)
    ratio = 1.0 - (pruned / baseline)
    return ratio

def main(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=2048,
        padding_side="right",
        use_fast=True
    )
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).cuda()

    for m in model.modules():
        m.tokenizer = tokenizer

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device='cuda', dtype=torch.float16)
    image_processor = vision_tower.image_processor

    patch_num = getattr(model.config, 'patch_num', '9')
    preprocess = Compose([ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))])

    while True:
        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        image = load_image(args.image_file)

        image_tensor = process_images([image], image_processor, model.config)[0]
        image_tensor = image_tensor.unsqueeze(0)
        h_block = 1
        w_block = 1

        # --- Measure visual tokens from projector (or fallback) BEFORE generation
        visual_tokens = get_visual_token_count(model, image_tensor)

        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()
        # Count how many IMAGE placeholders are in the prompt
        n_image_placeholders = int((input_ids == IMAGE_TOKEN_INDEX).sum().item())

        seq_len = int(input_ids.shape[1])
        seq_len_no_image = seq_len - n_image_placeholders

        K = max(visual_tokens, 0)  # if -1, treat as 0 or fallback
        L_eff = seq_len_no_image + K

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        mode = model.config.image_aspect_ratio
        with torch.inference_mode():
            # model.orig_forward = model.forward
            # model.forward = partial(model.orig_forward, mode=mode, h_block=h_block, w_block=w_block)

            # Timing starts right before generate()
            start = time.time()
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.bfloat16, device='cuda', non_blocking=True),
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens = args.max_new_tokens,
                # streamer=streamer,
                eos_token_id = tokenizer.eos_token_id,  # still ignored until min_new_tokens hit
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )
            # model.forward = model.orig_forward

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        end = time.time()

        elapsed = end - start
        num_new_tokens = output_ids.shape[1] - input_ids.shape[1]
        tok_per_sec = num_new_tokens / elapsed if elapsed > 0 else float("inf")
        conv.messages[-1][-1] = outputs

        d_model = int(getattr(model.config, "hidden_size", getattr(model.config, "n_embd", 0)))
        d_ff    = int(getattr(model.config, "intermediate_size", int(4 * max(d_model,1))))
        n_layers= int(getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layer", 0)))

        prefill_flops, decode_flops, total_flops = estimate_decoder_flops(
            d_model=d_model,
            d_ff=d_ff,
            n_layers=n_layers,
            prompt_len=L_eff,           # <-- include K here
            new_tokens=int(num_new_tokens)
        )

        # Convert to TFLOPs for display
        tf_total = total_flops / 1e12
        tf_prefill = prefill_flops / 1e12
        tf_decode = decode_flops / 1e12

        print(f"\n[Timing] Generation took {elapsed:.3f} seconds "
              f"({num_new_tokens} new tokens, {tok_per_sec:.2f} tok/s)\n")

        # --- Compact metrics line
        print("[Metrics]")
        print(f"  # Visual Tokens: {visual_tokens if visual_tokens >= 0 else 'N/A'}")
        print(f"  Latency (s):    {elapsed:.3f}")
        print(f"  Throughput:     {tok_per_sec:.2f} tok/s")
        print(f"  FLOPs (TF):     total={tf_total:.3f}, prefill={tf_prefill:.3f}, decode={tf_decode:.3f}")

        # --- Ready-to-paste LaTeX table row
        vis_tok_str = f"{visual_tokens}" if visual_tokens >= 0 else "--"
        latex_row = (
            f"{vis_tok_str} & "
            f"{elapsed:.3f} & "
            f"{tok_per_sec:.2f} & "
            f"{tf_total:.3f} & "
            f"{tf_prefill:.3f} & "
            f"{tf_decode:.3f} \\\\"
        )
        print("\n[LaTeX row]")
        print(latex_row + "\n")

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
        # --- Compact metrics line
        print("[Metrics]")
        print(f"  # Visual Tokens: {visual_tokens if visual_tokens >= 0 else 'N/A'}")
        print(f"  Latency (s):    {elapsed:.3f}")
        print(f"  Throughput:     {tok_per_sec:.2f} tok/s")
        print(f"  FLOPs (TF):     total={tf_total:.3f}, prefill={tf_prefill:.3f}, decode={tf_decode:.3f}")
        if args.fastv:
            R = 1.0 - (args.pruned_k / args.baseline_k)
            ratio = fastv_reduction_ratio(
                n=args.baseline_k,
                R=R,
                K=args.prune_layer,
                T=n_layers,
                d=d_model,
                m=d_ff
            )
            print("\n[FastV Eq.(5) Reduction Ratio]")
            print(f"  Baseline n={args.baseline_k}, pruned n={args.pruned_k}, K={args.prune_layer}, T={n_layers}")
            print(f"  Reduction ratio = {ratio*100:.2f}%")

        # NEW: optional human-friendly explanation (in B = billions of FLOPs)
        if args.explain:
            print_flops_explanation(
                model_name=model_name,
                n_layers=n_layers,
                d_model=d_model,
                d_ff=d_ff,
                prompt_len=L_eff,
                new_tokens=int(num_new_tokens),
                prefill_flops=prefill_flops,
                decode_flops=decode_flops,
                total_flops=total_flops,
            )
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="path/to/delatallava")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default='vicuna_v1')
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--explain", action="store_true", help="Print a short params-vs-FLOPs explanation.", default=True)
    parser.add_argument("--fastv", action="store_true",
                        help="Compute FastV-style FLOPs reduction ratio (Eq. 5).", default=True)
    parser.add_argument("--baseline-k", type=int, default=576,
                        help="Baseline image token count (n).")
    parser.add_argument("--pruned-k", type=int, default=144,
                        help="Pruned token count (n_hat).")
    parser.add_argument("--prune-layer", type=int, default=0,
                        help="Layer index K after which pruning is applied.")
    args = parser.parse_args()
    main(args)
