#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLaVA Evaluator on a Custom Generated Dataset

Usage:
  python llava_eval.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --data-file /path/to/dataset.jsonl \
    --image-root /path/to/images \
    --out /path/to/results.jsonl \
    --csv /path/to/summary.csv

Dataset expectations (flexible):
- Each example is a dict with:
    {
      "image": "val_000001",            # or "image_filename": "val_000001.png"
      "question": "How many ...?",
      "answer": "3",
      "template": "q_count_total"       # optional; can also be "template_filename" or "question_family_index"
    }
- If "image" has no extension, the script will try common ones: .png, .jpg, .jpeg, .webp

Metrics:
- Exact-match (string normalized).
- Numeric-match with tolerance for floats (configurable via --num-tol).
- Yes/No normalization (true/false/yes/no variants).
- Per-template accuracy & overall accuracy.

Notes:
- Requires llava installed and a compatible pretrained model.
- For speed/repro, defaults to greedy decoding (temperature=0.0).
"""
import glob

import argparse
import json
import jsonlines
import os
import re
import string
import sys
import time
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

# LLaVA imports (make sure llava is in PYTHONPATH)
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


# -----------------------------
# Helpers
# -----------------------------

EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]

YES_SET = {"yes", "y", "true", "1", "yeah", "yep"}
NO_SET  = {"no", "n", "false", "0", "nope"}

def _strip_punct(s: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return s.translate(table)

def normalize_text(s: str) -> str:
    if s is None: return ""
    s = s.strip().lower()
    s = s.replace("\u00a0", " ")  # NBSP
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_for_em(s: str) -> str:
    return normalize_text(_strip_punct(s))

def try_parse_number(s: str) -> Optional[float]:
    s_norm = normalize_text(s)
    # extract first number-like token (handles "3.", "3,000", etc. simply)
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s_norm)
    if not m:
        return None
    try:
        # also remove commas
        token = m.group(0).replace(",", "")
        return float(token)
    except Exception:
        return None

def yesno_normalize(s: str) -> Optional[str]:
    t = normalize_for_em(s)
    if t in YES_SET: return "yes"
    if t in NO_SET:  return "no"
    return None

def resolve_image_path(image_root: str, image_field: str) -> str:
    # If it already includes an extension and exists, return it
    p = os.path.join(image_root, image_field)
    if os.path.isfile(p):
        return p
    # If there is no extension, try common ones
    base, ext = os.path.splitext(image_field)
    candidates = [p] if ext else [os.path.join(image_root, base + e) for e in EXTS]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"Image not found for '{image_field}' under '{image_root}'. Tried: {candidates}")

def get_group_key(d: Dict[str, Any]) -> str:
    # Prefer "template" then "template_filename", then "question_family_index"
    if "template" in d and d["template"]:
        return str(d["template"])
    if "template_filename" in d and d["template_filename"]:
        return str(d["template_filename"])
    if "question_family_index" in d:
        return f"family_{d['question_family_index']}"
    return "unknown"


# -----------------------------
# LLaVA inference wrapper
# -----------------------------

@dataclass
class LlavaState:
    tokenizer: Any
    model: Any
    image_processor: Any
    device: str
    conv_mode: str

def load_llava(model_path: str, device: str = "cuda", load_8bit: bool = False, load_4bit: bool = False, dtype: str = "auto") -> LlavaState:
    model_name = get_model_name_from_path(model_path)
    if dtype == "auto":
        torch_dtype = "auto"
    elif dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = "auto"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        torch_dtype=torch_dtype,
        device=device,
        use_flash_attn=False
    )

    # Pick a default conversation template
    conv_mode = "llava_v1"
    if "v1.6" in model_name.lower():
        conv_mode = "llava_v1"  # still fine; adjust if you use a different template

    return LlavaState(tokenizer, model, image_processor, device, conv_mode)

@torch.no_grad()
def generate_answer(state: LlavaState, image: Image.Image, question: str, max_new_tokens: int = 64,
                    temperature: float = 0.0, top_p: float = 1.0, stop_str: Optional[str] = None) -> str:
    """
    Single-image single-question generation, greedy by default.
    """
    tokenizer = state.tokenizer
    model = state.model
    image_processor = state.image_processor
    device = state.device

    # Prepare image tensor
    image_tensor = process_images([image], image_processor, model.config).to(device, dtype=torch.float16 if model.dtype==torch.float16 else None)

    # Build conversation prompt with special image tokens
    conv = conv_templates[state.conv_mode].copy()
    prompt = question.strip()
    conv.append_message(conv.roles[0], f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{prompt}")
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(
        conv.get_prompt(),
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )

    # Ensure 2D (batch, seq)
    if isinstance(input_ids, torch.Tensor):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
    else:
        # In rare cases some tokenizers return lists; coerce to tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)

    input_ids = input_ids.to(device)

    # Stopping
    stop_str = stop_str or conv.sep if conv.sep_style not in [SeparatorStyle.TWO] else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # Generate
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        stopping_criteria=[stopping_criteria]
    )
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    # Post-trim by stop_str if present
    if stop_str and outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)].strip()
    return outputs

def _natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]

def build_fallback_sequence(
    image_root: str,
    image_list_path: Optional[str],
    image_glob: Optional[str]
) -> Optional[List[str]]:
    if image_list_path:
        with open(image_list_path, "r") as f:
            seq = [ln.strip() for ln in f if ln.strip()]
        return seq
    if image_glob:
        paths = glob.glob(os.path.join(image_root, image_glob))
        # sort naturally so ..._9.png < ..._10.png is correct
        paths.sort(key=_natural_key)
        # return just basenames; resolve_image_path will join root
        return [os.path.basename(p) for p in paths]
    return None

# -----------------------------
# Scoring
# -----------------------------

def score_prediction(pred: str, gold: str, num_tol: float) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns (is_correct, extra_info)
    - Exact match (punct/space insensitive)
    - If numeric parse succeeds for either, compare numerically within tol
    - If yes/no, compare normalized yes/no tokens
    """
    pred_raw = pred or ""
    gold_raw = gold or ""

    pred_em = normalize_for_em(pred_raw)
    gold_em = normalize_for_em(gold_raw)

    # yes/no path
    pred_yn = yesno_normalize(pred_raw)
    gold_yn = yesno_normalize(gold_raw)
    if (pred_yn is not None) and (gold_yn is not None):
        return (pred_yn == gold_yn), {"mode": "yesno", "pred": pred_yn, "gold": gold_yn}

    # numeric path
    pnum = try_parse_number(pred_raw)
    gnum = try_parse_number(gold_raw)
    if (pnum is not None) and (gnum is not None):
        ok = abs(pnum - gnum) <= num_tol
        return ok, {"mode": "numeric", "pred": pnum, "gold": gnum, "tol": num_tol}

    # fallback exact-match (punctuation/space-insensitive)
    ok = (pred_em == gold_em)
    return ok, {"mode": "exact", "pred": pred_em, "gold": gold_em}
def coerce_example(ex: Any) -> Dict[str, Any]:
    if isinstance(ex, dict) and ("question" in ex or "answer" in ex):
        return ex
    # keep any outer image fields if present
    outer_img = None
    if isinstance(ex, dict):
        outer_img = ex.get("image_filename") or ex.get("image") or ex.get("image_path")
    conv = None
    if isinstance(ex, dict) and isinstance(ex.get("conversations"), list):
        conv = ex["conversations"]
    elif isinstance(ex, list) and ex and isinstance(ex[0], dict) and ("from" in ex[0] or "role" in ex[0]):
        conv = ex
    else:
        return {"question": "", "answer": "", "image_filename": outer_img} if outer_img else {"question": "", "answer": ""}

    q, a, img = "", "", None
    for m in conv:
        role = (m.get("from") or m.get("role") or "").lower()
        val  = (m.get("value") or m.get("content") or "").strip()
        if role in ("human","user"):
            q = val.replace("<image>", "").strip()
            m2 = re.search(r"<image:([^>]+)>", val)
            if m2:
                img = os.path.basename(m2.group(1).strip())
        elif role in ("gpt","assistant"):
            a = val
    out = {"question": q, "answer": a}
    out_img = img or outer_img
    if out_img:
        out["image_filename"] = out_img
    return out

def load_dataset(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    if path.endswith(".jsonl"):
        with jsonlines.open(path, "r") as reader:
            for obj in reader:
                data.append(coerce_example(obj))
    elif path.endswith(".json"):
        with open(path, "r") as f:
            payload = json.load(f)
            if isinstance(payload, dict) and "questions" in payload:
                data = [coerce_example(o) for o in payload["questions"]]
            elif isinstance(payload, list):
                data = [coerce_example(o) for o in payload]
            else:
                raise ValueError("JSON must be a list of examples or contain a 'questions' list.")
    else:
        raise ValueError("Unsupported dataset extension. Use .json or .jsonl")

    # prune empties/malformed
    data = [d for d in data if isinstance(d, dict) and (d.get("question") or "").strip() and ("answer" in d)]
    if not data:
        raise ValueError("Dataset is empty after coercion.")
    return data

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a LLaVA model on a custom dataset.")
    parser.add_argument("--model-path", required=True, help="HF repo or local path for the LLaVA model.")
    parser.add_argument("--data-file", required=True, help="Path to dataset .json or .jsonl")
    parser.add_argument("--image-root", required=True, help="Directory containing images referenced by dataset")
    parser.add_argument("--out", default="predictions.jsonl", help="Where to write per-example predictions (JSONL)")
    parser.add_argument("--csv", default="summary.csv", help="Where to write the summary metrics CSV")
    parser.add_argument("--device", default="cuda", help="Device: 'cuda' or 'cpu'")
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16"], help="Model dtype hint")
    parser.add_argument("--load-8bit", action="store_true", help="Load model in 8-bit")
    parser.add_argument("--load-4bit", action="store_true", help="Load model in 4-bit")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-tol", type=float, default=0.0, help="Numeric tolerance (abs error) for numeric answers")
    parser.add_argument("--max-samples", type=int, default=0, help="If >0, limit the number of evaluated samples")
    parser.add_argument("--seed", type=int, default=42)
    # add to argparse in main()

    parser.add_argument("--fallback-image-pattern", default=None,
                        help="If no image field, build filename with this pattern, e.g. 'CLEVR_new_{idx:06d}.png'")
    parser.add_argument("--index-start", type=int, default=0,
                        help="Starting index for {idx} in pattern fallback (default: 0)")
    parser.add_argument("--fallback-image-glob", default=None,
                        help="Glob (relative to --image-root) to build a sorted fallback list, e.g. 'CLEVR_new_*.png'")
    parser.add_argument("--image-list", default=None,
                        help="Path to a text file containing one image filename per line (aligned to dataset order).")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # Load dataset
    data = load_dataset(args.data_file)
    if args.max_samples and args.max_samples > 0:
        data = data[: args.max_samples]

    # Load model
    print("Loading model...", file=sys.stderr)
    state = load_llava(
        model_path=args.model_path,
        device=args.device,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        dtype=args.dtype,
    )
    print("Model loaded.", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)

    total = 0
    correct = 0
    per_group_counts = Counter()
    per_group_correct = Counter()

    t0 = time.time()
    fallback_seq = build_fallback_sequence(
        image_root=args.image_root,
        image_list_path=args.image_list,
        image_glob=args.fallback_image_glob
    )
    with jsonlines.open(args.out, "w") as writer:
        for i, ex in enumerate(data, start=args.index_start):
            q = ex.get("question", "").strip()
            gold = ex.get("answer", "")

            img_field = ex.get("image_filename") or ex.get("image")
            if not img_field:
                if fallback_seq and (0 <= (i - args.index_start) < len(fallback_seq)):
                    img_field = fallback_seq[i - args.index_start]
                elif args.fallback_image_pattern:
                    try:
                        img_field = args.fallback_image_pattern.format(idx=i)
                    except Exception as e:
                        print(f"[WARN] Pattern format failed for index {i}: {e}", file=sys.stderr)

            if not img_field:
                print("Skipping example with no image field:", {"question": q, "answer": gold}, file=sys.stderr)
                continue


            try:
                img_path = resolve_image_path(args.image_root, img_field)
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] Unable to open image for {img_field}: {e}", file=sys.stderr)
                continue

            # Generate
            pred = generate_answer(
                state=state,
                image=image,
                question=q,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            print(f"Pred: {pred}")

            ok, meta = score_prediction(pred, gold, num_tol=args.num_tol)
            group_key = get_group_key(ex)

            writer.write({
                "image": img_field,
                "image_path": img_path,
                "question": q,
                "pred": pred,
                "answer": gold,
                "correct": bool(ok),
                "mode": meta.get("mode"),
                "group": group_key
            })
            print(f"[{total+1}] Q: {q}\n   Pred: {pred}\n   Gold: {gold}\n   Correct: {ok}\n", file=sys.stderr)


            total += 1
            if ok: correct += 1
            per_group_counts[group_key] += 1
            if ok: per_group_correct[group_key] += 1

            if total % 20 == 0:
                acc = 100.0 * correct / max(1, total)
                print(f"Processed {total} examples | running acc = {acc:.2f}%", file=sys.stderr)

    elapsed = time.time() - t0
    overall_acc = 100.0 * correct / max(1, total)

    # Write summary CSV
    import csv
    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "n", "correct", "accuracy_percent"])
        # per-group
        for g, n in sorted(per_group_counts.items(), key=lambda kv: kv[0]):
            c = per_group_correct[g]
            writer.writerow([g, n, c, 100.0 * c / max(1, n)])
        # overall
        writer.writerow(["OVERALL", total, correct, overall_acc])

    print(f"Done. Evaluated {total} examples in {elapsed:.1f}s | Overall ACC = {overall_acc:.2f}%")
    print(f"Predictions written to: {args.out}")
    print(f"Summary CSV written to: {args.csv}")


if __name__ == "__main__":
    main()
