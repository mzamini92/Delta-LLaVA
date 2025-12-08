import argparse
import json
import math
import os
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # kept if you want to add progress later

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from llava.utils import disable_torch_init
# pip install fvcore
from fvcore.nn import FlopCountAnalysis
# Prepare a one-step forward (e.g., pass a single token with past_key_values)
from statistics import mean

# GPU logging
import threading, csv, datetime
try:
    import pynvml
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

# -----------------------------
# Helpers
# -----------------------------
def split_list(lst: List, n: int) -> List[List]:
    """Split a list into n (roughly) equal-sized chunks."""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]
def infer_vision_tokens_from_config(model) -> int:
    """
    Try to read '# of vision tokens passed to the LLM' directly from the pretrained config
    or projector module. Returns 0 if unknown (so caller can fall back to H/patch grid).
    """
    # 1) Common config fields people use for resamplers / token packers
    keys_cfg = [
        "mm_resampler_n_tokens", "mm_resampler_num_tokens",
        "mm_num_query", "num_queries", "n_queries",
        "resampler_num_tokens", "carrier_tokens",
        "target_token_count", "vision_token_count",
        "num_visual_tokens", "mm_vision_tokens", "mm_vision_token_q"
    ]
    for k in keys_cfg:
        v = getattr(getattr(model, "config", object()), k, None)
        if isinstance(v, int) and v > 0:
            return v

    # 2) Look on the projector itself (LLaVA variants differ on where this lives)
    proj = (
        getattr(getattr(model, "model", model), "mm_projector", None)
        or getattr(model, "visual_projector", None)
        or getattr(model, "projector", None)
    )
    keys_proj = ["num_queries", "n_queries", "num_tokens", "K",
                 "target_token_count", "carrier_tokens", "num_query"]
    for k in keys_proj:
        v = getattr(proj, k, None) if proj is not None else None
        if isinstance(v, int) and v > 0:
            return v

    # 3) Sometimes the attribute is nested in a submodule (e.g., Perceiver/Resampler blocks)
    if proj is not None:
        for m in proj.modules():
            for k in keys_proj:
                v = getattr(m, k, None)
                if isinstance(v, int) and v > 0:
                    return v

    return 0


def get_chunk(lst: List, n: int, k: int) -> List:
    chunks = split_list(lst, n)
    if not (0 <= k < len(chunks)):
        raise IndexError(f"chunk_idx {k} out of range for {len(chunks)} chunks")
    return chunks[k]
def safe_decode_ids(ids, tokenizer, skip_special_tokens=False) -> str:
    """Decode while filtering out non-vocabulary ids (e.g., IMAGE_TOKEN_INDEX)."""
    # Some tokenizers may not expose vocab_size as int; guard accordingly.
    try:
        vocab_size = int(getattr(tokenizer, "vocab_size", 0))
    except Exception:
        vocab_size = 0

    # Fallback: if vocab_size missing, try len(tokenizer)
    if vocab_size <= 0:
        try:
            vocab_size = len(tokenizer)
        except Exception:
            vocab_size = 0

    # Filter invalid ids
    clean = [int(t) for t in ids if isinstance(t, (int,)) and 0 <= int(t) < vocab_size]
    if not clean:
        return ""
    return tokenizer.decode(clean, skip_special_tokens=skip_special_tokens)

def count_new_tokens(input_ids: torch.Tensor,
                     sequences: torch.Tensor,
                     pad_id: int | None,
                     bos_id: int | None) -> int:
    """
    Return # of generated tokens, robust to:
      - models that keep/remove the input in `sequences`
      - left/right padding, BOS tokens
    Expects tensors with shape (B, L).
    """
    assert input_ids.dim() == 2 and sequences.dim() == 2 and input_ids.size(0) == 1 and sequences.size(0) == 1
    inp = input_ids[0]
    seq = sequences[0]

    # strip PADs
    if pad_id is not None:
        inp = inp[inp != pad_id]
        seq = seq[seq != pad_id]

    # drop leading BOS (some tokenizers insert it)
    if bos_id is not None and inp.numel() > 0 and inp[0].item() == bos_id:
        inp = inp[1:]
    if bos_id is not None and seq.numel() > 0 and seq[0].item() == bos_id:
        seq = seq[1:]

    Lin, Lout = inp.numel(), seq.numel()

    # Common case: sequences begins with the prompt
    if Lout >= Lin and torch.equal(seq[:Lin], inp):
        return Lout - Lin

    # Some forks return only the generated continuation
    if Lout < Lin:
        # then sequences might be just the continuation; assume all are new
        return Lout

    # Fallback: try to align from the end (rare templates)
    # Find the longest suffix of `inp` that matches a prefix of `seq`
    # (simple O(n) scan, ok for debugging)
    best = 0
    for k in range(min(Lin, Lout), 0, -1):
        if torch.equal(inp[Lin - k :], seq[:k]):
            best = k
            break
    return max(0, Lout - best)

def _strip_specials_for_len(t: torch.Tensor, pad_id: int | None, bos_id: int | None) -> int:
    x = t.clone()
    if pad_id is not None:
        x = x[x != pad_id]
    if bos_id is not None and x.numel() > 0 and int(x[0].item()) == bos_id:
        x = x[1:]
    return int(x.numel())

def estimate_decoder_flops(
    d_model: int,
    d_ff: int,
    n_layers: int,
    n_heads: int,
    prompt_len: int,
    new_tokens: int
) -> tuple[float, float, float]:
    """
    Returns (prefill_flops, decode_flops, total_flops) in FLOPs (not TFLOPs).
    Formulas assume multiply-add = 2 FLOPs and standard Transformer blocks.
    """
    d = d_model
    L0 = prompt_len
    T = new_tokens

    # Per-layer prefill (no KV cache): 8*L0*d^2 + 4*L0*d*d_ff + 4*L0^2*d
    prefill_per_layer = 8*L0*(d**2) + 4*L0*d*d_ff + 4*(L0**2)*d
    prefill = n_layers * prefill_per_layer

    # Per-layer decode per token t with sequence length L0 + (t-1):
    # 8*d^2 + 4*d*d_ff + 4*(L)*d, where L = L0 + (t-1)
    # Sum over t=1..T of 4*(L0 + t - 1)*d  = 4*d * (T*L0 + T*(T-1)/2)
    decode_attn_sum = 4*d * (T*L0 + T*(T-1)/2)
    decode_proj_mlp_sum = T * (8*(d**2) + 4*d*d_ff)
    decode_per_layer = decode_proj_mlp_sum + decode_attn_sum
    decode = n_layers * decode_per_layer

    total = prefill + decode
    return float(prefill), float(decode), float(total)

def tflops_per_second(total_flops: float, seconds: float) -> float:
    return (total_flops / 1e12) / max(seconds, 1e-9)

def _get_patch_size_and_cls_flag(model) -> tuple[int | None, bool]:
    """
    Try to read patch size & whether a CLS token is used.
    Works for common CLIP/ViT configs behind LLaVA.
    """
    patch = None
    use_cls = True
    try:
        vt = getattr(model, "get_vision_tower", lambda: None)()
        if vt is None:
            vt = getattr(model, "vision_tower", None)
        vc = getattr(vt, "vision_tower", vt)
        conf = getattr(vc, "config", None)
        if conf is not None:
            patch = getattr(conf, "patch_size", None) or getattr(conf, "patch_size_", None)
            # Most CLIP/ViT use a CLS token; if your tower is patch-only, flip this to False.
            use_cls = getattr(conf, "use_cls_token", True)
    except Exception:
        pass
    return patch, use_cls

def _guess_tokens_per_image(image_tensor: torch.Tensor, model) -> int:
    """
    Estimate how many vision tokens feed into the projector per image.
    Uses image_tensor spatial size and the patch size.
    """
    # image_tensor: (B, C, H, W)
    if image_tensor.dim() != 4:
        return 0
    _, _, H, W = image_tensor.shape
    patch, use_cls = _get_patch_size_and_cls_flag(model)
    if patch is None or patch <= 0:
        # Fallback: assume ViT-14-like
        patch = 14
    th = max(H // patch, 1)
    tw = max(W // patch, 1)
    tokens = th * tw + (1 if use_cls else 0)
    return tokens

def _linear_flops(n_tokens: int, d_in: int, d_out: int) -> int:
    # Multiply-add counted as 2 FLOPs
    return 2 * n_tokens * d_in * d_out

def _conv1x1_as_linear_dims(conv: torch.nn.Conv2d) -> tuple[int, int]:
    # Treat 1x1 conv as Linear over channels
    return conv.in_channels, conv.out_channels

def _guess_projector_tokens(image_tensor: torch.Tensor, model) -> int:
    # 1) Try to read from common config names for resamplers/selectors
    for name in [
        "mm_resampler_n_tokens", "mm_num_query", "num_queries",
        "n_queries", "resampler_num_tokens", "carrier_tokens",
        "target_token_count", "vision_token_count"
    ]:
        v = getattr(getattr(model, "config", object()), name, None)
        if isinstance(v, int) and v > 0:
            return v

    # 2) Fallback: infer from image H,W and ViT patch (raw grid + optional CLS)
    patch, use_cls = _get_patch_size_and_cls_flag(model)
    if image_tensor.dim() != 4 or (patch is None or patch <= 0):
        return 0
    _, _, H, W = image_tensor.shape
    th, tw = max(H // patch, 1), max(W // patch, 1)
    return th * tw + (1 if use_cls else 0)

def estimate_projector_flops(model, image_tensor: torch.Tensor, override_tokens: int | None = None) -> tuple[float, str]:
    """
    Returns (projector_flops_in_FLOPs, description).
    If override_tokens is provided, we use that as K.
    """
    proj = None
    for attr in ["mm_projector", "visual_projector", "projector"]:
        proj = getattr(getattr(model, "model", model), attr, None)
        if proj is not None:
            break
    if proj is None:
        return 0.0, "no_projector_found"

    # Tokens per image that actually reach the projector
    if override_tokens is not None and override_tokens > 0:
        toks_per_img = override_tokens
        origin = "override"
    else:
        toks_per_img = _guess_projector_tokens(image_tensor, model)
        origin = "config" if toks_per_img else "fallback-from-H/patch"

    if toks_per_img == 0:
        return 0.0, "could_not_infer_tokens"

    B = int(image_tensor.shape[0])
    n_tokens = B * toks_per_img

    import torch.nn as nn
    linear_like = []
    for m in proj.modules():
        if isinstance(m, nn.Linear):
            linear_like.append(("linear", m.in_features, m.out_features))
        elif isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1):
            linear_like.append(("conv1x1", m.in_channels, m.out_channels))

    if not linear_like:
        return 0.0, f"projector_not_linear_like[{origin}]"

    def _linear_flops(n_tokens: int, d_in: int, d_out: int) -> int:
        return 2 * n_tokens * d_in * d_out

    if len(linear_like) == 1:
        _, di, do = linear_like[0]
        flops = _linear_flops(n_tokens, di, do)
        desc = f"single_linear({di}->{do})[{origin}, K={toks_per_img}]"
        return float(flops), desc

    # Use first two linear-like ops (common 2-layer projector)
    (tag1, di1, do1), (tag2, di2, do2) = linear_like[0], linear_like[1]
    if di2 != do1:
        flops = _linear_flops(n_tokens, di1, do1) + _linear_flops(n_tokens, di2, do2)
        desc = f"two_stage_unmatched({di1}->{do1}, {di2}->{do2})[{origin}, K={toks_per_img}]"
    else:
        flops = _linear_flops(n_tokens, di1, do1) + _linear_flops(n_tokens, do1, do2)
        desc = f"two_stage({di1}->{do1}->{do2})[{origin}, K={toks_per_img}]"
    return float(flops), desc


# -----------------------------
# Dataset
# -----------------------------
class CustomDataset(Dataset):
    def __init__(
        self,
        questions: List[dict],
        image_folder: str,
        tokenizer,
        image_processor,
        model_config,
        conv_mode: str,
    ):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, index: int):
        line = self.questions[index]
        image_file = line["image"]

        # Build prompt with image tag(s)
        user_turn = line["conversations"][0]["value"].replace("<image>\n", "").replace("<image>", "")
        if getattr(self.model_config, "mm_use_im_start_end", False):
            img_tokens = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        else:
            img_tokens = DEFAULT_IMAGE_TOKEN
        qs = f"{img_tokens}\n{user_turn}"

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Load/process image
        image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        # Tokenize prompt (with image token index support)
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).squeeze(0)  # (seq,)

        return input_ids, image_tensor, image.size, image_file


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, image_files = zip(*batch)
    input_ids = torch.stack([x for x in input_ids], dim=0)        # (B, S)
    image_tensors = torch.stack([x for x in image_tensors], dim=0)  # (B, C, H, W)
    return input_ids, image_tensors, list(image_sizes), list(image_files)


def create_data_loader(
    questions: List[dict],
    image_folder: str,
    tokenizer,
    image_processor,
    model_config,
    conv_mode: str,
    batch_size: int = 1,
    num_workers: int = 4,
) -> DataLoader:
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(
        questions, image_folder, tokenizer, image_processor, model_config, conv_mode
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

def _get_vision_cfg(model):
    """Try to pull a ViT-like config from common LLaVA vision towers."""
    vt = getattr(model, "get_vision_tower", lambda: None)()
    if vt is None:
        vt = getattr(model, "vision_tower", None)
    vc = getattr(vt, "vision_tower", vt)
    conf = getattr(vc, "config", None)
    # Fallback: some wrappers stick the real config on .vision_tower.vision_model.config
    vm = getattr(vc, "vision_model", None)
    if conf is None and vm is not None:
        conf = getattr(vm, "config", None)
    return conf, vc

def _infer_image_hw(image_tensor: torch.Tensor) -> tuple[int, int]:
    # image_tensor: (B, C, H, W)
    if image_tensor is None or image_tensor.dim() != 4:
        return 0, 0
    return int(image_tensor.shape[2]), int(image_tensor.shape[3])

def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def _vit_layer_flops(S: int, D: int, M: int) -> int:
    """
    One encoder layer FLOPs for sequence length S (incl. CLS if present) and hidden dim D:
      - Q,K,V,Out projections: 8*S*D^2
      - MLP (two linears):   4*S*D*M
      - Self-attention matmuls (full): 4*S^2*D
    """
    return 8*S*(D**2) + 4*S*D*M + 4*(S**2)*D

def _patch_embed_flops(H: int, W: int, p: int, D: int, Cin: int = 3) -> int:
    """
    ViT patch embedding conv (kernel p, stride p), FLOPs = 2 * (#patches) * (Cin*D*p*p)
    """
    if p <= 0 or H <= 0 or W <= 0:
        return 0
    Nh, Nw = H // p, W // p
    S_patches = max(Nh * Nw, 1)
    return 2 * S_patches * (Cin * D * (p * p))

def estimate_vit_vision_flops(model, image_tensor: torch.Tensor) -> tuple[float, str]:
    """
    Estimate FLOPs for a ViT-like vision encoder on the given image_tensor.
    Returns (flops_in_FLOPs, description).
    """
    conf, vc = _get_vision_cfg(model)
    H, W = _infer_image_hw(image_tensor)

    # Try to read common fields; provide sane fallbacks.
    patch = _safe_int(getattr(conf, "patch_size", getattr(conf, "patch_size_", 14)), 14)
    D     = _safe_int(getattr(conf, "hidden_size", getattr(conf, "vision_embed_dim", 0)), 0)
    L     = _safe_int(getattr(conf, "num_hidden_layers", getattr(conf, "vision_layers", 0)), 0)
    M     = _safe_int(getattr(conf, "intermediate_size", 4 * max(D, 1)), 4 * max(D, 1))
    use_cls = bool(getattr(conf, "use_cls_token", True))

    # If dims are missing (non-ViT), bail gracefully.
    if D <= 0 or L <= 0:
        return 0.0, "vision_not_vit_like_or_unknown_dims"

    # Sequence length S (patches + optional CLS)
    Nh, Nw = (H // patch if patch > 0 else 0), (W // patch if patch > 0 else 0)
    S = max(Nh * Nw, 1) + (1 if use_cls else 0)

    # Patch embed + encoder stack
    fl_patch = _patch_embed_flops(H, W, patch, D, Cin=3)
    fl_layers = L * _vit_layer_flops(S, D, M)
    total = fl_patch + fl_layers

    desc = f"ViT-like: HxW={H}x{W}, patch={patch}, layers={L}, D={D}, M={M}, S={S}"
    return float(total), desc

# -----------------------------
# Main eval
# -----------------------------
import time
class GpuMonitor:
    """
    Polls NVML for utilization/memory/temperature/power while .start()..generate..().stop().
    Writes a CSV if output_path is provided. Works on single current CUDA device.
    """
    def __init__(self, poll_ms=100, device_index=None, output_path=None):
        self.poll_s = max(poll_ms, 1) / 1000.0
        self.device_index = device_index
        self.output_path = output_path
        self._stop = threading.Event()
        self._thr = None
        self.rows = []  # (iso_time, util_gpu, util_mem, mem_used_MB, mem_total_MB, temp_C, power_W)

    def _resolve_device(self):
        if self.device_index is not None:
            return self.device_index
        if torch.cuda.is_available():
            return int(torch.cuda.current_device())
        return 0

    def _poll(self):
        idx = self._resolve_device()
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        meminfo_total = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**2)  # MB

        while not self._stop.is_set():
            ts = datetime.datetime.utcnow().isoformat() + "Z"
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = None
                power = None
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    pass
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # W
                except Exception:
                    pass
                self.rows.append((
                    ts,
                    getattr(util, "gpu", None),
                    getattr(util, "memory", None),
                    meminfo.used / (1024**2),  # MB
                    meminfo_total,
                    temp,
                    power
                ))
            except Exception:
                # Swallow intermittent NVML errors
                pass
            self._stop.wait(self.poll_s)

    def start(self):
        if not _NVML_AVAILABLE:
            return
        try:
            pynvml.nvmlInit()
        except Exception:
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._poll, daemon=True)
        self._thr.start()

    def stop(self):
        if not _NVML_AVAILABLE:
            return
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=2.0)
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    def write_csv(self):
        if not self.rows or not self.output_path:
            return
        header = ["timestamp", "util_gpu_pct", "util_mem_pct", "mem_used_MB", "mem_total_MB", "temp_C", "power_W"]
        with open(self.output_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(self.rows)

    def summary(self):
        if not self.rows:
            return {}
        util_gpu = [r[1] for r in self.rows if isinstance(r[1], (int, float))]
        mem_used = [r[3] for r in self.rows if isinstance(r[3], (int, float))]
        temp_c   = [r[5] for r in self.rows if isinstance(r[5], (int, float))]
        power_w  = [r[6] for r in self.rows if isinstance(r[6], (int, float))]

        def _avg(x): return sum(x)/len(x) if x else None
        def _mx(x):  return max(x) if x else None

        return {
            "util_gpu_avg": _avg(util_gpu),
            "util_gpu_max": _mx(util_gpu),
            "mem_used_max_MB": _mx(mem_used),
            "temp_max_C": _mx(temp_c),
            "power_max_W": _mx(power_w),
        }

def eval_model(args):
    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    disable_torch_init()

    # Load model/tokenizer/processor
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    # Resolve vision token count once from the pretrained model
    if args.vision_tokens <= 0:
        auto_K = infer_vision_tokens_from_config(model)
        if auto_K > 0:
            args.vision_tokens = auto_K
            print(f"[info] Using vision_tokens from model config: {args.vision_tokens}")
        else:
            # Leave as 0; we'll fall back to H/patch later with the actual image size.
            print("[info] No explicit vision token count in config; will fall back to H/patch grid.")
    # Load questions and select chunk
    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = [json.loads(q) for q in f]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    # Auto-switch to mmtag if needed
    conv_mode = args.conv_mode
    if (
        "plain" in model_name
        and "finetune" not in model_name.lower()
        and "mmtag" not in conv_mode
    ):
        conv_mode = f"{conv_mode}_mmtag"
        print(f"[info] Auto-switched conv mode to: {conv_mode}")

    # Data
    os.makedirs(args.output_folder, exist_ok=True)
    loader = create_data_loader(
        questions,
        args.image_folder,
        tokenizer,
        image_processor,
        model.config,
        conv_mode=conv_mode,
    )

    # -----------------------------
    # Multi-sample evaluation
    # -----------------------------
    n_total = min(args.n_samples, len(loader))
    warmup = max(0, args.warmup)

    # Metrics we’ll aggregate
    lat_list = []
    newtok_list = []
    tokps_list = []

    dec_pref_list, dec_dec_list, dec_tot_list = [], [], []
    proj_list, vis_list, all_list = [], [], []
    thr_dec_list, thr_all_list = [], []

    # Iterate
    for idx, batch in enumerate(loader):
        if idx >= n_total:
            break

        input_ids, image_tensor, image_sizes, image_files = batch

        # (Optional) Save first original image only
        if idx == 0 and image_files:
            orig_path = os.path.join(args.image_folder, image_files[0])
            save_dir = os.path.join(args.output_folder, "originals")
            os.makedirs(save_dir, exist_ok=True)
            from shutil import copyfile
            copyfile(orig_path, os.path.join(save_dir, os.path.basename(orig_path)))

        # Device moves
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_ids = input_ids.to(device=device, non_blocking=True)
        image_tensor = image_tensor.to(
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device=device,
            non_blocking=True
        )

        # Safe pad/eos defaults (common for LLaMA-ish tokenizers)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

        # -----------------------------
        # Measure latency + tokens/sec (CUDA-synchronized)
        # -----------------------------
        # -----------------------------
        # Force EXACTLY N new tokens per sample
        # -----------------------------

        # Optional: clamp to context window to avoid overflow
        max_ctx = getattr(model.config, "max_position_embeddings", None)
        if max_ctx is not None:
            # rough prompt length after stripping PAD/BOS (same as below)
            prompt_len = _strip_specials_for_len(
                input_ids[0],
                tokenizer.pad_token_id,
                getattr(tokenizer, "bos_token_id", None)
            )
            room = max_ctx - prompt_len


        # Prepare per-sample GPU logger path
        gpu_log_dir = os.path.join(args.output_folder, "gpu_logs")
        if args.log_gpu:
            os.makedirs(gpu_log_dir, exist_ok=True)
            per_sample_csv = os.path.join(gpu_log_dir, f"sample_{idx:04d}.csv")
        else:
            per_sample_csv = None

        # Reset PyTorch mem stats for clean per-sample maxima
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # -----------------------------
        # Measure latency + tokens/sec
        # -----------------------------
        target_new = int(args.max_new_tokens)

        if device == "cuda":
            torch.cuda.synchronize()

        # start GPU monitor before timing, if you want it running during gen
        mon = None
        if args.log_gpu and _NVML_AVAILABLE and torch.cuda.is_available():
            mon = GpuMonitor(
                poll_ms=args.gpu_poll_ms,
                device_index=int(torch.cuda.current_device()),
                output_path=per_sample_csv,
            )
            mon.start()

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        with torch.inference_mode():
            out = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=bool(args.temperature > 0),
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # enforce exact length
                early_stopping=True,                 # don't end beams early
                eos_token_id=tokenizer.eos_token_id,  # EOS ignored until min_new_tokens reached
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                output_attentions=False,
                return_dict_in_generate=True,
                max_time=args.max_time_s,  
            )

        if device == "cuda":
            torch.cuda.synchronize()
        end = time.time()
        latency = end - start

        ### GPU LOGGING: stop sampler and persist CSV
        if mon is not None:
            mon.stop()
            mon.write_csv()
            mon_summary = mon.summary()
        else:
            mon_summary = {}

        latency = end - start

        # --- PyTorch memory (max during this sample) ---
        if device == "cuda":
            max_alloc_gb = torch.cuda.max_memory_allocated() / (1024**3)
            max_rsrv_gb  = torch.cuda.max_memory_reserved()  / (1024**3)
        else:
            max_alloc_gb = max_rsrv_gb = 0.0

        # (your existing new token counting and FLOPs computation...)


        # New tokens & tokens/sec
        new_tokens = count_new_tokens(
            input_ids=input_ids,
            sequences=out["sequences"],
            pad_id=tokenizer.pad_token_id,
            bos_id=getattr(tokenizer, "bos_token_id", None),
        )
        tok_per_sec = new_tokens / latency if latency > 0 else float("nan")


        # --- Decoder FLOPs ---
        d_model = int(getattr(model.config, "hidden_size", getattr(model.config, "n_embd", 0)))
        d_ff    = int(getattr(model.config, "intermediate_size", int(4 * d_model)))
        n_layers = int(getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layer", 0)))
        n_heads  = int(getattr(model.config, "num_attention_heads", getattr(model.config, "n_head", 0)))

        # After you compute prompt_len and before estimate_decoder_flops(...)
        # Get K = # vision tokens seen by the LLM (post-resampler/projector)
        if args.vision_tokens > 0:
            K = int(args.vision_tokens)
        else:
            # fall back to a guess from config or H/patch
            K = _guess_projector_tokens(image_tensor, model)

        prompt_len_eff = prompt_len + K  # include vision tokens in the LLM context length

        prefill_flops, decode_flops, total_flops = estimate_decoder_flops(
            d_model=d_model,
            d_ff=d_ff,
            n_layers=n_layers,
            n_heads=n_heads,
            prompt_len=prompt_len_eff,   # <-- use effective prompt length
            new_tokens=new_tokens,
        )

        decoder_tflops_s = tflops_per_second(total_flops, latency)

        # --- Projector FLOPs ---
        proj_flops, proj_desc = estimate_projector_flops(
            model, image_tensor, override_tokens=(args.vision_tokens if args.vision_tokens > 0 else None)
        )

        # --- Vision encoder FLOPs ---
        vision_flops, vision_desc = estimate_vit_vision_flops(model, image_tensor)

        # --- Totals ---
        total_with_proj = total_flops + proj_flops
        total_with_proj_vis = total_with_proj + vision_flops
        thr_all = tflops_per_second(total_with_proj_vis, latency)

        # Prompt length
        prompt_len = _strip_specials_for_len(
            input_ids[0],
            tokenizer.pad_token_id,
            getattr(tokenizer, "bos_token_id", None)
        )

        # Per-sample quick line (append GPU stats)
        gpu_str = ""
        if mon_summary:
            gpu_str = (f" | GPU util avg/max={mon_summary.get('util_gpu_avg',0):.0f}%/"
                    f"{mon_summary.get('util_gpu_max',0):.0f}%, "
                    f"mem_peak≈{mon_summary.get('mem_used_max_MB',0):.0f}MB, "
                    f"Tmax={mon_summary.get('temp_max_C','-')}°C, "
                    f"Pmax={mon_summary.get('power_max_W','-')}W")
        pt_mem_str = f" | torch max_alloc={max_alloc_gb:.2f}GB max_reserved={max_rsrv_gb:.2f}GB" if device=="cuda" else ""

        print(f"[{idx+1}/{n_total}] lat={latency:.3f}s  new={new_tokens:3d}  tok/s={tok_per_sec:6.2f}  "
            f"dec={total_flops/1e12:.3f}T  proj={proj_flops/1e12:.3f}T  vis={vision_flops/1e12:.3f}T  "
            f"all={total_with_proj_vis/1e12:.3f}T{gpu_str}{pt_mem_str}")

        # # Per-sample quick line (kept brief)
        # print(f"[{idx+1}/{n_total}] lat={latency:.3f}s  new={new_tokens:3d}  tok/s={tok_per_sec:6.2f}  "
        #       f"dec={total_flops/1e12:.3f}T  proj={proj_flops/1e12:.3f}T  vis={vision_flops/1e12:.3f}T  "
        #       f"all={total_with_proj_vis/1e12:.3f}T")

        # Collect (skip warmup in averages)
        if idx >= warmup:
            lat_list.append(latency)
            newtok_list.append(new_tokens)
            tokps_list.append(tok_per_sec)

            dec_pref_list.append(prefill_flops / 1e12)
            dec_dec_list.append(decode_flops / 1e12)
            dec_tot_list.append(total_flops / 1e12)

            proj_list.append(proj_flops / 1e12)
            vis_list.append(vision_flops / 1e12)
            all_list.append(total_with_proj_vis / 1e12)

            thr_dec_list.append(decoder_tflops_s)
            thr_all_list.append(thr_all)

    # -----------------------------
    # Averages
    # -----------------------------
    # === Subset: exactly 128 new tokens ===
    # right after you set target_new (or just reuse args.max_new_tokens)
    subset_N = int(target_new)  # or: subset_N = int(args.max_new_tokens)

    idxN = [i for i, n in enumerate(newtok_list) if n == subset_N]

    print(f"\n=== Subset Averages ({subset_N} new tokens only) ===")
    print(f"samples: {len(idxN)}")
    # -----------------------------
    # Averages (all completed, excluding warmup)
    # -----------------------------
    if tokps_list:
        avg_lat    = mean(lat_list)
        avg_tokps  = mean(tokps_list)
        avg_newtok = mean(newtok_list)

        avg_prefill = mean(dec_pref_list) if dec_pref_list else 0.0
        avg_decode  = mean(dec_dec_list)  if dec_dec_list else 0.0
        avg_dec_tot = mean(dec_tot_list)  if dec_tot_list else 0.0
        avg_proj    = mean(proj_list)     if proj_list    else 0.0
        avg_vis     = mean(vis_list)      if vis_list     else 0.0
        avg_all     = mean(all_list)      if all_list     else 0.0

        agg_thr_dec = (sum(dec_tot_list) / max(sum(lat_list), 1e-9)) if dec_tot_list else 0.0
        agg_thr_all = (sum(all_list)     / max(sum(lat_list), 1e-9)) if all_list     else 0.0

        print("\n=== Overall Averages (natural stop) ===")
        print(f"samples:                     {len(tokps_list)}")
        print(f"Avg latency:                 {avg_lat:.3f} s")
        print(f"Avg new tokens:              {avg_newtok:.2f}")
        print(f"Avg tokens/sec (mean):       {avg_tokps:.2f}")
        print(f"Avg total FLOPs:             {avg_all:.3f} TFLOPs")
        print(f"Breakdown (TFLOPs): decoder {avg_dec_tot:.3f}, vision {avg_vis:.3f}, projector {avg_proj:.3f}")
        print(f"  Decoder prefill:           {avg_prefill:.3f}")
        print(f"  Decoder decode:            {avg_decode:.3f}")
        print("Throughput (TFLOPs/s):")
        print(f"  Decoder (aggregate):       {agg_thr_dec:.2f}")
        print(f"  All (aggregate):           {agg_thr_all:.2f}")

        # Optional: distribution of generated lengths
        try:
            import statistics as _st
            p50 = _st.median(newtok_list)
            p90 = sorted(newtok_list)[int(0.9*len(newtok_list))-1]
            p99 = sorted(newtok_list)[int(0.99*len(newtok_list))-1]
            print(f"Length distribution (new tokens): median={p50:.0f}, p90={p90:.0f}, p99={p99:.0f}")
        except Exception:
            pass
    else:
        print("\nNo samples collected (after warmup).")



# -----------------------------
# CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--output-folder", type=str, default="attn_outputs")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=2, help="Which sample in the chunk to run")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=1000, help="How many samples to run (max limited by dataset)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs excluded from averages")
    parser.add_argument(
        "--vision-tokens",
        type=int,
        default=0,   # 0/negative => auto
        help="If >0, force this many vision tokens; if <=0, read from pretrained config, else fall back to H/patch."
    )
    parser.add_argument("--log-gpu", default=True, action="store_true",
                        help="If set, logs GPU utilization/memory/temperature/power while generating.")
    parser.add_argument("--gpu-poll-ms", type=int, default=100,
                        help="Polling interval in milliseconds for GPU sampler.")
    parser.add_argument("--max_time_s", type=float, default=None,
                        help="Optional wall-clock cap per sample; generation stops when this time is hit. "
                            "Keeps runs bounded without fixing token count.")                        

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    eval_model(args)
