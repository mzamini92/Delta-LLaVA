import argparse
import torch
import os
# === INSERT: imports near the top ===
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

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
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from functools import partial
import time
# ===================== Grad-CAM helpers (LLaVA) =====================
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
def infer_grid_hw(P: int) -> tuple[int, int]:
    """Find an HxW such that H*W=P, preferring square-ish; fallback to nearest factors."""
    r = int(round(P ** 0.5))
    if r * r == P:
        return r, r
    # search factors near sqrt
    h = int(np.floor(np.sqrt(P)))
    for h_try in range(h, 0, -1):
        if P % h_try == 0:
            return h_try, P // h_try
    # fallback rectangle
    return r, max(1, P // r)

def save_token_grid_image(cam_tokens_1d: torch.Tensor,
                          out_png="gradcam_token_grid.png",
                          draw_grid=True):
    """
    Save a *token-grid* image (no upsampling). cam_tokens_1d shape: (P,)
    Shows exactly P cells (1, 144, 576, ...).
    """
    P = int(cam_tokens_1d.numel())
    H, W = infer_grid_hw(P)
    grid = cam_tokens_1d[:H*W].reshape(H, W).detach().cpu().numpy()
    # normalize 0..1
    g = grid - grid.min()
    if g.max() > 0:
        g = g / (g.max() + 1e-8)

    plt.figure(figsize=(max(3, W/4), max(3, H/4)))
    plt.imshow(g, cmap='jet', interpolation='nearest')  # <= blocky cells
    if draw_grid:
        # draw thin gridlines between cells
        for y in range(H):
            plt.axhline(y-0.5, color='k', linewidth=0.3)
        for x in range(W):
            plt.axvline(x-0.5, color='k', linewidth=0.3)
    plt.xticks([]); plt.yticks([]); plt.tight_layout(pad=0.1)
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()
    print(f"[Grad-CAM] Saved token-grid ({H}x{W}={P}): {out_png}")

def save_overlay_nearest(rgb_image_pil, cam_2d, out_png="gradcam_overlay_nearest.png"):
    """Overlay using NEAREST upsampling so block boundaries stay crisp."""
    img = np.array(rgb_image_pil).astype(np.float32) / 255.0
    H, W = img.shape[:2]
    cam = cam_2d
    if cam.shape != (H, W):
        cam = F.interpolate(torch.from_numpy(cam)[None, None].float(),
                            size=(H, W), mode='nearest')[0,0].numpy()
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / (cam.max() + 1e-8)
    cm = plt.get_cmap('jet')
    heat = cm(cam)[..., :3]
    overlay = (0.45 * heat + 0.55 * img)
    overlay = (overlay * 255).clip(0,255).astype(np.uint8)
    plt.figure()
    plt.imshow(overlay); plt.axis('off'); plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()
    print(f"[Grad-CAM] Saved NEAREST overlay: {out_png}")

# replaces _ProjTapPre
class _ProjTapPre:
    """
    Pre-hook on mm_projector:
      - finds the FIRST Tensor in (args, kwargs)
      - clones it to float32 + requires_grad=True so BF16 autograd issues vanish
      - stores it as self.feats and returns the modified inputs
    """
    def __init__(self, mm_projector):
        self.feats = None
        self._hits = 0
        self.h = mm_projector.register_forward_pre_hook(self._pre, with_kwargs=True)

    def _wrap_first_tensor(self, obj):
        def _inner(x, replaced):
            if replaced:
                return x, True
            if isinstance(x, torch.Tensor):
                x2 = x.detach().to(torch.float32).clone().requires_grad_(True)
                self.feats = x2
                return x2, True
            if isinstance(x, (list, tuple)):
                out, did = [], False
                for it in x:
                    it2, did = _inner(it, did)
                    out.append(it2)
                return (type(x)(out), did)
            if isinstance(x, dict):
                out, did = {}, False
                for k, v in x.items():
                    v2, did = _inner(v, did)
                    out[k] = v2
                return out, did
            return x, False
        return _inner(obj, False)

    def _pre(self, module, args, kwargs):
        new_args, did = self._wrap_first_tensor(list(args))
        if not did:
            new_kwargs, did2 = self._wrap_first_tensor(kwargs)
            if not did2:
                raise RuntimeError("mm_projector pre-hook: no Tensor input found.")
            kwargs = new_kwargs
        else:
            args = tuple(new_args)
        self._hits += 1
        return args, kwargs

    @property
    def grads(self):
        return None if (self.feats is None) else self.feats.grad

    def close(self):
        self.h.remove()
# Replace _ProjTapPre with a post-hook that captures the projector OUTPUT

class _ProjTapPost:
    """
    Forward hook on mm_projector:
      - captures the projector *output* tensor and retains grad
      - exposes feats (activations) and grads at packed resolution (grid^2)
    """
    def __init__(self, mm_projector):
        self.feats = None
        self._hits = 0
        self.h = mm_projector.register_forward_hook(self._post, with_kwargs=True)

    def _post(self, module, args, kwargs, output):
        # output is typically [B, grid^2, hidden_size] for your TokenPacker
        if isinstance(output, torch.Tensor):
            out = output
        elif isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            out = output[0]
        else:
            raise RuntimeError("Unexpected projector output type for Grad-CAM.")
        self.feats = out.detach().requires_grad_(True)
        # swap in requires_grad copy so backprop hits this tensor
        return self.feats

    @property
    def grads(self):
        return None if (self.feats is None or self.feats.grad is None) else self.feats.grad

    def close(self):
        self.h.remove()
def packed_grid_hw_from_model(model):
    # TokenPacker config: raw_grid and scale_factor exist on build
    cfg = getattr(model.config, "mm_projector_cfg", None)
    if cfg is None:
        # or reach into the actual module if accessible
        tp = getattr(model, "mm_projector", None)
        raw_grid = getattr(tp, "raw_grid", 24)
        scale = getattr(tp, "scale_factor", 1)
    else:
        raw_grid = cfg.get("raw_grid", 24)
        scale = cfg.get("scale_factor", 1)
    gs = raw_grid // scale
    return gs, gs  # H, W

from contextlib import contextmanager

@contextmanager
def temporary_cast_model_dtype(model, dtype=torch.float32):
    """
    Temporarily cast all params/buffers to `dtype`, then restore originals.
    Keeps device the same. Safe for one forward/backward.
    """
    # capture original dtypes
    orig_dtypes = {}
    for name, p in model.named_parameters(recurse=True):
        orig_dtypes[name] = p.dtype
    for name, b in model.named_buffers(recurse=True):
        orig_dtypes[f"__buf__{name}"] = b.dtype

    try:
        model.to(dtype=dtype)
        yield
    finally:
        # restore
        for name, p in model.named_parameters(recurse=True):
            want = orig_dtypes[name]
            if p.dtype != want:
                p.data = p.data.to(want)
        for name, b in model.named_buffers(recurse=True):
            want = orig_dtypes[f"__buf__{name}"]
            if b.dtype != want:
                b.data = b.data.to(want)


def _normalize_cam(cam):
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / (cam.max() + 1e-8)
    return cam


def _save_cam_overlay(rgb_image_pil, cam_2d, out_overlay="gradcam_overlay.png", out_raw="gradcam_raw.png"):
    img = np.array(rgb_image_pil).astype(np.float32) / 255.0
    H, W = img.shape[:2]
    cam = cam_2d
    if cam.shape != (H, W):
        cam = F.interpolate(torch.from_numpy(cam)[None, None].float(),
                            size=(H, W), mode='bilinear', align_corners=False)[0,0].numpy()
    cam = _normalize_cam(cam)

    # raw heatmap
    plt.figure()
    plt.imshow(cam, cmap='jet'); plt.axis('off'); plt.tight_layout(pad=0)
    plt.savefig(out_raw, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()

    # overlay
    cm = plt.get_cmap('jet')
    heat = cm(cam)[..., :3]
    overlay = (0.45 * heat + 0.55 * img)
    overlay = (overlay * 255).clip(0,255).astype(np.uint8)

    plt.figure()
    plt.imshow(overlay); plt.axis('off'); plt.tight_layout(pad=0)
    plt.savefig(out_overlay, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()


@torch.no_grad()
def _first_non_special(gen_ids, tokenizer):
    bad = set()
    for name in ("eos_token_id", "pad_token_id", "bos_token_id"):
        tid = getattr(tokenizer, name, None)
        if tid is not None:
            bad.add(int(tid))
    for t in gen_ids.tolist():
        if t not in bad:
            return int(t)
    return int(gen_ids[0].item()) if gen_ids.numel() > 0 else None

def _resolve_projector_module(model):
    """
    Return (module, dotted_name) for the projector.
    Tries common LLaVA placements, then searches by suffix.
    """
    # 1) common nesting: model.model.mm_projector
    try:
        core = getattr(model, "model", None) or getattr(model, "get_model")()
    except Exception:
        core = getattr(model, "model", None)
    cand = None
    name = None
    if core is not None and hasattr(core, "mm_projector"):
        cand = core.mm_projector
        name = "model.mm_projector"
    elif hasattr(model, "mm_projector"):
        cand = model.mm_projector
        name = "mm_projector"
    else:
        # 2) search by name
        for n, m in model.named_modules():
            if n.endswith("mm_projector"):
                cand, name = m, n
                break
    return cand, (name or "<not found>")


class _ProjTapPostCompat:
    """
    Forward hook: capture projector OUTPUT (packed tokens).
    Compatible with PyTorch versions that don't support with_kwargs.
    """
    def __init__(self, module):
        self.feats = None
        self._hits = 0
        # No with_kwargs; standard 3-arg signature
        self.h = module.register_forward_hook(self._post)

    def _select_tensor(self, output):
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
            return output[0]
        if isinstance(output, dict):
            # try common keys
            for k in ("x", "out", "hidden_states", "features"):
                if k in output and isinstance(output[k], torch.Tensor):
                    return output[k]
        raise RuntimeError("Unexpected projector output type for Grad-CAM.")

    def _post(self, module, args, output):
        out = self._select_tensor(output)
        # replace with a detached, grad-enabled copy so we can get grads here
        out2 = out.detach().requires_grad_(True)
        self.feats = out2
        self._hits += 1
        return out2

    @property
    def grads(self):
        return None if (self.feats is None) else self.feats.grad

    def close(self):
        self.h.remove()
def packed_grid_hw_from_feats_or_model(model, P: int):
    # Prefer the *actual* packed token count from the hook
    H_det, W_det = infer_grid_hw(P)

    # Try to read attributes for sanity-check only
    tp = getattr(getattr(model, "model", model), "mm_projector", None) or getattr(model, "mm_projector", None)
    raw = getattr(tp, "raw_grid", None)
    scale = getattr(tp, "scale_factor", None)
    if isinstance(raw, int) and isinstance(scale, int) and raw > 0 and scale > 0 and raw % scale == 0:
        gs = raw // scale
        if gs * gs != P:
            print(f"[Grad-CAM][warn] Packed tokens P={P} (≈{H_det}x{W_det}) "
                  f"disagree with attrs raw_grid={raw}, scale_factor={scale} (=> {gs}x{gs}). "
                  f"Using detected {H_det}x{W_det}.")
        else:
            H_det, W_det = gs, gs
    else:
        print(f"[Grad-CAM][note] Using detected grid {H_det}x{W_det} from P={P} (attrs missing or invalid).")
    return H_det, W_det

def gradcam_first_token_overlay(model, tokenizer, image_tensor, prompt_input_ids, first_gen_token_id, rgb_image_pil,
                                out_overlay="gradcam_overlay.png", out_raw="gradcam_raw.png"):
    device = next(model.parameters()).device
    model.eval()
    model.zero_grad(set_to_none=True)

    # ---- robust projector resolution ----
    mm_proj, proj_name = _resolve_projector_module(model)
    if mm_proj is None:
        raise RuntimeError("Could not locate mm_projector on the model.")
    print(f"[Grad-CAM] Hooking projector at: {proj_name} ({mm_proj.__class__.__name__})")

    with torch.enable_grad():
        with temporary_cast_model_dtype(model, torch.float32):
            tap = _ProjTapPostCompat(mm_proj)  # robust forward hook (no with_kwargs)

            inputs = prompt_input_ids.to(device)
            attention_mask = None
            position_ids = None

            # IMPORTANT: install the hook BEFORE this call, because the projector is used inside it
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = \
                model.prepare_inputs_labels_for_multimodal(
                    inputs, position_ids, attention_mask, None, None,
                    images=image_tensor.to(device=device, dtype=torch.float32, non_blocking=True),
                    image_sizes=None
                )

            # One forward pass (no cache) so grads flow to the hooked tensor
            out = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=False,
            )
            logits = out.logits
            score = logits[0, -1, first_gen_token_id]
            score.backward()  # retain_graph not needed

            feats = tap.feats
            grads = tap.grads
            hits  = tap._hits
            tap.close()

    if hits == 0:
        raise RuntimeError("Grad-CAM projector hook failed (hook did not fire).")
    if feats is None or grads is None:
        raise RuntimeError("Grad-CAM projector hook failed (no feats/grads).")

    print(f"[Grad-CAM] Hook hits: {hits}, feats shape: {tuple(feats.shape)}")

    # No CLS handling here; projector output is packed tokens already.
    B, P, C = feats.shape
    weights = grads.mean(dim=1)                             # [B, C]
    cam_tokens = torch.relu((feats * weights[:,None,:]).sum(dim=2))  # [B, P]

    Hc, Wc = packed_grid_hw_from_feats_or_model(model, P)   # <-- change here
    cam_grid = cam_tokens[0].reshape(Hc, Wc).detach().cpu().numpy()

    # Save exact token grid (no upsample ambiguity)
    save_token_grid_image(cam_tokens[0], out_png="gradcam_token_grid.png", draw_grid=True)
    # NEAREST overlay to keep blocks crisp
    save_overlay_nearest(rgb_image_pil, cam_grid, out_png="gradcam_overlay_nearest.png")
    # Optional bilinear overlay
    _save_cam_overlay(rgb_image_pil, cam_grid, out_overlay="gradcam_overlay_bilinear.png", out_raw="gradcam_raw.png")

    return "gradcam_overlay_nearest.png", "gradcam_raw.png"

# =================== End Grad-CAM helpers ===================

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image
def main(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length = 2048,
        padding_side="right",
        use_fast = True
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
    preprocess = Compose([ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711))])

    
    while True:
        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        image = load_image(args.image_file)


        image_tensor = process_images([image], image_processor, model.config)[0]
        image_tensor = image_tensor.unsqueeze(0)

        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break
        # inp = "what is in the image?"

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

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        mode = model.config.image_aspect_ratio
        with torch.inference_mode():
            start = time.time()

            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.bfloat16, device='cuda', non_blocking=True),
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        # ---------------- Grad-CAM (answer-conditioned) ----------------
        # After generate(...)
        seq = output_ids[0]  # shape: (len_seq_or_len_new,)
        prompt_len = input_ids.shape[1]

        # Robustly extract the generated tokens:
        if seq.size(0) >= prompt_len and torch.equal(seq[:prompt_len], input_ids[0]):
            gen_ids = seq[prompt_len:]        # case A: output = [prompt || new]
        else:
            gen_ids = seq                     # case B: output = [new] only (common with TextStreamer)

        # Sanity: decode like before (just for your own check)
        # print("GEN TEXT:", tokenizer.decode(gen_ids, skip_special_tokens=True))

        try:
            gen_ids = output_ids[0, input_ids.shape[1]:]
            if gen_ids.numel() == 0:
                print("\n[Grad-CAM] Skipped (no generated tokens).")
            else:
                first_id = _first_non_special(gen_ids, tokenizer)
                if first_id is None:
                    print("\n[Grad-CAM] Skipped (no usable token).")
                else:
                    overlay_path, raw_path = gradcam_first_token_overlay(
                        model=model,
                        tokenizer=tokenizer,
                        image_tensor=image_tensor,                    # same tensor you fed to generate()
                        prompt_input_ids=input_ids,                   # same prompt ids
                        first_gen_token_id=first_id,
                        rgb_image_pil=load_image(args.image_file),
                        out_overlay="gradcam_overlay.png",
                        out_raw="gradcam_raw.png",
                    )
                    print(f"\n[Grad-CAM] Saved overlay: {overlay_path} | raw: {raw_path}")
        except Exception as e:
            print(f"\n[Grad-CAM] Failed: {e}")
        # ---------------------------------------------------------------

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        end = time.time()
        print("***time: ", end-start)
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="path/to/tokenpacker")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default='vicuna_v1')
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-file", type=str, required=True)
    args = parser.parse_args()
    main(args)