import torch
import torch.nn as nn
import re
from functools import partial
import numpy as np
from torch.nn.init import trunc_normal_
from torch.nn import functional as F
import math


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

import torch
import torch.nn as nn
from functools import partial
import numpy as np
from torch.nn.init import trunc_normal_
import torch.nn.functional as F
import math
from einops import rearrange

# --------------------------------------------------------------------------
# 1) Positional Embedding Utilities
#    - get_abs_pos() interpolates a 2D embedding to a new size
#    - get_2d_sincos_pos_embed() creates a base sinusoidal pos embed
# --------------------------------------------------------------------------
def get_abs_pos(abs_pos, tgt_size):
    """
    abs_pos: [L, C], a flattened 2D grid of embeddings (e.g. raw_grid^2, embed_dim)
    tgt_size: (H, W) to which we want to resize
    returns: [H*W, C], an interpolated 2D embedding
    """
    src_size = int(math.sqrt(abs_pos.size(0)))
    dtype = abs_pos.dtype

    # Reshape to (1, C, H, W) -> interpolate -> reshape back
    out = F.interpolate(
        abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
        size=(tgt_size[0], tgt_size[1]),
        mode="bicubic",
        align_corners=False,
    )
    out = out.permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)  # [H*W, C]
    return out

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int, for a grid of size (grid_size x grid_size)
    embed_dim: dimension of the embedding
    return:
      pos_embed: [grid_size*grid_size, embed_dim] or
                 [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w first, h second
    grid = np.stack(grid, axis=0)       # shape: [2, grid_size, grid_size]
    grid = grid.reshape([2, 1, grid_size, grid_size])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """ grid: [2, 1, H, W] """
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (H, W) -> flattened to (H*W,)
    out: (H*W, D)
    """
    assert embed_dim % 2 == 0
    pos = pos.reshape(-1)  # (H*W,)

    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= (embed_dim / 2.)
    omega = 1. / (10000 ** omega)  # shape: [D/2]

    # outer product -> [M, D/2]
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------------------------
# 2) The Delta_LLaVA Module
#    Includes:
#    - A buffer for raw-grid sincos pos_embed
#    - Interpolation of pos_embed to grid_size x grid_size
#    - Injection of position into the queries
# --------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
   
from timm.models.layers import DropPath, trunc_normal_

def merge_pre_bn(module, pre_bn_1, pre_bn_2=None):
    """ Merge pre BN to reduce inference runtime.
    """
    weight = module.weight.data
    if module.bias is None:
        zeros = torch.zeros(module.out_channels, device=weight.device).type(weight.type())
        module.bias = nn.Parameter(zeros)
    bias = module.bias.data
    if pre_bn_2 is None:
        assert pre_bn_1.track_running_stats is True, "Unsupport bn_module.track_running_stats is False"
        assert pre_bn_1.affine is True, "Unsupport bn_module.affine is False"

        scale_invstd = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        extra_weight = scale_invstd * pre_bn_1.weight
        extra_bias = pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd
    else:
        assert pre_bn_1.track_running_stats is True, "Unsupport bn_module.track_running_stats is False"
        assert pre_bn_1.affine is True, "Unsupport bn_module.affine is False"

        assert pre_bn_2.track_running_stats is True, "Unsupport bn_module.track_running_stats is False"
        assert pre_bn_2.affine is True, "Unsupport bn_module.affine is False"

        scale_invstd_1 = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        scale_invstd_2 = pre_bn_2.running_var.add(pre_bn_2.eps).pow(-0.5)

        extra_weight = scale_invstd_1 * pre_bn_1.weight * scale_invstd_2 * pre_bn_2.weight
        extra_bias = scale_invstd_2 * pre_bn_2.weight *(pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd_1 - pre_bn_2.running_mean) + pre_bn_2.bias

    if isinstance(module, nn.Linear):
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
    elif isinstance(module, nn.Conv2d):
        assert weight.shape[2] == 1 and weight.shape[3] == 1
        weight = weight.reshape(weight.shape[0], weight.shape[1])
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
        weight = weight.reshape(weight.shape[0], weight.shape[1], 1, 1)
    bias.add_(extra_bias)

    module.weight.data = weight
    module.bias.data = bias
class MHCA(nn.Module):
    """
    Multi-Head Convolutional Attention with LayerNorm
    """
    def __init__(self, out_channels, head_dim):
        super(MHCA, self).__init__()
        self.group_conv3x3 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_channels // head_dim,
            bias=False
        )

        self.norm = nn.LayerNorm(out_channels)  # LayerNorm replaces BatchNorm
        self.act = nn.ReLU(inplace=True)
        self.projection = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = x.to(self.group_conv3x3.weight.dtype)
        out = self.group_conv3x3(x)

        # Apply LayerNorm over channel dim
        out = out.permute(0, 2, 3, 1)  # (B, C, H, W) → (B, H, W, C)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)

        out = self.act(out)
        out = self.projection(out)
        return out

NORM_EPS=1e-5


class PatchEmbed(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(PatchEmbed, self).__init__()
        # norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        if stride == 2:
            self.avgpool = nn.AvgPool2d((2, 2), stride=2, ceil_mode=True, count_include_pad=False)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            # self.norm = norm_layer(out_channels)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            # self.norm = norm_layer(out_channels)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            # self.norm = nn.Identity()

    def forward(self, x):
        return self.conv(self.avgpool(x))
 
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class Mlp(nn.Module):
    def __init__(self, in_features, out_features=None, mlp_ratio=None, drop=0., bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_dim = _make_divisible(in_features * mlp_ratio, 32)
        self.conv1 = nn.Conv2d(in_features, hidden_dim, kernel_size=1, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, out_features, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def merge_bn(self, pre_norm):
        merge_pre_bn(self.conv1, pre_norm)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x
    
import torch
import torch.nn as nn

class E_MHSA(nn.Module):
    """
    Efficient Multi-Head Self Attention with LayerNorm for stability.
    """

    def __init__(
        self,
        dim,
        out_dim=None,
        head_dim=32,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim**-0.5

        self.norm = nn.LayerNorm(dim)  # Added LayerNorm

        self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.out_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.N_ratio = sr_ratio**2
        if sr_ratio > 1:
            self.sr = nn.AvgPool1d(kernel_size=self.N_ratio, stride=self.N_ratio)

    def forward(self, x):
        B, N, C = x.shape

        x = self.norm(x)  # Normalize input before projection

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2)
            if x_.shape[-1] < self.N_ratio:
                print("⚠️ SR ratio too high for sequence length")
                x_ = x.transpose(1, 2)  # fallback
            else:
                x_ = self.sr(x_)
            x_ = x_.transpose(1, 2)

            k = self.k(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
            v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            k = self.k(x).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
            v = self.v(x).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k) * self.scale

        if not torch.isfinite(attn).all():
            print("⚠️ Non-finite attention detected in E_MHSA")
            attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

        attn = attn - attn.max(dim=-1, keepdim=True).values  # stable softmax trick
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class NTB(nn.Module):
    """
    Next Transformer Block
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        path_dropout,
        stride=1,
        sr_ratio=1,
        mlp_ratio=2,
        head_dim=32,
        mix_block_ratio=0.75,
        attn_drop=0,
        drop=0,
    ):
        super(NTB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mix_block_ratio = mix_block_ratio
        # norm_func = partial(nn.BatchNorm2d, eps=NORM_EPS)

        self.mhsa_out_channels = _make_divisible(
            int(out_channels * mix_block_ratio), 32
        )
        self.mhca_out_channels = out_channels - self.mhsa_out_channels

        self.patch_embed = PatchEmbed(in_channels, self.mhsa_out_channels, stride)
        # self.norm1 = norm_func(self.mhsa_out_channels)
        self.e_mhsa = E_MHSA(
            self.mhsa_out_channels,
            head_dim=head_dim,
            sr_ratio=sr_ratio,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.mhsa_path_dropout = DropPath(path_dropout * mix_block_ratio)

        self.projection = PatchEmbed(
            self.mhsa_out_channels, self.mhca_out_channels, stride=1
        )
        self.mhca = MHCA(self.mhca_out_channels, head_dim=head_dim)
        self.mhca_path_dropout = DropPath(path_dropout * (1 - mix_block_ratio))

        # self.norm2 = norm_func(out_channels)
        self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop)
        self.mlp_path_dropout = DropPath(path_dropout)
        self.norm_mhsa = nn.LayerNorm(self.mhsa_out_channels)


        self.is_bn_merged = False

    def merge_bn(self):
        if not self.is_bn_merged:
            # self.e_mhsa.merge_bn(self.norm1)
            self.mlp.merge_bn(self.norm2)
            self.is_bn_merged = True

    def forward(self, x):
        assert x.dim() == 4, f"NTB expects 4D input, got shape {x.shape}"
        x = self.patch_embed(x)
        B, C, H, W = x.shape

        out = rearrange(x, "b c h w -> b (h w) c")
        out = self.norm_mhsa(out)  # 🔧 This stabilizes q/k/v
        out = self.e_mhsa(out)


        out = self.mhsa_path_dropout(out)
        x = x + rearrange(out, "b (h w) c -> b c h w", h=H)

        out = self.projection(x)
        out = out + self.mhca_path_dropout(self.mhca(out))
        out = F.interpolate(out, size=x.shape[2:], mode='nearest')

        x = torch.cat([x, out], dim=1)

        x = x + self.mlp_path_dropout(self.mlp(x))
        return x


import torch
import torch.nn as nn

class NCBAndNTBBlock(nn.Module):
    """
    A single block that first applies NCB, then NTB in sequence.
    """
    def __init__(
        self,
        in_channels,
        mid_channels,      # output of NCB
        out_channels,      # output of NTB
        stride_ncb=2,
        stride_ntb=1,
        path_dropout=0.1,
        drop=0.0,
        head_dim=32,
        mlp_ratio_ncb=3,   # for NCB
        mlp_ratio_ntb=2,   # for NTB
        sr_ratio=1,        # for NTB
        mix_block_ratio=0.75 # for NTB
    ):
        super().__init__()


        # 2) NTB block
        self.ntb = NTB(
            in_channels=mid_channels,
            out_channels=out_channels,
            path_dropout=path_dropout,
            stride=stride_ntb,
            sr_ratio=sr_ratio,
            mlp_ratio=mlp_ratio_ntb,
            head_dim=head_dim,
            mix_block_ratio=mix_block_ratio,
            attn_drop=0.0,
            drop=drop
        )

    def forward(self, x):
        # x shape: [B, in_channels, H, W]
        x = self.ntb(x)  # [B, out_channels, H_out_ntb, W_out_ntb]
        return x

class DeltaProjection(nn.Module):
    """ DeltaLLM-style low-rank weight sharing projection layer. """

    def __init__(self, in_features, out_features, rank=16):
        super().__init__()
        self.shared_weight = nn.Linear(in_features, out_features, bias=False)  # Shared weight matrix W_s
        
        # Low-rank delta matrices (per-layer updates)
        self.U = nn.Parameter(torch.randn(in_features, rank))  # U matrix (in_features x rank)
        self.V = nn.Parameter(torch.randn(rank, out_features))  # V matrix (rank x out_features)

        # Ensure small initialization
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)

        # Additional non-linearity and normalization
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(out_features, eps=1e-6)

    def forward(self, x):
        """ Computes the projection with low-rank update. """
        delta_w = self.U @ self.V  # Shape: (in_features, out_features)
        delta_w = delta_w.T  # Transpose to match shared_weight shape: (out_features, in_features)
        
        proj_weight = self.shared_weight.weight + delta_w  # Compute effective weight W = W_s + ΔW
        device = x.device  # usually CUDA:0
        proj_weight = proj_weight.to(device)

        x = F.linear(x, proj_weight)  # Apply projection
        x = self.activation(x)
        x = self.norm(x)
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class Delta_LLaVA(nn.Module):
    def __init__(
            self,
            scale_factor,
            raw_grid=24,
            embed_dim=1024,
            num_heads=1024//128,
            kv_dim=1024,
            hidden_size=4096,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
        if raw_grid % scale_factor != 0:
            raise ValueError("scale_factor must divide raw_grid")

        self.raw_grid = raw_grid
        self.stride_ncb=1
        self.stride_ntb=1
        self.scale_factor = scale_factor * self.stride_ncb * self.stride_ntb
        self.grid_size = raw_grid // self.scale_factor
        self.num_queries = self.grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # ---------------------------
        # Q, K, V projections
        # ---------------------------
        self.q_proj_1 = DeltaProjection(kv_dim, embed_dim, rank=16)
        self.k_proj_1 = DeltaProjection(4096, 1024, rank=16)
        self.v_proj_1 = DeltaProjection(4096, 1024, rank=16)

        # ---------------------------
        # Multi-Head Attention
        # ---------------------------
        self.clip_attn = nn.MultiheadAttention(embed_dim, num_heads)

        # ---------------------------
        # MLP
        # ---------------------------
        modules = [nn.Linear(1024, hidden_size)]
        for _ in range(1, 2):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        self.mlp = nn.Sequential(*modules)

        # ---------------------------
        # 2D Positional Embedding
        # ---------------------------
        pos_embed_np = get_2d_sincos_pos_embed(embed_dim, raw_grid)
        self.register_buffer(
            "pos_embed",
            torch.from_numpy(pos_embed_np).float(),
            persistent=False
        )

        # ---------------------------
        # Combined NCB + NTB block
        # ---------------------------
        self.combined_block = NCBAndNTBBlock(
            in_channels=embed_dim,   # same as the "channel" dimension for q
            mid_channels=embed_dim,  # you could vary this if you want
            out_channels=embed_dim,  # keep the same final channel dimension
            stride_ncb=self.stride_ncb,
            stride_ntb=self.stride_ncb,
            path_dropout=0.1,
            drop=0.0,
            head_dim=32,
            mlp_ratio_ncb=3,
            mlp_ratio_ntb=2,
            sr_ratio=1,
            mix_block_ratio=0.75
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def divide_feature(self, x, kernel_size, token_num, N, c):
        """
        x: [token_num, N, c]
        """
        h = w = int(token_num**0.5)

        reshape_x = x.reshape(h, w, N, c).reshape(h//kernel_size, kernel_size, w, N, c)
        reshape_x = reshape_x.permute(0, 2, 1, 3, 4)
        reshape_x = reshape_x.reshape(h//kernel_size, w//kernel_size, kernel_size, kernel_size, N, c)
        reshape_x = reshape_x.permute(0, 1, 3, 2, 4, 5).reshape(h//kernel_size, w//kernel_size, kernel_size*kernel_size, N, c)
        reshape_x = reshape_x.permute(2, 0, 1, 3, 4).reshape(kernel_size*kernel_size, -1, c)

        return reshape_x

    def forward(self, x, attn_mask=None):
        # x is a tuple: (x_single_level, x_multi_level)
        x_multi = x[1]  # multi-level features for K, V
        x_single = x[0] # single-level features for Q

        # 1) Project multi-level features -> K, V
        key = self.k_proj_1(x_multi).permute(1, 0, 2)
        value = self.v_proj_1(x_multi).permute(1, 0, 2)
        token_num, N, c = key.shape

        # 2) Interpolate x_single -> [B, embed_dim, grid_size, grid_size]
        q = x_single.reshape(x_single.shape[0], self.raw_grid, self.raw_grid, -1)
        q = q.permute(0, 3, 1, 2)  
        q = F.interpolate(q, size=(self.grid_size, self.grid_size), mode='bilinear')

        # 3) Pass through combined NCB+NTB
        q = self.combined_block(q)  # => [B, embed_dim, grid_size, grid_size]

        # 4) Flatten to [B, grid_size^2, embed_dim]
        q = q.permute(0, 2, 3, 1)
        q = q.reshape(q.size(0), -1, q.size(-1)) 

        # 5) Add pos embed
        pos_embed = self.pos_embed.to(q.device)
        pos_embed_2d   = get_abs_pos(pos_embed, (self.grid_size, self.grid_size))
        # print("q shape:", q.shape)
        # print("pos_embed_2d shape:", pos_embed_2d.shape)

        q = q + pos_embed_2d.unsqueeze(0)

        # 6) Final Q projection
        q = q.to(self.q_proj_1.shared_weight.weight.dtype)
        query = self.q_proj_1(q).permute(1, 0, 2)  # [grid_size^2, B, embed_dim]

        # 7) Local-chunk attention
        reshape_query = self.divide_feature(query, 1, self.num_queries, N, c)
        reshape_key   = self.divide_feature(key, self.scale_factor, token_num, N, c)
        reshape_value = self.divide_feature(value, self.scale_factor, token_num, N, value.shape[-1])

        out = self.clip_attn(reshape_query, reshape_key, reshape_value, attn_mask=attn_mask)[0]
        x_out = out.reshape(self.num_queries, N, -1).permute(1, 0, 2)  # [B, grid_size^2, embed_dim]

        # 8) Final MLP
        x_out = self.mlp(x_out)
        # print(x_out.shape)
        return x_out

def build_vision_projector(config):
    return Delta_LLaVA(hidden_size=config.hidden_size, scale_factor=config.scale_factor)