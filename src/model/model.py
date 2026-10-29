# Fill in `...` with the necessary paths

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as spnorm
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT / "src"))
from model.vae import CompressionEncoder, CompressionDecoder, rotate
from dataset.dataset import NVPhysicalAIWindows, prepare_batch
from torch.utils.checkpoint import checkpoint

DATASET_ROOT = Path("...")
ENCODER_CHECKPOINT = Path("...")
DECODER_CHECKPOINT = ENCODER_CHECKPOINT

def rhalf(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def axis_cos_sin(pos, dim, base):
    i = torch.arange(dim // 2, device=pos.device, dtype=torch.float32)
    freqs = 1.0 / (base ** (2 * i / dim))
    angles = torch.outer(pos, freqs)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()

def rope_1d(q, k, base=100.0):
    """1D RoPE along the sequence dim. q, k: (..., T, head_dim)."""
    T, d = q.shape[-2], q.shape[-1]
    pos = torch.arange(T, device=q.device, dtype=torch.float32)
    cos, sin = axis_cos_sin(pos, d, base)

    def apply(x):
        xf = x.float()
        return (xf * cos + rhalf(xf) * sin).to(x.dtype)

    return apply(q), apply(k)

def rope_nd(q, k, positions, dims, base=100.0):
    tables = [axis_cos_sin(p, d, base) for p, d in zip(positions, dims)]

    def apply(x):
        xf = x.float()
        parts = xf.split(dims, dim=-1)
        out = [p * c + rhalf(p) * s for p, (c, s) in zip(parts, tables)]
        return torch.cat(out, dim=-1).to(x.dtype)

    return apply(q), apply(k)



def chunked_sdpa(q, k, v, chunk=16):
    outs = []
    for i in range(0, q.shape[0], chunk):
        qi, ki, vi = q[i:i+chunk], k[i:i+chunk], v[i:i+chunk]
        if torch.is_grad_enabled():
            outs.append(checkpoint(F.scaled_dot_product_attention, qi, ki, vi, use_reentrant=False))
        else:
            outs.append(F.scaled_dot_product_attention(qi, ki, vi))
    return torch.cat(outs)


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, num_heads=2, head_dim=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(in_channels, 3 * num_heads * head_dim)
        self.proj = nn.Linear(num_heads * head_dim, in_channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x):
        # x: (B, T, C, H, W) - performing attention on H, W
        B, T, C, H, W = x.shape
        x = x.reshape(-1, C, H, W)
        q, k, v = self.qkv(x.permute(0,2,3,1)).permute(3,0,1,2).reshape(3, self.num_heads, self.head_dim, B*T, -1).permute(0, 3, 1, 4, 2).contiguous().unbind(0)
        q, k = rotate(q, k, H, W)
        # spattn = chunked_sdpa(q, k, v, chunk=4) # (B*T, num_heads, H*W, head_dim)
        spattn = F.scaled_dot_product_attention(q, k, v, is_causal=False) # (B*T, num_heads, H*W, head_dim)
        spattn = spattn.permute(0,1,3,2).reshape(B*T, -1, H, W)
        spattn = self.proj(spattn.permute(0,2,3,1)).permute(0,3,1,2) # (B*T, in_channels, H, W)
        x = x + spattn
        x = x.reshape(B, T, C, H, W)
        return x


class TemporalAttention(nn.Module):
    def __init__(self, in_channels, res, num_heads=2, head_dim=32, res_tokens=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.res_tokens = res_tokens
        self.res = res
        self.in_channels = in_channels
        token_dim = (res // res_tokens) ** 2 * in_channels
        self.qkv = nn.Linear(token_dim, 3 * num_heads * head_dim)
        self.proj = nn.Linear(num_heads * head_dim, token_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x):
        # x: (B, T, C, H, W) - performing attention on T
        B, T, C, H, W = x.shape
        # = x.reshape(B, T, C, self.res_tokens, H//self.res_tokens, self.res_tokens, W//self.res_tokens).permute(0,1,3,5,2,4,6).reshape(B, T*self.res_tokens*self.res_tokens, -1)
        q,k,v = self.qkv(x.reshape(-1, C, H, W).reshape(B, T, C, self.res_tokens, H//self.res_tokens, self.res_tokens, W//self.res_tokens).permute(0,1,3,5,2,4,6).reshape(B, T*self.res_tokens*self.res_tokens, -1)).permute(2,0,1).reshape(3, self.num_heads, self.head_dim, B, T*self.res_tokens*self.res_tokens).permute(0, 3, 1, 4, 2).contiguous().unbind(0)
        r = self.res_tokens
        frame = torch.arange(T, device=x.device).repeat_interleave(r * r)
        row = torch.arange(r, device=x.device).repeat_interleave(r).repeat(T)
        col = torch.arange(r, device=x.device).repeat(r * T)
        d = self.head_dim
        q, k = rope_nd(q, k, [frame.float(), row.float(), col.float()], [d // 2, d // 4, d // 4])
        mask = frame[None, :] <= frame[:, None] 
        if torch.is_grad_enabled():
            temp_attn = checkpoint(F.scaled_dot_product_attention, q, k, v, mask, use_reentrant=False)
        else:
            temp_attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        temp_attn = temp_attn.permute(0, 2, 1, 3).reshape(B, T*self.res_tokens*self.res_tokens, -1)
        x = x + self.proj(temp_attn).reshape(B, T, self.res_tokens, self.res_tokens, C, H//self.res_tokens, W//self.res_tokens).permute(0,1,4,2,5,3,6).reshape(B,T,C,H,W)
        return x
        

class FutureModulation(nn.Module):
    def __init__(self, in_channels, act_dim=4):
        super().__init__()
        self.in_channels = in_channels
        self.act_proj = nn.Sequential(
            nn.Linear(act_dim + 1, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels*2)
        )
        self.norm = nn.GroupNorm(in_channels//2, in_channels)
        self.noise = nn.Parameter(torch.randn(in_channels))
        self.noise_mod = nn.Linear(in_channels, in_channels)
        self.noise_latent_weight = nn.Parameter(torch.randn(1))

    def forward(self, x, a, sigma=None):
        # x: [B,T,C,H,W]
        # a: [B,T,A]
        B, T, C, H, W = x.shape
        noise = self.noise.repeat(B, T, 1)
        noise = noise + torch.randn_like(noise) * torch.sigmoid(self.noise_latent_weight)
        noise = self.noise_mod(noise)
        if sigma is None:
            sigma = torch.zeros(B, T, 1, device=x.device, dtype=x.dtype)
        elif sigma.ndim == 2:
            sigma = sigma.unsqueeze(-1)
        gamma, beta = self.act_proj(torch.cat((a.float(), sigma.float()), dim=-1)).chunk(2, dim=-1)
        out = (self.norm(x.reshape(-1, C, H, W)).reshape(B, T, C, H, W) + noise[..., None, None]) 
        out = out * (1 + gamma[..., None, None]) + beta[..., None, None]
        return out

class ActionFrameAttention(nn.Module):
    def __init__(self, in_channels, res, act_dim=4, num_heads=2, head_dim=32, res_tokens=4):
        super().__init__()
        self.act_dim = act_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.res_tokens = res_tokens
        self.res = res
        self.in_channels = in_channels
        self.act_proj = nn.Sequential(
            nn.Linear(act_dim, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, 2*in_channels)
        )
        nn.init.zeros_(self.act_proj[-1].weight)        
        nn.init.zeros_(self.act_proj[-1].bias)
        self.x_s = SpatialAttention(in_channels=in_channels)
        self.x_t = TemporalAttention(in_channels=in_channels, res=self.res)
        self.modulation = FutureModulation(in_channels=in_channels, act_dim=act_dim)
    def forward(self, x, a, sigma=None):
        B, T, C, H, W = x.shape
        _,_,A = a.shape
        mat = self.modulation(x, a, sigma) # (B, T, C, H, W)
        mat = self.x_s(mat) # (B, T, C, H, W)
        mat = self.x_t(mat) # (B, T, C, H, W)
        return mat #


class ConvBackbone(nn.Module):
    def __init__(self, in_channels, out_channels, res, act_dim=4, num_heads=2, head_dim=32, res_tokens=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.res = res
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.attn = ActionFrameAttention(in_channels=in_channels, res=res, act_dim=act_dim, num_heads=num_heads, head_dim=head_dim, res_tokens=res_tokens)
    
    def forward(self, x, a, sigma=None):
        B, T, C, H, W = x.shape
        x = x.reshape(-1, C, H, W)
        x = self.attn(x.reshape(B, T, C, H, W), a, sigma) + x.reshape(B, T, C, H, W)
        x = self.conv(x.reshape(-1, C, H, W)).reshape(B, T, -1, H, W)
        return x




class Anna(nn.Module):
    def __init__(self, in_channels=8, out_channels=3, num_heads=2, head_dim=32, res_tokens=8, act_dim=4):
        super().__init__()
        self.latent_res = 32
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.module_list = nn.ModuleList()

        self.conv1 = ConvBackbone(in_channels=in_channels, out_channels=32, res=32)
        self.conv2 = ConvBackbone(in_channels=32, out_channels=128, res=32)
        self.conv3 = ConvBackbone(in_channels=128, out_channels=64, res=32)
        self.conv4 = ConvBackbone(in_channels=64, out_channels=8, res=32)
        self.module_list.append(self.conv1)
        self.module_list.append(self.conv2)
        self.module_list.append(self.conv3)
        self.module_list.append(self.conv4)
    
    def forward(self, x, a, sigma=None):
        for module in self.module_list:
            x = module(x, a, sigma)
        return x




