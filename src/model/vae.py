import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path
from torch.nn.utils import spectral_norm as spnorm
from dataset.dataset import NVPhysicalAIWindows, prepare_batch

def rhalf(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def axis_cos_sin(pos, dim, base):
    i = torch.arange(dim // 2, device=pos.device, dtype=torch.float32)
    freqs = 1.0 / (base ** (2 * i / dim))
    angles = torch.outer(pos, freqs)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()

def rotate(q, k, h, w, base=100.0):
    # q, k: (B*T, num_heads, h*w, head_dim); head_dim divisible by 4
    d, dev = q.shape[-1], q.device
    rows = torch.arange(h, device=dev, dtype=torch.float32).repeat_interleave(w)
    cols = torch.arange(w, device=dev, dtype=torch.float32).repeat(h)
    cos_r, sin_r = axis_cos_sin(rows, d // 2, base)
    cos_c, sin_c = axis_cos_sin(cols, d // 2, base)

    def apply(x):
        xf = x.float()
        xr, xc = xf.chunk(2, dim=-1)                    # rows half, cols half
        xr = xr * cos_r + rhalf(xr) * sin_r
        xc = xc * cos_c + rhalf(xc) * sin_c
        return torch.cat([xr, xc], dim=-1).to(x.dtype)

    return apply(q), apply(k)



class CompressionEncoder(nn.Module):
    def __init__(self, in_channels=3, dk=16, num_heads=8, latent_channels=8):
        super().__init__()
        self.dk = dk
        self.num_heads = num_heads
        self.downsample_8x = nn.Sequential(
            spnorm(nn.Conv2d(in_channels, num_heads*dk//4, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spnorm(nn.Conv2d(num_heads*dk//4, num_heads*dk//2, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spnorm(nn.Conv2d(num_heads*dk//2, num_heads*dk, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.resid_proj = nn.Conv2d(dk*num_heads, latent_channels, kernel_size=1, stride=1, padding=0)
        self.qkv = nn.Linear(dk*num_heads, 3 * dk*num_heads)
        self.proj = nn.Linear(dk*num_heads, latent_channels)
        self.norm = nn.GroupNorm(num_groups=latent_channels//2, num_channels=latent_channels, eps=1e-6, affine=True)
    def forward(self, x):
        "x: (B,T,C,H,W)"
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W) # (B*T, C, H, W)
        x = self.downsample_8x(x)
        _, _, h, w = x.shape
        q,k,v = self.qkv(x.permute(0,2,3,1)).reshape(B*T, h*w, 3, self.num_heads, self.dk).permute(2,0,3,1,4).contiguous().unbind(0)
        q, k = rotate(q, k, h, w)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn = self.proj(attn.permute(0,2,1,3).reshape(B*T, h*w, -1)).permute(0,2,1).reshape(B, T, -1, h, w)
        attn = self.norm(attn.view(B*T, -1, h, w)).view(B, T, -1, h, w) + self.resid_proj(x).view(B, T, -1, h, w)
        return attn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return F.leaky_relu(x + self.block(x), 0.2)


class CompressionDecoder(nn.Module):
    def __init__(self, in_channels=8, out_channels=3, upscale_rat=8):
        super().__init__()
        self.upscale_num = np.log2(upscale_rat)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.in_proj = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.res1 = ResidualBlock(32)
        self.up1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.up2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.res2 = ResidualBlock(128)
        self.up3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.res3 = ResidualBlock(64)
        self.up4 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        # x: [B,T,8,32,32]
        B,T,_,_,_ = x.shape
        x = x.reshape(-1, self.in_channels, 32, 32)
        x = self.in_proj(x)
        x = self.res1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up1(x)
        x = F.leaky_relu(x, 0.2)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.res2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.res3(x)
        x = self.up4(x)
        return x.reshape(B,T,self.out_channels,x.shape[2],x.shape[3])


def main():
    frames_path = Path("...")
    seq_len = 32
    batch_size = 4
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    N = CompressionEncoder().to(device)
    D = CompressionDecoder().to(device)
    