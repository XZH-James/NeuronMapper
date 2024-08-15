"""
By introducing relative position bias, the self-attention mechanism can better capture the relative position information
in three-dimensional space, thus improving the perception ability of spatial structure.

SelfAttention:
Implementation of Multi-Head Self-Attention (MHSA) with relative position bias

Residual:
The Residual Connection is realized, that is, the input is directly added to the output after a specific operation
(such as self-attention or feedforward neural network), thereby alleviating the problem of disappearing gradients in
the deep network and promoting the effective dissemination of information

PreNorm & PreNormDrop:
LayerNorm normalization of the input data is performed before the main operation, such as self-attention or feedforward
networking, is performed

FeedForward:
A two-layer Feedforward Neural Network is implemented, typically used to handle the output of each attention head in
Transformer. The first layer extends the dimension to a higher hidden dimension, which is then activated and linearly
mapped back to the original dimension through the second layer

TransformerModel:
A complete Transformer model is constructed, including multi-layer residual connected self-attention module and
feedforward neural network.


"""

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_3tuple, trunc_normal_


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, window_size=4, qkv_bias=False, qk_scale=None, dropout_rate=0.0):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        # Window size for attention mechanism
        self.window_size = to_3tuple(window_size)  # Ensure window_size is a 3-tuple

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                        heads)
        )

        # Compute relative position index
        self.relative_position_index = self.compute_relative_position_index()

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

        # Initialize relative position bias table
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def compute_relative_position_index(self):
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = coords.flatten(1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # Shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        return relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)]
        relative_position_bias = relative_position_bias.reshape(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, window_size, dropout_rate=0.1, attn_dropout_rate=0.1):
        super().__init__()
        self.window_size = to_3tuple(window_size)
        layers = [
                     Residual(
                         PreNormDrop(
                             dim,
                             dropout_rate,
                             SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate,
                                           window_size=self.window_size),
                         )
                     ),
                     Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))),
                 ] * depth
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
