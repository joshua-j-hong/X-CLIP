# modules/adapter_rap.py
import torch
from torch import nn
import torch.nn.functional as F

def straight_through_topk_mask(scores: torch.Tensor, k: int):
    if k <= 0 or k >= scores.size(1):
        return scores.new_ones(scores.size()) if k >= scores.size(1) else scores.new_zeros(scores.size())
    topk_vals, topk_idx = torch.topk(scores, k, dim=1)
    hard = scores.new_zeros(scores.size())
    hard.scatter_(1, topk_idx, 1.0)

    return hard - scores.detach() + scores

class LoRMAdapter(nn.Module):
    def __init__(self, dim: int, rank: int = 8, dropout: float = 0.0,
                 use_topk: bool = True, k_ratio: float = 0.5):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.W_down = nn.Linear(dim, rank, bias=False)
        self.W_up   = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.W_down.weight, a=math.sqrt(5)) if rank>0 else None
        nn.init.zeros_(self.W_up.weight)
        self.alpha = nn.Parameter(torch.tensor(1.0))

        self.g_proj = nn.Linear(dim, 1, bias=True)
        self.use_topk = use_topk
        self.k_ratio = k_ratio

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None):
        B, T, D = x.shape

        gate = torch.sigmoid(self.g_proj(x).squeeze(-1))
        if mask is not None:
            gate = gate * mask + (1 - mask) * (-1e4)
        if self.use_topk and self.k_ratio>0:
            k = max(1, int(T * self.k_ratio))
            sel = straight_through_topk_mask(gate, k)      # [B, T], 0/1 (STE)
            gate = torch.clamp(sel, 0.0, 1.0)
        else:
            gate = torch.sigmoid(gate) if mask is None else torch.sigmoid(gate) * mask

        delta = self.W_up(self.W_down(x))                  # [B, T, D]
        y = x + self.alpha * self.dropout(delta) * gate.unsqueeze(-1)
        return y, gate

import math

class ASAAdapter(nn.Module):
    def __init__(self, dim: int, num_heads: int=8, max_off: int=2, topk_ratio: float=0.5, attn_drop: float=0.0, proj_drop: float=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by heads"
        self.scale = self.head_dim ** -0.5
        self.max_off = max_off
        self.topk_ratio = topk_ratio

        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop>0 else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop>0 else nn.Identity()

        self.off_param = nn.Parameter(torch.zeros(num_heads))

        self.score_proj = nn.Linear(dim, 1, bias=True)

    @staticmethod
    def _round_ste(x):
        return (x - x.detach()) + torch.round(x.detach())

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None, base_gate: torch.Tensor=None):
        """
        x: [B, T, D]; mask: [B, T]; base_gate: 可传入 LoRM 的 gate 做联合筛选（可选）
        """
        B, T, D = x.shape
        score = self.score_proj(x).squeeze(-1)  # [B, T]
        if mask is not None:
            score = score + (mask - 1) * 1e4  # pad -> -inf
        if base_gate is not None:
            score = score + base_gate

        k = max(1, int(T * self.topk_ratio))
        sel = straight_through_topk_mask(score, k)  # [B, T]
        sel_bool = (sel > 0.0).to(x.dtype)

        qkv = self.qkv(x)                     # [B, T, 3D]
        q, k_, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, Hd]
        k_ = k_.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        off = torch.clamp(self._round_ste(self.off_param), -self.max_off, self.max_off)  # [H]
        k_shifted, v_shifted = [], []
        for h in range(self.num_heads):
            shift = int(off[h].item())
            k_shifted.append(torch.roll(k_[:, h], shifts=shift, dims=1))
            v_shifted.append(torch.roll(v[:, h], shifts=shift, dims=1))
        k_shifted = torch.stack(k_shifted, dim=1)  # [B, H, T, Hd]
        v_shifted = torch.stack(v_shifted, dim=1)

        attn_logits = (q * self.scale) @ k_shifted.transpose(-2, -1)  # [B, H, T, T]

        # mask padding
        if mask is not None:
            m = (mask[:, None, None, :].to(dtype=attn_logits.dtype))  # [B,1,1,T]
            attn_logits = attn_logits + (m - 1) * 1e4

        q_sel = sel_bool[:, None, :, None]  # [B,1,T,1]
        attn_logits = attn_logits + (q_sel - 1) * 1e4

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v_shifted  # [B, H, T, Hd]
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.proj_drop(self.proj(out))

        y = x * (1 - sel_bool.unsqueeze(-1)) + (x + out) * sel_bool.unsqueeze(-1)
        return y, sel
