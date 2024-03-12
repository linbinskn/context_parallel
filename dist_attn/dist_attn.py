import torch
import torch.nn as nn
import torch.distributed as dist
from ring_flash_attn import (
    ring_flash_attn_qkvpacked_func,
    zigzag_ring_flash_attn_qkvpacked_func,
    stripe_flash_attn_qkvpacked_func,
    ulysses_flash_attn_qkvpacked_func,
    fastseq_flash_attn_qkvpacked_func,
)
from ring_flash_attn.utils import AsyncAllGatherForTwo, AsyncAllGatherMulti

class DistAttention(nn.Module):
    def __init__(
        self, 
        num_head: int, 
        head_dim: int,
        sequence_parallel_size: int = 1,
        sequence_parallel_type: str = "ring",
    ):
        super().__init__()
        self.num_head = num_head
        self.head_dim = head_dim
        self.hidden_size = num_head * head_dim
        self.qkv = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=True)
        self.sequence_parallel_size = sequence_parallel_size
        self.sequence_parallel_type = sequence_parallel_type
        # For FastSeq
        if self.sequence_parallel_size > 1:
            self.sequence_parallel_rank = dist.get_rank()
            self.sequence_parallel_param_slice = slice(
                self.qkv.out_features // sequence_parallel_size * self.sequence_parallel_rank,
                self.qkv.out_features // sequence_parallel_size * (self.sequence_parallel_rank + 1),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B: batch size, S: seqlen, HS: hidden_size
        B, S, HS = x.shape
        assert self.sequence_parallel_type in ["ring", "ulysses", "fastseq"]
        if self.sequence_parallel_type == "ulysses":
            qkv = self.qkv(x)
            qkv = qkv.reshape(B, S, 3, self.num_head, self.head_dim).contiguous()
            out, lse, _ = ulysses_flash_attn_qkvpacked_func(
                qkv,
                dropout_p=0,
                causal=True,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
        elif self.sequence_parallel_type == "ring":
            qkv = self.qkv(x)
            qkv = qkv.reshape(B, S, 3, self.num_head, self.head_dim).contiguous()
            out, lse, _ = zigzag_ring_flash_attn_qkvpacked_func(
                qkv,
                dropout_p=0,
                causal=True,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
        else:
            num_head = self.num_head // self.sequence_parallel_size
            total_S = S * self.sequence_parallel_size
            qkv = AsyncAllGatherMulti.apply(
                x,
                self.qkv.weight[self.sequence_parallel_param_slice],
                self.qkv.bias[self.sequence_parallel_param_slice],
                self.sequence_parallel_rank,
                self.sequence_parallel_size,
                dist.group.WORLD,
            )
            qkv_shape = (B, total_S, num_head, 3, self.head_dim)
            qkv_permute_shape = (3, 0, 1, 2, 4)
            qkv = qkv.view(qkv_shape).permute(qkv_permute_shape)
            out, lse, _ = fastseq_flash_attn_qkvpacked_func(
                qkv,
                dropout_p=0,
                causal=True,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            # raise NotImplementedError
        
        return out