import torch
import torch.nn as nn
from ring_flash_attn import (
    ring_flash_attn_qkvpacked_func,
    zigzag_ring_flash_attn_qkvpacked_func,
    stripe_flash_attn_qkvpacked_func,
    ulysses_flash_attn_qkvpacked_func,
)

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
            raise NotImplementedError
        
        return out