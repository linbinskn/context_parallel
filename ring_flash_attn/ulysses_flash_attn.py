import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from .utils import RingComm, update_out_and_lse, all_to_all_comm


def ulysses_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    B, S, N, H = q.shape
    context_parallel_size = dist.get_world_size(process_group)

    q = q.reshape(B, S, N * H).contiguous()
    k = k.reshape(B, S, N * H).contiguous()
    v = v.reshape(B, S, N * H).contiguous()

    q = all_to_all_comm(q, process_group)
    k = all_to_all_comm(k, process_group)
    v = all_to_all_comm(v, process_group)

    q = q.reshape(B, S * context_parallel_size, N // context_parallel_size, H).contiguous()
    k = k.reshape(B, S * context_parallel_size, N // context_parallel_size, H).contiguous()
    v = v.reshape(B, S * context_parallel_size, N // context_parallel_size, H).contiguous()

    out = None
    lse = None

    out, _, _, _, _, lse, _, _ = _flash_attn_forward(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        return_softmax=True and dropout_p > 0,
    )
    out = out.reshape(B, S * context_parallel_size, N // context_parallel_size * H).contiguous()

    out = all_to_all_comm(out, process_group, scatter_dim=1, gather_dim=2)
    out = out.reshape(B, S, N, H).contiguous()

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse

class UlyssesFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = ulysses_flash_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        # this should be out_padded
        # ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        return out if not return_softmax else (out, softmax_lse, None)

def ulysses_flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return UlyssesFlashAttnFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )
