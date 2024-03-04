from flash_attn import flash_attn_qkvpacked_func
import torch
import torch.distributed as dist
from ring_flash_attn import (
    ring_flash_attn_qkvpacked_func,
    zigzag_ring_flash_attn_qkvpacked_func,
    stripe_flash_attn_qkvpacked_func,
    ulysses_flash_attn_qkvpacked_func,
)
import torch.cuda


def benchmark(f, num_iter=1, forward_only=True, log=True):
    dtype = torch.bfloat16
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    batch_size = 1
    seqlen = 200000 * 8
    nheads = 16
    d = 72
    dropout_p = 0
    causal = True
    deterministic = False

    assert seqlen % (2 * world_size) == 0
    assert d % 8 == 0

    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)

    with torch.no_grad():
        for _ in range(num_iter):
            _ = f(
                qkv,
                dropout_p=dropout_p,
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=deterministic,
                return_attn_probs=False,
            )

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()

    if forward_only:
        with torch.no_grad():
            for _ in range(num_iter):
                _ = f(
                    qkv,
                    dropout_p=dropout_p,
                    causal=causal,
                    window_size=(-1, -1),
                    alibi_slopes=None,
                    deterministic=deterministic,
                    return_attn_probs=False,
                )

    else:
        for _ in range(num_iter):
            qkv.grad = None
            out = f(
                qkv,
                dropout_p=dropout_p,
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=deterministic,
                return_attn_probs=False,
            )
            out.backward(dout)
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000.0

    if rank == 0 and log:
        print(f"{num_iter / time:.3f} iter/s, {time / num_iter} sec")


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    forward_only = True

    for f in [
        flash_attn_qkvpacked_func,
        ring_flash_attn_qkvpacked_func,
        zigzag_ring_flash_attn_qkvpacked_func,
        stripe_flash_attn_qkvpacked_func,
        ulysses_flash_attn_qkvpacked_func,
    ]:
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"# {f.__name__}")
        benchmark(f, forward_only=forward_only, log=False)
        benchmark(f, forward_only=forward_only, log=True)
