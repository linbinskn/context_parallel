import os

from flash_attn import flash_attn_qkvpacked_func
import torch
import torch.distributed as dist
from ring_flash_attn import fastseq_flash_attn_qkvpacked_func
from dist_attn import DistAttention

def log(msg, a, rank0_only=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank0_only:
        if rank == 0:
            print(
                f"{msg}: "
                f"max {a.abs().max().item()}, "
                f"mean {a.abs().mean().item()}",
                flush=True,
            )
        return

    for i in range(world_size):
        if i == rank:
            if rank == 0:
                print(f"{msg}:")
            print(
                f"[{rank}] "
                f"max {a.abs().max().item()}, "
                f"mean {a.abs().mean().item()}",
                flush=True,
            )
        dist.barrier()

def test_parallel_attention(seqlen, verify):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    sequence_parallel_type = "fastseq"

    batch_size = 1
    # seqlen = 3816
    nheads = 16
    d = 72
    hidden_size = nheads * d
    dropout_p = 0
    causal = True
    deterministic = False

    atten_module = DistAttention(nheads, d, world_size, sequence_parallel_type).cuda(device).to(dtype)
    x = torch.randn(batch_size, seqlen, nheads * d, dtype=dtype, device=device)
    # keep consistency of the content of input
    dist.broadcast(x, src=0)

    qkv_weight = torch.randn(3 * hidden_size, hidden_size, dtype=dtype, device=device)
    qkv_bias = torch.randn(3 * hidden_size, dtype=dtype, device=device)
    dist.broadcast(qkv_weight, src=0)
    dist.broadcast(qkv_bias, src=0)

    atten_module.qkv.weight = torch.nn.Parameter(qkv_weight)
    atten_module.qkv.bias = torch.nn.Parameter(qkv_bias)
    
    # split input into world size
    local_x = x.chunk(world_size, dim=1)[rank].detach().clone()
    
    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# forward:")
        print("#" * 30)
    
    fastseq_out = atten_module(local_x)

    if verify:
        ### reference
        refer_qkv = atten_module.qkv(x)
        # refer_qkv = refer_qkv.reshape(batch_size, seqlen, 3, nheads, d).contiguous()
        refer_qkv = refer_qkv.reshape(batch_size, seqlen, nheads, 3, d).contiguous()
        refer_qkv = refer_qkv.permute((0, 1, 3, 2, 4)).contiguous()
        refer_out, refer_lse, _ = flash_attn_qkvpacked_func(
            refer_qkv,
            dropout_p=dropout_p,
            causal=causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=deterministic,
            return_attn_probs=True,
        )
        refer_local_out = refer_out.chunk(world_size, dim=1)[rank]

        log("out diff", refer_local_out - fastseq_out)

    num_iter = 20
    for i in range(num_iter):
        ring_out = atten_module(local_x)
    import time
    torch.cuda.synchronize(device)
    t0 = time.time()

    # begin = torch.cuda.Event(enable_timing=True)
    # begin.record()

    for i in range(num_iter):
        ring_out = atten_module(local_x)

    # end = torch.cuda.Event(enable_timing=True)
    # end.record()
    torch.cuda.synchronize(device)
    t1 = time.time()
    # time = begin.elapsed_time(end)
    time = (t1-t0) * 1000

    free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
    peak_memory = total_gpu_memory - free_gpu_memory
    # Byte -> GB
    peak_memory = peak_memory / (1024 * 1024 * 1024)

    if rank == 0 and log:
        print(f"{num_iter / time:.3f} iter/s, {time / num_iter} ms, peak_memory: {peak_memory} GB")

if __name__ == "__main__":
    dist.init_process_group("nccl")
    test_seqlens = [512 * 8, 2048 * 8, 8192 * 8, 20000 * 8, 100000 * 8]
    verify = False
    for test_seqlen in test_seqlens:
        test_parallel_attention(test_seqlen, verify)