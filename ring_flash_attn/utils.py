from typing import Optional, Tuple, Any

from einops import rearrange
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._functional_collectives import all_gather_tensor, reduce_scatter_tensor

__all__ = ["update_out_and_lse", "RingComm"]


# using all_to_all_single api to perform all to all communication
def _all_to_all_single(input_, seq_world_size, group, scatter_dim, gather_dim):
    inp_shape = list(input_.shape)
    inp_shape[scatter_dim] = inp_shape[scatter_dim] // seq_world_size
    if scatter_dim < 2:
        input_t = input_.reshape([seq_world_size, inp_shape[scatter_dim]] + inp_shape[scatter_dim + 1 :]).contiguous()
    else:
        input_t = (
            input_.reshape([-1, seq_world_size, inp_shape[scatter_dim]] + inp_shape[scatter_dim + 1 :])
            .transpose(0, 1)
            .contiguous()
        )

    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group)

    if scatter_dim < 2:
        output = output.transpose(0, 1).contiguous()

    return output.reshape(
        inp_shape[:gather_dim]
        + [
            inp_shape[gather_dim] * seq_world_size,
        ]
        + inp_shape[gather_dim + 1 :]
    ).contiguous()


# using all_to_all api to perform all to all communication
def _all_to_all(input_, world_size, group, scatter_dim, gather_dim):
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()

class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        world_size = dist.get_world_size(process_group)
        bsz, _, _ = input_.shape

        # Todo: Try to make all_to_all_single compatible with a large batch size
        if bsz == 1:
            return _all_to_all_single(input_, world_size, process_group, scatter_dim, gather_dim)
        else:
            return _all_to_all(input_, world_size, process_group, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx, *grad_output):
        process_group = ctx.process_group
        scatter_dim = ctx.gather_dim
        gather_dim = ctx.scatter_dim
        return_grad = _AllToAll.apply(*grad_output, process_group, scatter_dim, gather_dim)
        return (return_grad, None, None, None)

def all_to_all_comm(input_, process_group=None, scatter_dim=2, gather_dim=1):
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim)

@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))

    out = torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out

    lse = new_lse
    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


@torch.jit.script
def flatten_varlen_lse(lse, cu_seqlens):
    new_lse = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse.append(lse[i, :, : end - start])
    return torch.cat(new_lse, dim=1)


@torch.jit.script
def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen: int):
    num_seq = len(cu_seqlens) - 1
    num_head = lse.shape[-2]
    new_lse = torch.empty(
        (num_seq, max_seqlen, num_head, 1), dtype=torch.float32, device=lse.device
    )
    for i in range(num_seq):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse[i, : end - start] = lse[start:end]
    return new_lse


class RingComm:
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

    def send_recv(
        self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_rank = (self.rank + 1) % self.world_size
        recv_rank = (self.rank - 1) % self.world_size

        if self._process_group is not None:
            send_rank = dist.get_global_rank(self._process_group, send_rank)
            recv_rank = dist.get_global_rank(self._process_group, recv_rank)

        send_op = dist.P2POp(dist.isend, to_send, send_rank, group=self._process_group)
        recv_op = dist.P2POp(dist.irecv, res, recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []

class AsyncAllGatherForTwo(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        sp_rank: int,
        sp_size: int,
        group: Optional[dist.ProcessGroup] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Returns:
            outputs: Tensor
            handle: Optional[Work], if overlap is True
        """
        ctx.group = group
        ctx.sp_rank = sp_rank
        ctx.sp_size = sp_size

        # all gather inputs
        all_inputs = all_gather_tensor(inputs.unsqueeze(0), 0, group)
        # compute local qkv
        local_qkv = F.linear(inputs, weight, bias).unsqueeze(0)

        # remote compute
        remote_inputs = all_inputs[1 - sp_rank].view(list(local_qkv.shape[:-1]) + [-1])
        # compute remote qkv
        remote_qkv = F.linear(remote_inputs, weight, bias)

        # concat local and remote qkv
        if sp_rank == 0:
            qkv = torch.cat([local_qkv, remote_qkv], dim=0)
        else:
            qkv = torch.cat([remote_qkv, local_qkv], dim=0)
        qkv = rearrange(qkv, "sp b n c -> b (sp n) c")

        ctx.save_for_backward(inputs, weight, remote_inputs)
        return qkv

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> Tuple[torch.Tensor, None, None]:
        group = ctx.group
        sp_rank = ctx.sp_rank
        sp_size = ctx.sp_size
        inputs, weight, remote_inputs = ctx.saved_tensors

        # split qkv_grad
        qkv_grad = grad_outputs[0]
        qkv_grad = rearrange(qkv_grad, "b (sp n) c -> sp b n c", sp=sp_size)
        qkv_grad = torch.chunk(qkv_grad, 2, dim=0)
        if sp_rank == 0:
            local_qkv_grad, remote_qkv_grad = qkv_grad
        else:
            remote_qkv_grad, local_qkv_grad = qkv_grad

        # compute remote grad
        remote_inputs_grad = torch.matmul(remote_qkv_grad, weight).squeeze(0)
        weight_grad = torch.matmul(remote_qkv_grad.transpose(-1, -2), remote_inputs).squeeze(0).sum(0)
        bias_grad = remote_qkv_grad.squeeze(0).sum(0).sum(0)

        # launch async reduce scatter
        remote_inputs_grad_zero = torch.zeros_like(remote_inputs_grad)
        if sp_rank == 0:
            remote_inputs_grad = torch.cat([remote_inputs_grad_zero, remote_inputs_grad], dim=0)
        else:
            remote_inputs_grad = torch.cat([remote_inputs_grad, remote_inputs_grad_zero], dim=0)
        remote_inputs_grad = reduce_scatter_tensor(remote_inputs_grad, "sum", 0, group)

        # compute local grad and wait for reduce scatter
        local_input_grad = torch.matmul(local_qkv_grad, weight).squeeze(0)
        weight_grad += torch.matmul(local_qkv_grad.transpose(-1, -2), inputs).squeeze(0).sum(0)
        bias_grad += local_qkv_grad.squeeze(0).sum(0).sum(0)

        # sum remote and local grad
        inputs_grad = remote_inputs_grad + local_input_grad
        return inputs_grad, weight_grad, bias_grad, None, None, None

class AsyncAllGatherMulti(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        sp_rank: int,
        sp_size: int,
        group: Optional[dist.ProcessGroup] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Returns:
            outputs: Tensor
            handle: Optional[Work], if overlap is True
        """
        ctx.group = group
        ctx.sp_rank = sp_rank
        ctx.sp_size = sp_size

        qkv = inputs
        next_qkv = None
        inter_tensors = [[] for i in range(sp_size)]

        comm = RingComm(group)
        for i in range(sp_size):
            if i != sp_size - 1:
                next_qkv: torch.Tensor = comm.send_recv(qkv)
                comm.commit()
            
            remote_qkv = F.linear(qkv, weight, bias).unsqueeze(0)
            inter_tensors[((sp_rank - i + sp_size) % sp_size)] = remote_qkv

            if i != sp_size - 1:
                comm.wait()
                qkv = next_qkv
        
        qkv = torch.cat(inter_tensors, dim=0)
        qkv = rearrange(qkv, "sp b n c -> b (sp n) c")

        ctx.save_for_backward(inputs, weight)
        return qkv

        # # all gather inputs
        # all_inputs = all_gather_tensor(inputs.unsqueeze(0), 0, group)
        # # compute local qkv
        # local_qkv = F.linear(inputs, weight, bias).unsqueeze(0)

        # # remote compute
        # remote_inputs = all_inputs[1 - sp_rank].view(list(local_qkv.shape[:-1]) + [-1])
        # # compute remote qkv
        # remote_qkv = F.linear(remote_inputs, weight, bias)

        # # concat local and remote qkv
        # if sp_rank == 0:
        #     qkv = torch.cat([local_qkv, remote_qkv], dim=0)
        # else:
        #     qkv = torch.cat([remote_qkv, local_qkv], dim=0)
        # qkv = rearrange(qkv, "sp b n c -> b (sp n) c")

        # ctx.save_for_backward(inputs, weight, remote_inputs)
        # return qkv
