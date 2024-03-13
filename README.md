## Ulysses/Ring/FastSeq Attention

This repo implements the Ulysses, Ring and FastSeq Attention.

### Test

```bash
PYTHONPATH=. torchrun --nproc_per_node 8 test_attention/test_attention_ulysses.py
PYTHONPATH=. torchrun --nproc_per_node 8 test_attention/test_attention_ring_zigzag.py
PYTHONPATH=. torchrun --nproc_per_node 8 test_attention/test_attention_fastseq.py
```