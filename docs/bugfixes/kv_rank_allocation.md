# KV / weight rank allocation fixes

A series of fixes to how HARMONI places KV-cache and weight tensors in DRAM and
walks their addresses. Each section below is one commit.

Address layout used in the examples (`DDR5-M4-R4-C8-8-A2`, `RoChRaCoBgBaBt`, AB mode):

```
 MSB                                              LSB
 | Ro (15) | Ch (2) | Ra (2) | Co (6) | Bg (3) | Ba (2) | Bt (3) |
 ranks 0-1 = weight ranks, ranks 2-3 = KV ranks (kv_rank 0, 1)
```

---

## Fix 1 — KV tensors spilled onto every rank (batch round-robin ignored)

**TL;DR:** With batch=1, every KV tensor was reported on *both* KV ranks, so both
KV-rank logic-unit queues filled up. The address walk in `get_tensor_locations`
used plain integer addition, and the carry out of the column bits flowed into the
rank bits. KV tensors now step through memory with the rank bits held fixed.

### Why it happened
`calculate_interleaving_aware_offset_with_batch_round_robin` gives each KV tensor
a base address whose `Ra` field is pinned to `num_wt_ranks + batch % num_kv_ranks`.
`get_tensor_locations` then walks the tensor's partitions:

```
for i in partitions:
    addr = base + i * chip_partition_size        # plain add
    Ra   = decode(addr).Ra                       # carry from Co lands here
```

`Ra` sits directly above `Co`, so once a tensor filled one row of columns the
next partition decoded to the next rank. That rank could be the other KV rank, or
a weight rank. An older patch moved partitions that landed on a *weight* rank back
to the target KV rank. It left partitions that landed on the *other KV rank* alone, so
every tensor ended up split across both KV ranks.

### The fix
```
for i in partitions:
    if tensor is KV:
        addr = add_to_address_skipping_rank(base, i * chip_partition_size)
    else:
        addr = base + i * chip_partition_size    # weights unchanged

add_to_address_skipping_rank(addr, delta):
    take the Ra bits out, add delta to what is left, put the same Ra bits back
    -> carries skip over Ra and go to Ch / Ro instead
```
Code: `modeling/core/dram_utils.py:add_to_address_skipping_rank`,
`modeling/core/tensor.py:get_tensor_locations`.

### Validation
| Check | Before | After |
|---|---|---|
| KV tensors on >1 rank (LLAMA2-7B, batch 1, seq 1024) | 2048 / 2048 | 0 |
| Batches 2–4: batch *i* only on kv_rank *i* % 2 | no | yes |
| Full `run.py --simulate`, batch 1: tasks on kv_rank 1 | 624 | 0 |
| Critical-path latency (same run) | 31,768 µs | 31,832 µs |
| Weight tensor locations | — | byte-identical |

Checked on 12 configs: 4 KV ranks, 1 KV rank, DDR4, GDDR6, 16 chips/rank,
Mistral-7B and PHI3-7B (GQA), and seq 16 / 1024 / 8192.

### Impact on results
KV column-access counts are unchanged. KV row-access counts rise about 1.8×,
because each tensor now sits in fewer places with more rows in each.
Attention results produced before this fix had KV work spread over both
KV ranks.
