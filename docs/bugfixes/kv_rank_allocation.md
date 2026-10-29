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

---

## Fix 2 — Dead KV rank override that wrote into the mapping cache

**TL;DR:** `get_tensor_locations` had a KV branch for "the walked address decoded to a
weight rank". It rewrote `Ra` in the decoded mapping, and that mapping is the dict held
by `get_mapping`'s `lru_cache`, so a later decode of the same address returned the
rewritten rank. After Fix 1 the branch can no longer run, so it is removed. Results
are unchanged.

### Why it happened
Before Fix 1, carries from `Co` flowed into `Ra`, so a KV walk could land on a weight
rank. An older patch handled that case:

```
m = get_mapping(addr)               # lru_cache: returns the cached dict itself
if KV and m.Ra < num_wt_ranks:
    b = batch index parsed from tensor.name
    m['Ra'] = num_wt_ranks + b % num_kv_ranks   # writes into the cache
```

`get_mapping_cached` returns `dict(mapping)`, but `lru_cache` stores that copy and hands
the same object to every later caller. Example: decoding `0x4000` gives `Ra 1`; after
`m['Ra'] = 2`, a fresh `get_mapping(0x4000, ...)` also gives `Ra 2`.

Why the branch is now unreachable:
- Only two KV tensors per (head, batch) reach `get_tensor_locations`:
  `t_vcache_head{h}_batch{b}` and `t_kcacheT_head{h}_batch{b}` in `transformer_dfg.py`
  (both `add_fused_headwise_blocks` and `add_headwise_blocks`). Both copy `addr_offset`
  from `model_kv` tensors. `t_kcache_head{h}_batch{b}` (addr 0) is created but never
  walked or used by a node. `task_mapping.create_modified_tensor` sets `locations`
  directly and does not walk.
- Every `model_kv` address comes from
  `calculate_interleaving_aware_offset_with_batch_round_robin(..., "kv", batch_idx=i)`
  with `i >= 0`. That always takes the round-robin path and writes
  `Ra = num_wt_ranks + i % num_kv_ranks`, which is `>= num_wt_ranks`. Since
  `Ra = num_wt + k < num_ranks`, it always fits in the `Ra` field.
- Fix 1's `add_to_address_skipping_rank` keeps the `Ra` bits fixed for each walked address.
- `--headsplit` defaults to True and cannot be turned off, so there is no other attention
  path. `--fused_attn`, `--fused_qkv`, `--optimization` and batch size change which nodes
  use the tensors, not their addresses.

### The fix
```
elif KV:
    if m.Ra < num_wt_ranks:
        raise ValueError(...)       # a negative kv_rank would match no logic unit and drop KV work silently
    rank = m.Ra - num_wt_ranks      # Ra is the base's KV rank; never written back
```
`extract_batch_idx_from_name` (no other users) and the unused imports are removed.
The old case now raises instead of being patched up: if `kv_rank` went negative,
`find_matching_logic_units` would return nothing and the KV work would disappear from
the simulation without any error.
No other code writes to a mapping dict, so `get_mapping` still returns the cached
object, and a comment there now marks it as read-only.
Code: `modeling/core/tensor.py:get_tensor_locations`, `modeling/core/dram_utils.py:get_mapping_cached`.

### Validation
Probe at Fix 1, with the branch still present (`HARMONI_phase2_diag/branch_probe.py`).
"Default" is `run.py --model_name LLAMA2-7B --dtype BF16 --dram DDR5-M4-R4-C8-8-A2
-i 32 -o 16 -b 1 --simulate --optimization layer static_mapping --fused_attn --fused_qkv
--no_cache`:

| Run | KV tensors walked | Decodes with Ra < num_wt_ranks | Branch entered |
|---|---|---|---|
| default | 1,024 | 0 / 24,064 | 0 |
| same, no `--fused_attn` | 1,024 | 0 / 24,064 | 0 |
| same, no `--fused_qkv` | 1,024 | 0 / 24,064 | 0 |
| same, batch 2 | 2,048 | 0 / 48,128 | 0 |
| kv_check, all layers, R4, b4, seq 1024 | 8,192 | 0 / 4,194,304 | 0 |
| kv_check, all layers, R2 (1 KV rank), b3, seq 1024 | 6,144 | 0 / 3,145,728 | 0 |

| Check | Fix 1 | Fix 2 |
|---|---|---|
| kv_check JSON, 12 configs | baseline | identical |
| Default run: tasks on kv_rank 0 / 1 | 1648 / 0 | 1648 / 0 |
| Critical-path latency (same run) | 31,832.268 µs | 31,832.268 µs |

### Impact on results
None. The branch did not run after Fix 1, so outputs are byte-identical. The fix
removes a hidden way for one decode to change the result of later, unrelated decodes.

**Stale DFG cache (applies to every fix in this file):** `graph_cache/*.pkl` files built
before these fixes still hold the old KV locations. A cached run loads them and never
calls `get_tensor_locations`. The repo's `.sh` scripts do not pass `--no_cache`, so the
fixes only take effect after those pickles are deleted, or with `--no_cache`. The cache
filename (`utils/cache.py:get_model_dfg_cache_filename`) also ignores `--fused_attn` and
`--fused_qkv`: `fusion_suffix` is computed but not used. As a result, runs that differ only
in those flags share one pickle.
