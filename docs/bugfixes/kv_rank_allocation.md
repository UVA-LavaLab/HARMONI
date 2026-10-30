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

---

## Fix 3 — KV allocator gave two requests the same address

**TL;DR:** The KV allocator pins `Ra` to the batch's KV rank, but it still divided the
request number by `num_kv_ranks`, as if `Ra` had used up request bits. That discarded one
request bit (two with 4 KV ranks). As a result, requests with the same `Co` in the same
`num_kv`-aligned group of rows (`r // csl_lines`) shared an address, and every KV tensor's reservation
collapsed to half (or a quarter) of the cells it was given. Pinned `Ra` now consumes no request bits.

### Why it happened
`calculate_interleaving_aware_offset_with_batch_round_robin` fills the fields from LSB to MSB
(`Bt Ba Bg Co Ra Ch Ro`), taking bits from the request number for each one:

```
Co = r & (csl_lines - 1);  r >>= log2(csl_lines)
Ra = num_wt + batch % num_kv           # pinned: does not depend on r
r //= num_kv                           # <-- drops the low bit(s) of what is left
Ch = r & (num_channels - 1); ...; Ro = ...
```

The dropped bit belongs to no field, so on `DDR5-M4-R4-C8-8-A2` (2 KV ranks, 64 columns)
requests 0 and 64 both give `0x8000`, and 128 and 192 both give `0x18000`. With 4 KV ranks,
four requests share each address. With 1 KV rank (`//= 1`) nothing was lost.

### The fix
```
elif component == 'Ra':
    Ra = num_wt + batch % num_kv       # pinned; consume no request bits
```
Request `r` now maps one-to-one to `(Co, Ch, Ro)` inside the batch's rank. This fix does not
change the weight allocator (whose `Ra` really is taken from the request), `temp_tensor_kv`
(explicit channel/rank allocator), or the shared `kv_request_counter` (Fix 4).
Code: `modeling/core/dram_utils.py:calculate_interleaving_aware_offset_with_batch_round_robin`.

### Validation
`HARMONI_phase2_diag/alloc_check.py` logs every KV allocator call. Each (K or V) head group
owns the requests from its call up to the next call, and each head owns them on its own
chip. Every request is decoded to a cell `(Ch, Ra, chip, Ro, Co)`. The tool also runs
`kv_check.py` (walked cells). Example `DDR5-M4-R4-C8-8-A2`, LLAMA2-7B, batch 2, seq 1024:

| Check | Before | After |
|---|---|---|
| Requests sharing an address with another request on the same KV rank | 131,072 | 0 |
| Requests aliasing another request of the same tensor | 1,048,576 | 0 |
| Distinct reserved cells | 1,048,576 | 2,097,152 |
| Reserved cells shared by >1 tensor / off the target rank | 0 / 0 | 0 / 0 |
| Walked cells outside the tensor's own reservation (Fix 5) | 1,572,864 | 1,048,576 |
| KV row / column accesses, per-rank task counts | — | unchanged |
| Weight digest, `temp_tensor_kv` addresses | — | identical |

All 12 configs: aliasing goes to 0 everywhere and the reserved cells grow 2× (4× with
4 KV ranks); nothing is shared or off-rank (at seq 16, batch 1, 8,192 reserved cells were
shared between tensors before the fix). KV rows/cols, per-rank tasks, weights and
`temp_tensor_kv` are unchanged in all 12. The 1-KV-rank config (`DDR5-M16-R2`) is identical in every metric.
Except in the DDR4 control case below, walked cells shared by >1 tensor grow by the same
factor: about 2× with 2 KV ranks (e.g.
526,336 → 1,044,480 of 530,432 → 1,052,672 distinct) and about 4× with 4 KV ranks
(`DDR5-M4-R8`: 534,528 → 2,088,960); they are unchanged with 1 KV rank. The walk now reaches
distinct cells instead of collapsed ones, but it still runs into the next tensor's cells
until Fix 5. `DDR4-M8-R4-C8-8-A2` is the control
case: its walk step is exactly one request, so after this fix the walked cells equal the
reserved cells (shared walked cells 4,177,920 → 0, walked outside reservation 4,194,304 → 0).

Default run (`run.py --model_name LLAMA2-7B --dtype BF16 --dram DDR5-M4-R4-C8-8-A2 -i 32
-o 16 --simulate --optimization layer static_mapping --fused_attn --fused_qkv --no_cache`):

| Run | Tasks per rank | Latency before | Latency after |
|---|---|---|---|
| `-b 1` | unchanged (kv_rank 0 = 1648, kv_rank 1 = 0) | 31,832.268 µs | 31,712.120 µs |
| `-b 2` | unchanged (1648 / 1648) | 43,505.984 µs | 43,383.915 µs |

### Impact on results
Small, for configs that stay within capacity (see the caveat below). KV row/column counts and task counts do not change. The DFG's KV locations used to
crowd into channels 0–2, because aliased bases folded back onto the same channels. They now
spread over all four channels (batch 1, layer 0: per-channel locations {768, 752, 128, 0} →
{384, 632, 504, 128}), which lowers latency by about 0.3–0.4%. Before this fix, KV tensors
had about half (or a quarter, with 4 KV ranks) of their reserved space. The cells
the walker reports were never what the allocator reserved, and Fix 5 addresses that.

**Do not produce results from this commit on its own; Fix 4 must land with it.** Because
pinned `Ra` no longer consumes request bits, KV addresses now climb through `Ro` `num_kv`×
faster. With the counter still shared by all KV ranks, the number of requests that fit before
`Ro` wraps falls from `num_kv · 2^(Co+Ch+Ro)` to `2^(Co+Ch+Ro)`. A scan of all 120 configs in
`params/*.txt` (`HARMONI_phase2_diag/rev/ro_overflow.py`) finds 0 that overflow before this fix
and 3 after, all LLAMA2-7B on `DDR5-M1-R4-C8-8-A2` with batch 8:

| Config | KV requests / capacity | Effect |
|---|---|---|
| `ispass_ae_large1.txt:12` (i 2048, o 2048) | 2.0× | batches 4–7 wrap onto batches 0–3 (same KV rank) |
| `ispass_ae_small.txt:26` (i 2048, o 128) | 1.06× | wraps into cells unused on that rank; no collision |
| `ispass_ae_large.txt:27` (i 128, o 2048) | 1.06× | wraps into cells unused on that rank; no collision |

Fix 4 gives each KV rank its own counter, which halves the per-rank load (1.0×, exactly full, and 0.53×),
so all three fit again. The `temp_tensor_kv` wrap (31 configs) does not depend on this fix
and is also handled in Fix 4.
