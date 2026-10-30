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
because each tensor now sits in fewer places with more rows in each. Row and column counts
do not enter latency or energy (see Fix 5); Fix 1's latency change came from the task and
location structure (KV tasks moving from both KV ranks onto one).
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

## Fix 4 — KV ranks shared one request counter, and nothing stopped Ro from wrapping

**TL;DR:** `update_model_kv` had one `kv_request_counter` for all KV ranks, so each batch's
data started after every earlier batch, including batches on other ranks. That left holes in
each rank and used up its rows `num_kv`× too fast. Nothing checked the top: when `Ro` ran out,
the allocators silently dropped the extra bits and wrapped onto rows that already held KV data.
`temp_tensor_kv` took its request number from that same KV counter, but its allocator counts
in different units, so in 29 of the 120 AE configs it landed on KV cells. Head groups were
also miscounted when `kv_heads % num_chips != 0`. Now each KV rank has its own counter, every
head group advances it, the temp tensors sit in the first row above their rank's KV data, and
anything that does not fit raises a `ValueError`. Two AE configs now raise (see Impact).
**Fix 3 and Fix 4 together are safe for producing results; Fix 3 alone is not.**

### Why it happened
Four problems in `modeling/core/model_alloc.py:update_model_kv`:

```
counter = 0                                   # 1. one counter for every KV rank
for b in batches:                             #    batch b is on kv_rank b % num_kv
    for layer:
        k_start = counter
        for h in heads:                       #    group = num_chips heads, one per chip
            if h % chips == 0: base = rr_alloc(counter, batch=b)
            if (h + 1) % chips == 0: counter += group      # 3. a partial last group never advances
        counter = k_start + ceil(K_bytes / partition)      # 3. "rewind" to the exact K size
        for h in heads: ... same, for V ...
for ch, rank: temp = explicit_alloc(counter, ch, rank)    # 4. KV-counter units, explicit allocator
              counter += temp_requests
```

1. **Shared counter.** Batch 1 (kv_rank 1) started where batch 0 (kv_rank 0) ended, so
   kv_rank 1 stayed empty below that point and its data sat higher in `Ro` than needed.
2. **Silent wrap.** Both allocators fill fields with `field = r & mask; r >>= bits`. Bits left
   over after `Ro` were dropped, so request `cap + x` landed on request `x` (Fix 3's caveat).
3. **Partial head groups.** With `kv_heads % num_chips != 0`, the last V group never advanced
   the counter, so the next layer's K was placed on top of it. The K rewind was also wrong:
   each group needs `requests_local * num_chips` requests (one head per chip), but the rewind
   reserves only the exact K bytes. Whenever a partial group exists or `requests_local` rounds
   up, the last K group runs into V's first group. None of the 120 AE configs hits this. It does
   happen for MISTRAL-7B or LLAMA3-70B (8 KV heads) on 16-chip ranks, and whenever the K bytes
   per head are not a multiple of `partition_size`. For `head_dim` 128 in BF16, that means a
   `seq_len` that is not a multiple of 32, 16 or 8 on 16-, 8- or 4-chip ranks.
4. **temp_tensor_kv.** The explicit channel/rank allocator pins `Ch` and `Ra` and takes only
   `Co` and `Ro` from its request number. It was given the KV counter, which also counts `Ch`.
   That number is `num_channels`× too large for it, so it wrapped `Ro`. In 29 of 120 AE configs
   the temp tensor then sat on KV cells. 31 configs wrap; in 2 of them (`large1.txt:8` and
   `:10`, LLAMA3-70B) the wrap misses KV data.

### The fix
```
cap      = requests_before_wrap(dram, pinned=['Ra'])        # 2^(Co+Ch+Ro) per KV rank (Ra pinned, Fix 3)
counter  = [0] * num_kv                                       # one per KV rank
group    = ceil(requests_local * num_chips)
for b in batches:
    r = b % num_kv
    for layer, for K then V:
        for h in heads:
            if h % chips == 0:
                check(counter[r] + group <= cap)              # else ValueError
                base = rr_alloc(counter[r], batch=b)
            if (h + 1) % chips == 0 or h == kv_heads - 1:     # partial last group too
                counter[r] += group                           # (no K rewind)
per_row  = requests_before_wrap(dram, ['Ch', 'Ra', 'Ro'])    # 2^Co: explicit-allocator requests per row
tcap     = requests_before_wrap(dram, ['Ch', 'Ra'])          # 2^(Co+Ro) per (channel, rank)
for r: row[r] = Ro(rr_alloc(counter[r] - 1, batch=r)) + 1 if counter[r] else 0
for ch, r:
    check(row[r] * per_row + temp_requests <= tcap)           # else ValueError
    temp = explicit_alloc(row[r] * per_row, ch, r)            # first row above r's KV data
```
- `requests_before_wrap` (new, `dram_utils.py`) counts the request bits an allocator uses:
  every field except the pinned ones, and except `Ba/Bg/Bt` in AB mode. For the KV allocator
  it does not count `Ra`, matching Fix 3 (pinned `Ra` consumes no request bits).
- A KV rank's data fills rows `0..row[r]-1` in every channel (`Ch` sits under `Ro`), so the
  temp tensors in row `row[r]` cannot touch it.
- The KV round-robin allocator also raises when bits are left over after `Ro`. `update_model_kv`
  checks first, so this guard only catches other callers.
- The error names the tensor (batch, layer, K/V and heads, or the temp tensor and its row), the
  KV rank with its `Ra` (and channel for temp tensors), the model, DRAM config, `seq_len`, batch
  count, the request range and the capacity. Example:
  `K cache of batch 8 layer 0 heads 0-3 does not fit in kv_rank 0 (Ra 1) (LLAMA2-7B on
  DDR5-M4-R2-C4-8-A2, seq_len 4096, batch 9): it needs requests 8388608-8390655, but kv_rank 0
  (Ra 1) holds 8388608 requests. Use a smaller batch or sequence length, or a DRAM config with
  more KV ranks or capacity.`
- Every check raises; there is no log-only path. The explicit allocator itself is unchanged,
  because `temp_tensor_wt` also uses it.

### Validation
**Capacity scan** (`HARMONI_phase2_diag/scan_capacity.py` → `out_fix4_scan.txt`, all 120 configs in
`params/*.txt`): no config overflows with KV data. Two raise on `temp_tensor_kv`:

| Config | KV fill per KV rank | Error |
|---|---|---|
| `ispass_ae_large1.txt:16` LLAMA2-7B `DDR5-M4-R2-C4-8-A2` 2048/2048 b8 | 1.0 (1 KV rank, 8 batches) | `temp_tensor_kv_ch0_r0 (placed at row 32768, …) does not fit in channel 0 of kv_rank 0 (Ra 1) …: it needs requests 2097152-2097155, but channel 0 of kv_rank 0 (Ra 1) holds 2097152 requests` |
| `ispass_ae_large1.txt:12` LLAMA2-7B `DDR5-M1-R4-C8-8-A2` 2048/2048 b8 | 1.0 (2 KV ranks, 4 batches each) | same for `(Ra 2)`, `needs requests 2097152-2097153` |

The next fullest configs reach 0.53 of a rank (`large.txt:27`, `small.txt:26`), then 0.50.
The KV-data check was exercised separately: batch 9 on `DDR5-M4-R2-C4-8-A2` and seq 4097 at
batch 8 on `DDR5-M1-R4-C8-8-A2` both raise on a K/V group, with the message shown above.

**Overlap check** (`HARMONI_phase2_diag/overlap4.py --old` → `out_fix4_overlap.txt`). It covers
the 120 AE configs plus 4 extra configs with partial groups or rounding. Each tensor's true
extent is `ceil(size / chip_partition)` requests from its base, taken modulo the rank capacity.
"Before" is HEAD (Fix 3):

| Check | Before | After |
|---|---|---|
| AE configs with KV slots claimed by >1 tensor | 1 (`large1:12`, Ro wrap) | 0 (it raises) |
| AE configs with `temp_tensor_kv` on KV cells | 29 | 0 |
| MISTRAL-7B `DDR5-M4-R4-C16-8-A2` 512/512 b2 (partial groups): KV slots shared | 385,024 | 0 |
| MISTRAL-7B `DDR5-M4-R4-C16-8-A2` 512/512 b3 (partial groups) | 577,536 | 0 |
| LLAMA3-70B `DDR5-M4-R8-C16-8-A2` 128/128 b4 (partial groups) | 487,424 | 0 |
| LLAMA2-7B `DDR5-M4-R4-C16-8-A2` 32/16 b2 (`requests_local` rounds up) | 8,192 | 0 |
| Configs that raise | 0 | 2 |

**12-config matrix** (`alloc_check.py`, `out_fix4/` vs `out_fix3/post/`). `alloc_check` takes a
group's range up to the next call on the same KV rank, skips the RR calls made for temp
placement, and now also checks that every temp request decodes to its own channel and KV rank.

| Check (all 12 configs, plus the partial-group run) | Fix 3 | Fix 4 |
|---|---|---|
| Reserved KV cells shared by >1 tensor / off the target rank | 0 / 0 | 0 / 0 |
| Tensors given fewer / more requests than they need | 0 / 0 | 0 / 0 |
| Requests aliasing on a KV rank or inside a tensor | 0 | 0 |
| `temp_tensor_kv` cells on KV reservations / on another temp / off target | 0 / — / — | 0 / 0 / 0 |
| KV row / column accesses, per-rank task counts, weight digest | — | unchanged |
| Highest KV request, 2 KV ranks, b2 (`DDR5-M4-R4-C8`) | 261,632 | 130,560 |
| Highest KV request, 4 KV ranks, b4 (`DDR5-M4-R8-C8`) | 523,776 | 130,560 |
| Highest KV request, 1 KV rank, b2 (`DDR5-M16-R2-C8`) | 261,632 | 261,632 |
| Walked cells shared by >1 tensor, b3 / b4 / PHI3 b3 (Fix 5) | 1,566,720 / 2,088,960 / 193,536 | 1,568,768 / 2,093,056 / 194,560 |
| `temp_tensor_kv` addresses | — | changed in all 12 |

- The highest KV request halves with 2 KV ranks and quarters with 4; it is unchanged with 1.
- None of the 12 has temp tensors on KV at Fix 3 (that needs a fuller rank), so the temp check is
  a no-regression check here; the overlap scan above covers the 29 AE configs.
- Walked cells change only when a rank holds several batches: those batches are now adjacent,
  so the walker's overrun into the next tensor (Fix 5) reaches different cells.
- Partial groups: MISTRAL-7B `DDR5-M4-R4-C16-8-A2` b3 s1024 (`out_fix4/extra_*.log`, 8 KV heads on
  16 chips) gives 192 groups, 0 reserved cells shared, 0 off-rank, 0 temp/KV overlap, and every
  tensor gets exactly the requests it needs.
- Until Fix 5, the walker (2× stride) still runs past the last KV tensor's reservation on a
  rank and reaches the temp row: 8 / 64 / 32 of 128 temp cells at b1 s16 / b2 s1024 / b1 s8192
  (`temp_walk_check.py`, `out_fix4/tempwalk_*.txt`). The allocation itself is disjoint; Fix 5
  makes the walk equal the reservation.

Default run (`runwrap.py`, `-i 32 -o 16`, `DDR5-M4-R4-C8-8-A2`):

| Run | Tasks per KV rank | Latency Fix 3 | Latency Fix 4 |
|---|---|---|---|
| `-b 1` | unchanged (1648 / 0) | 31,712.120 µs | 31,712.120 µs |
| `-b 2` | unchanged (1648 / 1648) | 43,383.915 µs | 43,383.915 µs |

Energy and the prefill/decode split are identical as well.

### Impact on results
- **Two AE configs now stop with an error instead of producing numbers:** `ispass_ae_large1.txt:16`
  (LLAMA2-7B, `DDR5-M4-R2-C4-8-A2`, 2048/2048, batch 8) and `ispass_ae_large1.txt:12` (LLAMA2-7B,
  `DDR5-M1-R4-C8-8-A2`, 2048/2048, batch 8).
  - In both, the KV cache fills every KV rank exactly. `large1:16` has one KV rank holding all
    8 batches: 8 × 1,048,576 = 8,388,608 = `2^(Co+Ch+Ro)` requests. `large1:12` has two KV
    ranks with 4 batches each: 4 × 524,288 = 2,097,152 = `2^(Co+Ch+Ro)` (one channel, so `Ch` has no bits). That leaves
    no row for the `temp_tensor_kv`s.
  - Before this fix, their request number wrapped `Ro` and the temp tensors landed on KV row 0,
    on top of batch 0's K cache. On `large1:12` the KV data itself also wrapped after Fix 3,
    with batches 4–7 on top of batches 0–3.
  - So Fix 3's note that Fix 4 makes `large1:12` fit holds for its KV data only.
  - Their earlier results came from an allocation where KV and temp data shared cells. The
    simulator does not model per-cell occupancy, so that overlap did not change their numbers by
    itself; the results are still invalid because the configs do not fit. To run them, use a
    smaller batch or sequence length, or a config with more KV capacity.
- **Everything else still runs.** KV row/column counts and task counts do not change. KV data
  now sits lower in each rank, and the temp tensors move.
  - Latency is unchanged in the default b1/b2 runs. With b1 only the temp tensor moves. With
    b2, batch 1 also moves down by 6,144 requests. That is exactly 24 rows (a multiple of
    `num_channels * csl_lines`), so its channel and column layout stays the same. Configs whose
    per-batch size is not such a multiple will see their channel spread change.
  - Moving the temp tensors does not affect timing: `get_temp_tensor_locations` builds temp
    locations from the size of the dataflow-graph tensor, not from `temp_tensor_kv`'s address.
  - Removing the temp/KV overlap (29 AE configs) changes no result by itself: nothing models
    per-cell occupancy, and temp addresses do not affect timing (above). Results change only
    where KV base addresses moved: configs with partial head groups, and b≥2 configs whose
    per-batch size is not a multiple of `num_channels * csl_lines`.
- **Fix 3 + Fix 4 together are safe for producing results.** Fix 3 alone let three AE configs
  (`DDR5-M1-R4-C8-8-A2` b8) wrap `Ro`; with per-rank counters those fit or raise, and no
  allocation can wrap silently any more. The walker issues (Fix 5, Fix 6) remain and still
  affect walked cells, as they did before Fix 3.

## Fix 5 — KV walker stepped two (or four) columns per chip-partition

**TL;DR:** `get_tensor_locations` walked every tensor in steps of `chip_partition_size`
address units. For KV tensors one chip-partition is one allocator request, which is one `Co`
step (`0x100` on DDR5). `chip_partition_size` is 512 on DDR5 and 1024 on GDDR6, so the walk
touched every 2nd (GDDR6: 4th) column, covered 2× (4×) the tensor's reservation, and ran into
the next tensor and, after Fix 4, into the `temp_tensor_kv` row. KV tensors now step exactly one
request per chip-partition, with the step taken from the KV allocator. Every KV tensor's walked
cells now equal its reserved cells. KV row accesses halve on DDR5 (quarter on GDDR6), but row
counts do not enter latency; only the number and size of locations do (see Impact). Weights are
untouched (Fix 6).

### Why it happened
The walker turns partition `i` of a tensor into an address and decodes it:

```
chip_partition_size = partition_size / num_chips         # bytes per chip per partition
for i in range(num_partitions * num_chips):
    step = i * chip_partition_size                        # chip_idx path; (i // num_chips) * ... for chip_idx == -1
    addr = add_to_address_skipping_rank(base, step)       # KV (Fix 1); weights: base + step
    (Ch, Ra, Ro, Co) = get_mapping(addr)
```

It adds a byte count to an address. In `RoChRaCoBgBaBt` AB mode the bits below `Co`
(`Bg Ba Bt`, 8 bits on DDR5) are zero in every request, so one request is `0x100` address
units while `chip_partition_size` is `0x200`. With 2× the step the walk visits columns
`0, 2, 4, …`, crosses into the next channel and row twice as fast, and ends at twice the
tensor's reserved range. On GDDR6 (`chip_partition_size` 1024, request `0x100`) it is 4×.
DDR4 (`chip_partition_size` 128, request `0x80`) happened to match, so it was already correct.

### The fix
```
step_kv = rr_alloc(1, batch=0) - rr_alloc(0, batch=0)    # one request; checked against the allocator
addr_step = step_kv if KV else chip_partition_size        # weights unchanged until Fix 6
for i in ...:
    step = i * addr_step                                  # or (i // num_chips) * addr_step
    addr = add_to_address_skipping_rank(base, step)       # unchanged: Ra stays pinned
```
- `kv_request_address_step(dram)` (new, `dram_utils.py`, cached per DRAM config) takes the
  step from the KV round-robin allocator. It also checks, at the requests where `Co` carries into
  `Ch` and `Ch` into `Ro`, that `add_to_address_skipping_rank(base, r * step)` equals
  `rr_alloc(r)`. A layout where the address walk would not follow the allocator raises
  instead of walking the wrong cells.
- This keeps address arithmetic instead of walking in request space (`rr_alloc(base_request + i)`).
  Request space would need the batch index and an inverse of the allocator in the walker, which
  runs for every KV slice of every decode step. The check above gives the same guarantee.
- A partition's `end_addr` is now `addr + addr_step` for KV (nothing decodes it).

### Validation
`alloc_check.py` + `temp_walk_check.py` on the 12-config matrix (`out_fix5/` vs `out_fix4/`) and
the partial-group config (MISTRAL-7B `DDR5-M4-R4-C16-8-A2` b3 s1024):

| Check | Fix 4 | Fix 5 |
|---|---|---|
| Walked cells outside the tensor's reservation, `DDR5-M4-R4-C8` b2 s1024 | 1,048,576 | 0 |
| Reserved cells never walked, same | 1,048,576 | 0 |
| Walked cells shared by >1 tensor, same | 1,044,480 | 0 |
| Distinct walked cells, same (reserved: 2,097,152) | 1,052,672 | 2,097,152 |
| `temp_tensor_kv` cells reached by the KV walk, b1 s16 / b2 s1024 / b1 s8192 / partial groups | 8 / 64 / 32 / 64 (of 128) | 0 / 0 / 0 / 0 |
| GDDR6 b2 s1024: walked cells outside reservation / distinct walked cells | 786,432 / 265,216 | 0 / 1,048,576 |
| DDR4 b2 s1024 (step already one request) | — | identical in every metric |

- In all 12 configs and the partial-group run: walked cells outside the reservation, reserved
  cells never walked, walked cells shared by >1 tensor, and temp cells walked are all 0, and the
  distinct walked cells equal the reserved cells exactly.
- Weight digest, KV column accesses and all allocator metrics are unchanged in all 12.
- KV row accesses halve in every DDR5 config (e.g. b2 s1024 65,536 → 32,768; b1 s8192 262,144 →
  131,072), quarter on GDDR6 (131,072 → 32,768), and are unchanged on DDR4. At b1 s16 they go
  2,304 → 2,048.
- Locations (one per `(channel, kv_rank, chip)` a tensor touches, and one task each) change
  where the 2× span crossed more channels: `DDR5-M16-R2` b2 s1024 tensors span 8 channels
  instead of 16 (65,536 → 32,768 per-location tasks), and at b1 s16 256 of 2,048 tensors no
  longer spill into a second channel (2,304 → 2,048). Elsewhere a tensor already covered every
  channel, so location and task counts stay the same (`HARMONI_phase2_diag/loc_stats.py`).

Default run (`runwrap.py`, `-i 32 -o 16`, `DDR5-M4-R4-C8-8-A2`):

| Run | KV tasks per KV rank (per-location) | Aggregation tasks (`wt_rank=None`) | Latency Fix 4 → Fix 5 | Prefill / decode |
|---|---|---|---|---|
| `-b 1` | 1648 (1248) → 1272 (496) | 1281 → 905 | 31,712.120 → 31,705.794 µs (−0.02%) | 10.295 → 10.359 ms / 21.417 → 21.347 ms |
| `-b 2` | 1648 (1248) → 1272 (496), per rank | 2417 → 1665 | 43,383.915 → 43,379.510 µs (−0.01%) | 20.325 → 20.389 ms / 23.059 → 22.991 ms |

Weight tasks are unchanged (4736 / 3104 at b1). Energy: 3344.18 → 3342.07 mJ (b1),
4502.61 → 4498.38 mJ (b2).

### Impact on results
- **Row and column counts do not reach latency or energy.** `transformer_sim.get_node_metrics`
  computes `mem_time = (rows − 1)·tRC + cols·tCCDs` and `memory_access_energy(rows, cols)` from
  them, but neither is used: a node's `exec_time` is its compute time, and the energy comes
  from the kernel models. So halving KV row accesses changes no number by itself.
- **What does move results is the location structure.** `map_tasks_to_logic_units` creates
  one subtask per location `(channel, kv_rank, chip)`, plus aggregation nodes, and splits the
  work by location size. A tensor whose 2× walk crossed into an extra channel is now one
  location, not two. That gives fewer, larger subtasks and fewer aggregation nodes:
  - At seq 48 (default run, b1), 1248 → 496 per-location KV subtasks and 376 fewer aggregation
    nodes. Decode gets faster (21.417 → 21.347 ms). Prefill gets slightly slower
    (10.295 → 10.359 ms): a prefill slice is 16 requests on one chip, and with the 2× walk it
    spans 32 columns, so 3 of the 8 head groups' slices crossed a channel boundary and ran as two
    parallel half-size subtasks. With the 1× walk only 1 of 8 crosses, so that work now runs as
    one subtask. Net −6.3 µs (b1) and −4.4 µs (b2).
  - Where every KV tensor already spans all channels (seq ≥ 512 on `DDR5-M4`), locations do
    not change, so latency does not either. `-i 1024 -o 8 -b 1` gives 380,020.237 µs with both
    the Fix 4 and the Fix 5 step (prefill 368.863 ms, decode 11.158 ms, 2048 KV tasks both
    times). Energy moves by 29,089.85 → 29,088.43 mJ because location sizes shift even though
    their number does not: for decode slices of 1024–1031 tokens, 768 of 8,192 slices split their
    last partial chunk differently (e.g. 2,048 + 2,048 bytes over two channels before, 4,096 in one
    channel now), which changes the ceil-rounded subtask shapes the energy model sees. The "Fix 4"
    side was run with the old step
    patched in (`runwrap_oldstep.py`), which reproduces the Fix 4 default run exactly
    (31,712.120 µs).
  - `DDR5-M16-R2` b1 at seq 48 moves like `DDR5-M4`: 20,841.576 → 20,833.906 µs, with the same
    task pattern (reviewer's run). At seq 1024 the 2× walk spread each tensor over 16 channels
    instead of 8, so there its KV subtasks halve (65,536 → 32,768 at b2) and latency should move
    more; that was not run end to end.
- Results from before this fix modelled KV tensors spread over up to 2× (4× on GDDR6) the
  channels/rows they occupy, overlapping their neighbours. DDR4 results do not change.
