"""
Analytical Communication Latency Model
=======================================
Computes T_latency (ns) for collective operations on a 4-level hierarchical tree.
Model: Store-and-forward, generalized Hockney alpha-beta with D/D/1/K buffering.

Topology levels:
    0 = Root
    1 = Channel
    2 = Rank
    3 = Chip  (leaf)

Units (consistent throughout — no internal conversion needed):
    T_latency     : ns
    BW[k]         : GBps  (bytes/GBps = ns  ✓)
    l[k], o[k], L0: ns
    Q[k]          : bytes
    flit_size     : bytes
    flit_header   : bytes
    S             : bytes

Supported operations:
    gather        : leaves -> root,  switch concatenates children (fan-in, buffering applies)
    reduce        : leaves -> root,  switch aggregates to S'      (fan-in, buffering applies)
    broadcast     : root -> leaves,  switch replicates S'         (fan-out, no buffering)
    scatter       : root -> leaves,  switch demultiplexes block   (fan-out, no buffering)
    all_reduce    : reduce(s->d) + broadcast(d->s)
    gather_scatter: gather(s->d)  + scatter(d->s)

Direction rules:
    Upward   (gather, reduce)        : src > dst
    Downward (broadcast, scatter)    : src < dst
    Compound (all_reduce, gather_scatter): src > dst  (reduce/gather phase drives direction)

Payload factor phi_k per operation:
    gather    : phi_k = prod(F[j], j=k+1..s-1)  — grows toward root
    reduce    : phi_k = 1                         — aggregated at each switch
    broadcast : phi_k = 1                         — same S' replicated
    scatter   : phi_k = prod(F[j], j=k+1..d-1)  — shrinks toward leaves (mirror of gather)

Key symmetry: phi_k(scatter, s->d) == phi_k(gather, d->s) for the same link range.
Serialization costs are therefore identical between gather and scatter.

"""

import math
from misc.type import CommType

VALID_OPERATIONS = {
    "gather", "reduce", "broadcast", "scatter", "all_reduce", "gather_scatter"
}

# Operations that travel upward (src > dst)
UPWARD_OPS   = {"gather", "reduce"}
# Operations that travel downward (src < dst)
DOWNWARD_OPS = {"broadcast", "scatter"}
# Compound operations decomposed into two phases
COMPOUND_OPS = {"all_reduce", "gather_scatter"}


# ---------------------------------------------------------------------------
# Core model functions
# ---------------------------------------------------------------------------

def flit_align(S: float, flit_size: float, flit_header: float) -> float:
    """S' = ceil((S + h) / f) * f   [bytes]"""
    return math.ceil((S + flit_header) / flit_size) * flit_size


def get_fanout(cfg: dict) -> list:
    """Returns fanout array F[0..2]: F[0]=Channel, F[1]=Rank, F[2]=Chip"""
    return [cfg["Channel"], cfg["Rank"], cfg["Chip"]]


def payload_factor(k: int, s: int, d: int, operation: str, F: list) -> float:
    """
    phi_k: payload scaling factor at link k.

    Gather    (s>d): phi_k = prod(F[j], j=k+1..s-1)  grows toward root
    Reduce    (s>d): phi_k = 1
    Broadcast (s<d): phi_k = 1
    Scatter   (s<d): phi_k = prod(F[j], j=k+1..d-1)  shrinks toward leaves

    Note: phi_k(scatter, src=s, dst=d) == phi_k(gather, src=d, dst=s)
          for the same link range — serialization is symmetric.
    """
    if operation == "gather":
        phi = 1.0
        for j in range(k + 1, s):
            phi *= F[j]
        return phi
    elif operation == "scatter":
        phi = 1.0
        for j in range(k + 1, d):
            phi *= F[j]
        return phi
    else:
        # reduce, broadcast: phi_k = 1
        return 1.0


def compute_T(operation: str, s: int, d: int, S_prime: float, cfg: dict,
              include_buffering: bool = True) -> float:
    """
    Compute T(s, d) in ns for a given operation and flit-aligned message size S'.

    Units:  BW in GBps, l/o/L0 in ns, Q/S_prime in bytes → result in ns.

    Compound operations:
        all_reduce    = reduce(s->d)  + broadcast(d->s)
        gather_scatter = gather(s->d) + scatter(d->s)
    Buffering applies only in the upward phase of compound operations.

    Args:
        include_buffering: if False, omits the D/D/1/K buffering term.
    """

    # --- Compound operations: decompose into two sequential phases -----------
    if operation == "all_reduce":
        return (
            compute_T("reduce",    s, d, S_prime, cfg, include_buffering) +
            compute_T("broadcast", d, s, S_prime, cfg, include_buffering)
        )
    if operation == "gather_scatter":
        return (
            compute_T("gather",  s, d, S_prime, cfg, include_buffering) +
            compute_T("scatter", d, s, S_prime, cfg, include_buffering)
        )

    BW = cfg["BW"]  # GBps
    l  = cfg["l"]   # ns
    o  = cfg["o"]   # ns
    L0 = cfg["L0"]  # ns
    Q  = cfg["Q"]   # bytes
    F  = get_fanout(cfg)

    # --- Determine traversed links and intermediate switch nodes -------------
    if operation in UPWARD_OPS:
        if s <= d:
            raise ValueError(
                f"'{operation}' requires src > dst (upward), got src={s}, dst={d}"
            )
        links              = list(range(d, s))      # d, d+1, ..., s-1
        intermediate_nodes = list(range(d + 1, s))  # levels strictly between d and s

    elif operation in DOWNWARD_OPS:
        if s >= d:
            raise ValueError(
                f"'{operation}' requires src < dst (downward), got src={s}, dst={d}"
            )
        links              = list(range(s, d))  # s, s+1, ..., d-1
        # Scatter: one input per switch -> no fan-in -> no buffering
        # Broadcast: parallel outputs  -> no fan-in -> no buffering
        intermediate_nodes = []

    # --- Static latency (ns) -------------------------------------------------
    # T_static = L0 + sum_k( l[k] + o[k] )
    T_static = L0 + sum(l[k] + o[k] for k in links)

    # --- Serialization latency (ns) ------------------------------------------
    # T_serial = sum_k( phi_k * S'[bytes] / BW[k][GBps] )
    T_serial = sum(
        payload_factor(k, s, d, operation, F) * S_prime / BW[k]
        for k in links
    )

    # --- Buffering delay (ns) — D/D/1/K, intermediate nodes only -------------
    # Only applies to upward operations (gather, reduce).
    # Downward ops (broadcast, scatter) have no fan-in -> intermediate_nodes = []
    #
    # For node n (intermediate switch):
    #   k_in  = n   : input  link (children at level n+1 -> node n)
    #   k_out = n-1 : output link (node n -> parent at level n-1)
    #   injected = F[n] * phi_{k_in} * S'   [bytes arriving at buffer]
    #   Dq = max(0, (injected - Q[n]) / BW[k_out])   [ns]
    T_buffer = 0.0
    if include_buffering:
        for n in intermediate_nodes:
            k_in     = n
            k_out    = n - 1
            injected = F[n] * payload_factor(k_in, s, d, operation, F) * S_prime
            T_buffer += max(0.0, (injected - Q[n]) / BW[k_out])

    return T_static + T_serial + T_buffer   # ns


# ---------------------------------------------------------------------------
# Direction validation helper
# ---------------------------------------------------------------------------

def validate_direction(op: str, s: int, d: int) -> str | None:
    """
    Returns an error string if (op, src, dst) violates direction rules,
    or None if valid.
    """
    if op in UPWARD_OPS and s <= d:
        return f"'{op}' requires src > dst (upward), got src={s}, dst={d}"
    if op in DOWNWARD_OPS and s >= d:
        return f"'{op}' requires src < dst (downward), got src={s}, dst={d}"
    if op in COMPOUND_OPS and s <= d:
        return (f"'{op}' requires src > dst "
                f"(upward phase drives direction), got src={s}, dst={d}")
    return None

# Entry function
def comm_type_based_latency(
    s: int,
    d: int,
    S: float,
    cfg: dict,
    optype: CommType,
) -> float:
    """
    Compute communication latency (ns) from direct args.

    optype must be a CommType enum.

    Returns:
        T_ns (float)
    """
    s = int(s)
    d = int(d)
    S = float(S)
    if not isinstance(optype, CommType):
        raise TypeError("optype must be CommType (e.g., CommType.GATHER).")
    op = optype.value

    if op not in VALID_OPERATIONS:
        raise ValueError(
            f"unknown operation '{op}'. Must be one of {sorted(VALID_OPERATIONS)}."
        )

    if not (0 <= s <= 3 and 0 <= d <= 3): #CLEANUP: update while modeling more levels ( i.e. bank, bankgroups)
        raise ValueError("src/dst levels must be in [0, 3].")

    if s == d:
        return 0.0

    dir_err = validate_direction(op, s, d)
    if dir_err:
        raise ValueError(dir_err)

    S_prime = flit_align(S, cfg["flit_size"], cfg["flit_header"])
    T_ns = compute_T(op, s, d, S_prime, cfg, include_buffering=True)
    
    return T_ns
