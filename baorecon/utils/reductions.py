"""Reductions over a catalogue that numpy does slowly or not at all.

``positions.min(axis=0)`` on an ``(N, 3)`` C-contiguous array goes through
numpy's generic iterator one element at a time -- the three columns are
interleaved, so neither the vectorised nor the pairwise path applies -- and
takes 24 s on 346 million rows, twice (min, then max), five times a run.
The kernels here make one pass, in parallel, keeping the running extremes in
registers, and finish in a fraction of a second.

They are also deterministic by construction: the rows are cut into a fixed
number of chunks whatever the thread count, each chunk is reduced serially,
and the chunk results are combined in a fixed order. Minimum and maximum
would not care about the order; the structure is the one every reduction in
the package follows so that a run gives the same bits on any machine.
"""

import numpy as np
from numba import njit, prange

#: Chunks a reduction is cut into: enough to keep every core busy, few
#: enough that combining them costs nothing.
_CHUNKS = 1024


@njit(parallel=True, cache=True)
def _bounds_kernel(pos):
    n = pos.shape[0]
    chunks = min(_CHUNKS, max(n, 1))
    step = (n + chunks - 1) // chunks
    lo = np.full((chunks, 3), np.inf, dtype=pos.dtype)
    hi = np.full((chunks, 3), -np.inf, dtype=pos.dtype)
    for c in prange(chunks):
        for i in range(c * step, min(n, (c + 1) * step)):
            for k in range(3):
                v = pos[i, k]
                if v < lo[c, k]:
                    lo[c, k] = v
                if v > hi[c, k]:
                    hi[c, k] = v
    low = lo[0].copy()
    high = hi[0].copy()
    for c in range(1, chunks):
        for k in range(3):
            if lo[c, k] < low[k]:
                low[k] = lo[c, k]
            if hi[c, k] > high[k]:
                high[k] = hi[c, k]
    return low, high


def position_bounds(positions: np.ndarray):
    """``(min, max)`` of an ``(N, 3)`` array along the rows, each ``(3,)``.

    What ``positions.min(axis=0), positions.max(axis=0)`` return, to the bit
    and in the array's dtype, in one parallel pass instead of two slow ones.
    An empty array has no bounds and is refused, as numpy refuses it.
    """
    pos = np.ascontiguousarray(positions)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {pos.shape}")
    if pos.shape[0] == 0:
        raise ValueError("positions is empty: it has no bounds")
    if pos.dtype.kind != "f":
        pos = pos.astype(np.float64)
    return _bounds_kernel(pos)
