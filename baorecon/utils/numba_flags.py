"""Compilation flags shared by the catalogue kernels.

``FASTMATH`` is the rounding-preserving subset of numba's fast-math flags:
no NaN/Inf/signed-zero bookkeeping, but no contraction into FMAs and no
reassociation. Under the full set (``fastmath=True``) LLVM is free to fuse
and reorder the ``+``, ``-`` and ``*`` of a kernel, and it does so
differently for different specialisations of the same source (a kernel with
and without an ``origin`` argument moved ~2 % of the density cells by ~1e-13
at float64) and even from one compilation of the same source to the next
(the shift and rotation kernels moved ~0.1-10 % of the float32 positions by
one ulp whenever they were rebuilt, which numba's on-disk cache hid until
then). With these flags a kernel computes what its source spells out, the
same bits for every specialisation and every compilation, and runs as fast:
the catalogue kernels are bound by scattered memory access, not arithmetic.

Used by every kernel that reads or writes per-object quantities (mass
assignment, read-out, frame rotations, shifts). The grid kernels (multigrid
smoothers, FFT radial stream, line-of-sight projection) keep ``fastmath=True``.
"""

FASTMATH = {"nnan", "ninf", "nsz"}
