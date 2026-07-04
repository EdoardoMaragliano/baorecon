# GPU environment setup (`CUDA_PATH` / `CUDA_HOME`)

The GPU backend runs mesh operations through two independent libraries:

- **CuPy** — Gaussian smoothing, read-out, and the FFT solver. CuPy
  compiles kernels at runtime with NVRTC, which needs the **CUDA toolkit
  headers**. CuPy locates them via the `CUDA_PATH` environment variable.
- **`numba.cuda`** — mass assignment and other elementwise kernels. Numba
  locates CUDA on its own (driver + bundled/`nvidia-*` wheel headers) and
  does **not** use `CUDA_PATH`.

Because of this split, a broken `CUDA_PATH` breaks the CuPy paths while
the Numba paths keep working — a useful diagnostic signal (see below).

## Required environment variables

Point `CUDA_PATH` (and, for consistency, `CUDA_HOME`) at an installed
CUDA toolkit whose `include/` directory contains `cuda_fp16.h`:

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export CUDA_PATH=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
```

Set `NUMBA_CUDA_USE_NVIDIA_BINDING=1` if you want Numba to use the
`nvidia-*` CUDA wheels instead of a system toolkit.

If you only want to steer CuPy (leaving `CUDA_PATH` alone for other
tools), use `CUPY_CUDA_PATH` instead:

```bash
export CUPY_CUDA_PATH=/usr/local/cuda-12.6
```

### Host-portable export (shared home directories)

If the same shell profile runs on several hosts with different CUDA minor
versions, hardcoding one path is fragile. Auto-select the newest
`cuda-12.x` present on the host (avoiding any `cuda-13.x`), with a
fallback to the generic `/usr/local/cuda` symlink:

```bash
# Newest CUDA 12.x on this host; fall back to the generic symlink.
_cuda=$(ls -d /usr/local/cuda-12.* 2>/dev/null | sort -V | tail -n1)
export CUDA_HOME="${_cuda:-/usr/local/cuda}"
export CUDA_PATH="$CUDA_HOME"
unset _cuda
export PATH="$CUDA_HOME/bin:$PATH"
```

## Where to put these

- **Interactive shells:** add the exports to `~/.bashrc` (or your shell's
  rc file).
- **Conda environments:** put them in
  `$CONDA_PREFIX/etc/conda/activate.d/gpu_env.sh` so they are set only
  when the environment is active.

## Troubleshooting: `cannot open source file "cuda_fp16.h"`

A CuPy `NVRTCError` / `CompileException` such as:

```
catastrophic error: cannot open source file "cuda_fp16.h"
```

means NVRTC cannot find the CUDA toolkit headers — almost always because
`CUDA_PATH` points at a directory that does not exist or lacks
`include/cuda_fp16.h`.

**Tell-tale sign:** GPU tests that use `numba.cuda` (e.g. mass
assignment) pass, while CuPy paths (smoothing, read-out) fail — because
Numba does not rely on `CUDA_PATH`.

Diagnose and fix:

```bash
# 1. See where CuPy is looking:
echo "$CUDA_PATH"

# 2. Check whether the header is actually there:
ls "$CUDA_PATH/include/cuda_fp16.h"

# 3. If missing, find a toolkit that has it:
ls -d /usr/local/cuda-*/include/cuda_fp16.h

# 4. Point CUDA_PATH at that toolkit (see the exports above) and re-run:
python -m pytest tests/test_density_manager.py
```
