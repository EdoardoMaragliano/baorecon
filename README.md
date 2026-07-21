
# 🌌 baorecon

`baorecon` is a Python package for BAO reconstruction in cosmology. It builds density fields from tracer catalogs, solves for the displacement field, and shifts objects to reconstruct large-scale structure using the Zel'dovich approximation.

[![CI Status](https://github.com/EdoardoMaragliano/baorecon/actions/workflows/run-tests.yml/badge.svg)](https://github.com/EdoardoMaragliano/baorecon/actions/workflows/run-tests.yml)
[![codecov](https://codecov.io/gh/EdoardoMaragliano/baorecon/graph/badge.svg)](https://codecov.io/gh/EdoardoMaragliano/baorecon)

## ✨ Why `baorecon`?

In contrast to many existing implementations, `baorecon` is designed to be **lightweight, transparent, and highly modular**, stepping away from black-box C++ wrappers:

* 🐍 **100% Pure Python:** Written entirely in Python with no C++ wrapper layer or complex compilation steps required.
* ⚡ **JIT & GPU Accelerated:** Heavy mesh operations are extremely fast thanks to `njit`-compiled CPU kernels and `numba.cuda` GPU kernels.
* 🪶 **Lightweight Dependencies:** Relies on common, standard scientific Python packages (NumPy, SciPy, pandas) rather than specialized or heavy external frameworks.
* 🧩 **Highly Modular API:** Provides a one-line high-level API entry point, a survey-ready YAML-driven pipeline, and a modular structure that allows you to use individual components (e.g., solvers, density managers) independently.
* 🧮 **Multiple Solvers:** Includes both FFT-based solvers and a Multigrid backend. *(Note: The multigrid backend currently includes standard Jacobi smoothing as well as a semi-experimental V-cycle solver based on multi-color Gauss-Seidel).*

## ⚙️ How it works

The current pipeline is organized in three modular layers:

1. `DensityManager` prepares the input catalogs, infers or normalizes box parameters, applies mass assignment, and produces the mesh overdensity field.
2. A solver (`FFTSolverCPU`/`FFTSolverGPU`, or `MultigridSolver`) consumes the mesh field and exposes lazy `potential` and `displacement` properties.
3. The high-level BAO orchestrator (`BAOReconstructor`, or the one-call `reconstruct_positions`) combines both steps and returns the reconstructed catalogs.

The numerical core follows the [Burden, Percival & Howlett (2015)](https://arxiv.org/abs/1504.02591) IFFT reconstruction scheme or the [Martin White MultiGrid scheme](https://github.com/martinjameswhite/recon_code), with JIT-compiled kernels for the heavy mesh operations.

## Main data flow

At a high level, the reconstruction works like this:

1. Data and random positions are passed to `BAOReconstructor`.
2. `DensityManager` formats the inputs, applies the chosen mass-assignment scheme, and builds `delta_on_mesh`.
3. A solver (`FFTSolverCPU`/`FFTSolverGPU`, or `MultigridSolver`) computes the scalar potential and displacement field on the mesh.
4. The orchestrator interpolates or projects the displacement field back to the tracers.
5. The shifted catalogs are returned in the requested reconstruction mode.

## Basic usage

For a hands-on, end-to-end walkthrough — generating a mock catalog, running
`BAOReconstructor`, and inspecting the overdensity, potential and displacement
fields — see the annotated notebook
[examples/bao_reconstructor_walkthrough.ipynb](examples/bao_reconstructor_walkthrough.ipynb).

### High-level reconstruction

The quickest entry point is the functional API:

```python
import numpy as np
from baorecon import reconstruct_positions

data_pos = np.random.rand(10000, 3) * 1000.0
random_pos = np.random.rand(10000, 3) * 1000.0

shifted_galaxies, shifted_randoms = reconstruct_positions(
    data_pos, random_pos, f=0.78, bias=1.5,
    nmesh=256, smoothing=15.0, los="z", device="cpu",
)
```

For full control use `BAOReconstructor` directly:

```python
import numpy as np
from baorecon import BAOReconstructor

data_pos = np.random.rand(10000, 3) * 1000.0
random_pos = np.random.rand(10000, 3) * 1000.0
boxsize = 1000.0
boxcentre = np.array([boxsize / 2.0] * 3)

reconstructor = BAOReconstructor(
    data_pos=data_pos,
    random_pos=random_pos,
    RSDspace="RedshiftSpace",
    nmesh=256,            # scalar (cubic) or a length-3 array (anisotropic)
    boxsize=boxsize,      # scalar (cubic) or a length-3 array (rectangular)
    boxcentre=boxcentre,
    los="z",              # 'x'/'y'/'z' (plane-parallel) or None (local/radial)
    R_sm=15.0,
    pbc=False,
    rectype="rec-sym",
    f=0.78,
    bias=1.5,
    MAS="CIC",
    solver_type="ifft",   # or "multigrid"
    device="cpu",         # or "gpu"
)

shifted_galaxies, shifted_randoms = reconstructor.run_reconstruction()
```

Instead of `nmesh`/`boxsize` you may pass a target `cellsize` (mutually
exclusive with them); the per-axis grid is then derived from the catalogue
extent and rounded up to a multigrid-friendly size.

### Solver layer

If you already have `delta_on_mesh`, you can use the solvers directly. The
line-of-sight is an injected strategy:

```python
from baorecon.mesh.mesh import Mesh
from baorecon.mesh.los import FixedAxisLOS, LocalLOS
from baorecon.solvers.fft import FFTSolverCPU      # FFTSolverGPU for CuPy
from baorecon.solvers.multigrid import MultigridSolver

mesh = Mesh(nmesh=256, boxsize=1000.0, boxcentre=np.array([500.0, 500.0, 500.0]))
los = FixedAxisLOS(2)   # plane-parallel along z; or LocalLOS(...) for radial

fft_solver = FFTSolverCPU(delta_on_mesh, mesh, los=los, bias=1.5, RSDspace="RealSpace")
potential = fft_solver.potential
displacement = fft_solver.displacement

mg_solver = MultigridSolver(delta_on_mesh, mesh, los=los, bias=1.5, RSDspace="RealSpace")
potential_mg = mg_solver.potential
displacement_mg = mg_solver.displacement
```

## Pipeline

For survey-style workflows with FITS or Parquet catalogs, use the YAML-driven pipeline in `baorecon/pipeline`.

The pipeline covers:

1. loading data and random catalogs (FITS or Parquet)
2. reading a YAML config file
3. selecting coordinate, weight, and ID columns (with optional column pruning on read)
4. converting RA/DEC/redshift to Cartesian coordinates
5. running `BAOReconstructor`
6. preserving IDs through optional masking steps
7. converting reconstructed coordinates back to RA/DEC/redshift
8. saving the results (FITS or Parquet) to a tokenized output folder

The input format is inferred from the file extension (`.fits`/`.fit`,
`.parquet`/`.pq`) or set explicitly with `catalog.format`; the output format is
controlled by `output.format` (default `fits`).

See [baorecon/pipeline/README.md](baorecon/pipeline/README.md) and [examples/bao_pipeline_example.yaml](examples/bao_pipeline_example.yaml) for the full workflow.

## Working precision

Catalogs and mesh arrays are held at a configurable floating-point precision.
The default is **float32**, which roughly halves the memory footprint of the
(often multi-million-row) random catalog versus float64. `BAOReconstructor`
downcasts float64 inputs to the working precision automatically; pass
`dtype=np.float64` (or set `reconstruction.dtype: float64` in the pipeline YAML)
to opt into double precision end-to-end.

## Multithreading

The CPU reconstruction is parallel throughout: the heavy mesh kernels (mass
assignment, interpolation, divergence, multigrid smoothers, radial projection)
are `numba` `@njit(parallel=True)` loops, the FFTs run multithreaded, and NumPy
delegates to a threaded BLAS. Each of these has its **own** thread pool governed
by an environment variable — there is no single baorecon flag.

| Variable | Controls | Default if unset |
|---|---|---|
| `NUMBA_NUM_THREADS` | numba JIT kernels (MAS, interpolation, divergence, multigrid) | all cores |
| `OMP_NUM_THREADS` | OpenMP-based BLAS / library threads | all cores |
| `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` | NumPy linear-algebra BLAS backend | all cores |
| `NUMEXPR_NUM_THREADS` | numexpr, if present | all cores |
| `BAORECON_FFT_THREADS` | FFTW threads (only when `BAORECON_FFT=pyfftw`) | all cores |

Notes:

- The **default scipy FFT** always runs on all cores (`workers=-1`) and is not
  capped by any of these variables. To bound FFT threads, use the optional
  `pyfftw` backend (`BAORECON_FFT=pyfftw` + `BAORECON_FFT_THREADS`); see
  [docs/pyfftw_backend.md](docs/pyfftw_backend.md).
- On `device="gpu"` these variables are irrelevant (work runs on the GPU).

### ⚠️ Set them *before* importing anything

All of these are read **once, at import/startup time** (`NUMBA_NUM_THREADS` when
`numba` is imported, the BLAS caps when NumPy loads its backend). Setting them
after `import numpy` / `import baorecon` has **no effect**. There are three safe
ways to do it — pick one:

**1. Inline, in front of the command (recommended for one-off runs):**

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
NUMBA_NUM_THREADS=8 python your_run.py
```

**2. `export` in the shell / job script (e.g. a SLURM submit script):**

```bash
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMBA_NUM_THREADS=8
# export BAORECON_FFT=pyfftw BAORECON_FFT_THREADS=8   # optional pyfftw path
python your_run.py
```

**3. `os.environ` at the very top of your script — before any scientific import:**

```python
import os
THREADS = "8"
os.environ["NUMBA_NUM_THREADS"]    = THREADS
os.environ["OMP_NUM_THREADS"]      = THREADS
os.environ["OPENBLAS_NUM_THREADS"] = THREADS
os.environ["MKL_NUM_THREADS"]      = THREADS
os.environ["NUMEXPR_NUM_THREADS"]  = THREADS
# os.environ["BAORECON_FFT_THREADS"] = THREADS   # optional pyfftw path

import numpy as np                 # imports MUST come AFTER the lines above
from baorecon import BAOReconstructor
```

(The scripts in `benchmarks/` use this last pattern — see the header of
[benchmarks/bench_bao_reconstructor.py](benchmarks/bench_bao_reconstructor.py).)

Leaving the variables unset lets every backend use all available cores, which is
usually what you want on a dedicated node; set them explicitly on shared or
SLURM-managed machines to stay within your allocation.

## Installation

Install from PyPI:

```bash
pip install baorecon
```

Or from a local checkout, with optional extras:

```bash
pip install .                            # runtime only
pip install ".[test]"                    # + test suite
pip install ".[notebook]"                # + notebook/example dependencies
pip install ".[gpu]"                     # + CUDA GPU backend (see below)
pip install ".[docs]"                    # + Sphinx docs build
pip install -e ".[test,notebook,docs]"   # editable dev install, combined extras
```

### GPU support (optional)

The density assignment, FFT-based displacement/potential solver, and field
interpolation can run on a CUDA GPU via [CuPy](https://cupy.dev/). This is
enabled by setting `device: "gpu"` in the `reconstruction` section of the
pipeline YAML config (see
[baorecon/pipeline/README.md](baorecon/pipeline/README.md)). The
`multigrid` solver always runs on CPU regardless of this setting.

To install the GPU extra (defaults to CuPy for CUDA 12.x):

```bash
pip install ".[gpu]"
```

For CUDA 11.x, install the matching CuPy build afterwards instead:

```bash
pip install cupy-cuda11x
```

GPU support requires a CUDA-enabled GPU and a CuPy build matching your
CUDA toolkit version. If CuPy is not installed or no GPU is detected,
`device: "cpu"` is used and GPU-only tests are skipped automatically.

For the CUDA environment variables (`CUDA_PATH`/`CUDA_HOME`) that CuPy
needs — and how to fix a `cannot open source file "cuda_fp16.h"`
compile error — see [docs/gpu_environment.md](docs/gpu_environment.md).

### Faster FITS reads (optional)

Installing [`fitsio`](https://github.com/esheldon/fitsio) enables true
column-subset reads for FITS catalogs, so only the configured columns are
pulled off disk. Without it, FITS reads fall back to Astropy and prune columns
in memory. Parquet reads always push column selection down to the reader and
require `pyarrow` (a core `baorecon` dependency).

### Low-memory CPU FFT backend (optional)

The iterative FFT (iFFT) solver has an opt-in in-place CPU backend built on
[`pyfftw`](https://pyfftw.readthedocs.io) that transforms in place instead of
allocating a fresh array per `rfftn`/`irfftn`. Enable it with a single
environment variable — no code or config changes:

```bash
BAORECON_FFT=pyfftw python your_run.py
```

If `pyfftw` is not installed the run transparently falls back to scipy (the
default). The radial line-of-sight projection is streamed on both CPU paths
regardless of this setting. See [docs/pyfftw_backend.md](docs/pyfftw_backend.md)
for thread control, FFTW planning/wisdom, and measured memory savings.

## Documentation map

- `baorecon/reconstruction/`: orchestrator (`BAOReconstructor`) and density preparation (`DensityManager`)
- `baorecon/solvers/`: FFT (`fft/{cpu,gpu}.py`) and multigrid (`multigrid/`) displacement solvers behind the shared `PoissonSolver` interface. The FFT solvers stream the radial line-of-sight projection (`fft/_radial_stream.py`) to keep peak memory low
- `baorecon/mas/`: mass assignment (`assign`) and read-out (`readout`), CPU/GPU kernels
- `baorecon/field_ops/`: mesh field operations (divergence, smoothing, interpolation), CPU/GPU split
- `baorecon/mesh/README.md`: mesh geometry; `mesh/los.py` holds the line-of-sight strategies
- `baorecon/io/`: catalog I/O with pluggable FITS/Parquet backends (`io/backends/`), YAML config parsing, and output naming
- `baorecon/pipeline/README.md`: YAML-driven catalog pipeline and output flow
- `baorecon/utils/README.md`: formatting, logging, backend selection, and utility helpers
- [docs/pyfftw_backend.md](docs/pyfftw_backend.md): optional in-place, low-memory CPU FFT backend (`BAORECON_FFT=pyfftw`)
- `benchmarks/README.md`: profiling scripts comparing baorecon (CPU/GPU) against pyrecon

## License

This package is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
