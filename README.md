
# 🌌 baorecon

`baorecon` is a Python package for BAO reconstruction in cosmology. It builds density fields from tracer catalogs, solves for the displacement field, and shifts objects to reconstruct large-scale structure using the Zel'dovich approximation.

[![CI Status](https://github.com/EdoardoMaragliano/baorecon/actions/workflows/run-tests.yml/badge.svg)](https://github.com/EdoardoMaragliano/baorecon/actions/workflows/run-tests.yml)

## ✨ Why `baorecon`?

In contrast to many existing implementations, `baorecon` is designed to be **lightweight, transparent, and highly modular**, stepping away from black-box C++ wrappers:

* 🐍 **100% Pure Python:** Written entirely in Python with no C++ wrapper layer or complex compilation steps required.
* ⚡ **JIT & GPU Accelerated:** Heavy mesh operations are extremely fast thanks to `njit`-compiled CPU kernels and `numba.cuda` GPU kernels.
* 🪶 **Lightweight Dependencies:** Relies on common, standard scientific Python packages (NumPy, SciPy) rather than specialized or heavy external frameworks.
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

For FITS catalogs and survey-style workflows, use the YAML-driven pipeline in `baorecon/pipeline`.

The pipeline covers:

1. loading data and random FITS catalogs
2. reading a YAML config file
3. selecting coordinate, weight, and ID columns
4. converting RA/DEC/redshift to Cartesian coordinates
5. running `BAOReconstructor`
6. preserving IDs through optional masking steps
7. converting reconstructed coordinates back to RA/DEC/redshift
8. saving the results to a tokenized output folder

See [baorecon/pipeline/README.md](baorecon/pipeline/README.md) and [examples/bao_pipeline_example.yaml](examples/bao_pipeline_example.yaml) for the full workflow.

## Installation

You can install dependencies by scope.

Runtime only (package core):

```bash
pip install -r requirements/runtime.txt
```

Test dependencies:

```bash
pip install -r requirements/runtime.txt -r requirements/test.txt
```

Notebook / examples dependencies:

```bash
pip install -r requirements/runtime.txt -r requirements/notebook.txt
```

Full local development environment:

```bash
pip install -r requirements.txt
```

If you install the package via `setup.py`, extras are also available:

```bash
pip install .
pip install ".[test]"
pip install ".[notebook]"
```

### GPU support (optional)

The density assignment, FFT-based displacement/potential solver, and field
interpolation can run on a CUDA GPU via [CuPy](https://cupy.dev/). This is
enabled by setting `device: "gpu"` in the `reconstruction` section of the
pipeline YAML config (see
[baorecon/pipeline/README.md](baorecon/pipeline/README.md)). The
`multigrid` solver always runs on CPU regardless of this setting.

To install the GPU extra (defaults to CuPy for CUDA 12.x; edit
`requirements/gpu.txt` for CUDA 11.x):

```bash
pip install -r requirements/runtime.txt -r requirements/gpu.txt
# or, via setup.py extras:
pip install ".[gpu]"
```

GPU support requires a CUDA-enabled GPU and a CuPy build matching your
CUDA toolkit version. If CuPy is not installed or no GPU is detected,
`device: "cpu"` is used and GPU-only tests are skipped automatically.

## Documentation map

- `baorecon/reconstruction/`: orchestrator (`BAOReconstructor`) and density preparation (`DensityManager`)
- `baorecon/solvers/`: FFT (`fft/{cpu,gpu}.py`) and multigrid (`multigrid/`) displacement solvers behind the shared `PoissonSolver` interface
- `baorecon/mas/`: mass assignment (`assign`) and read-out (`readout`), CPU/GPU kernels
- `baorecon/field_ops/`: mesh field operations (divergence, smoothing, interpolation), CPU/GPU split
- `baorecon/mesh/README.md`: mesh geometry; `mesh/los.py` holds the line-of-sight strategies
- `baorecon/pipeline/README.md`: YAML-driven catalog pipeline and output flow
- `baorecon/utils/README.md`: formatting, logging, backend selection, and utility helpers

## License

This package is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
