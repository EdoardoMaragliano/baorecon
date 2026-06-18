
# Zeldareco

`zeldareco` is a Python package for BAO reconstruction in cosmology. It builds density fields from tracer catalogs, solves for the displacement field, and shifts objects to reconstruct large-scale structure using the Zel'dovich approximation.

[![CI Status](https://github.com/EdoardoMaragliano/baorecon/actions/workflows/run-tests.yml/badge.svg)](https://github.com/EdoardoMaragliano/baorecon/actions/workflows/run-tests.yml)

## What the package does

The current pipeline is organized in three layers:

1. `DensityManager` prepares the input catalogs, infers or normalizes box parameters, applies mass assignment, and produces the mesh overdensity field.
2. A solver (`FFTSolver` or `MultigridSolver`) consumes the mesh field and exposes lazy `potential` and `displacement` properties.
3. The high-level BAO orchestrator combines both steps and returns the reconstructed catalogs.

The numerical core follows the Burden, Percival and Howlett (2015) reconstruction scheme, with JIT-compiled kernels for the heavy mesh operations.

## Main data flow

At a high level, the reconstruction works like this:

1. Data and random positions are passed to `BAOReconstructor`.
2. `DensityManager` formats the inputs, applies the chosen mass-assignment scheme, and builds `delta_on_mesh`.
3. A solver (`FFTSolver` or `MultigridSolver`) computes the scalar potential and displacement field on the mesh.
4. The orchestrator interpolates or projects the displacement field back to the tracers.
5. The shifted catalogs are returned in the requested reconstruction mode.

## Basic usage

### High-level reconstruction

```python
import numpy as np
from zeldareco import BAOReconstructor

data_pos = np.random.rand(10000, 3) * 1000.0
random_pos = np.random.rand(10000, 3) * 1000.0
boxsize = 1000.0
boxcentre = np.array([boxsize / 2.0] * 3)

reconstructor = BAOReconstructor(
    data_pos=data_pos,
    random_pos=random_pos,
    RSDspace="RedshiftSpace",
    nmesh=256,
    boxsize=boxsize,
    boxcentre=boxcentre,
    los="z",
    R_sm=15.0,
    pbc=False,
    rectype="rec-sym",
    f=0.78,
    bias=1.5,
    dtype=np.float64,
    MAS="CIC",
    solver_type="ifft",
)

shifted_galaxies, shifted_randoms = reconstructor.run_reconstruction()
```

### Solver layer

If you already have `delta_on_mesh`, you can use the solvers directly:

```python
from zeldareco.displacement_solver.fft_solver import FFTSolver
from zeldareco.displacement_solver.multigrid_solver import MultigridSolver
from zeldareco.mesh.mesh import Mesh

mesh = Mesh(nmesh=256, boxsize=1000.0, boxcentre=np.array([500.0, 500.0, 500.0]), los="z")

fft_solver = FFTSolver(delta_on_mesh, mesh, bias=1.5, RSDspace="RealSpace")
potential = fft_solver.potential
displacement = fft_solver.displacement

mg_solver = MultigridSolver(delta_on_mesh, mesh, bias=1.5, RSDspace="RealSpace")
potential_mg = mg_solver.potential
displacement_mg = mg_solver.displacement
```

## Pipeline

For FITS catalogs and survey-style workflows, use the YAML-driven pipeline in `zeldareco/pipeline`.

The pipeline covers:

1. loading data and random FITS catalogs
2. reading a YAML config file
3. selecting coordinate, weight, and ID columns
4. converting RA/DEC/redshift to Cartesian coordinates
5. running `BAOReconstructor`
6. preserving IDs through optional masking steps
7. converting reconstructed coordinates back to RA/DEC/redshift
8. saving the results to a tokenized output folder

See [zeldareco/pipeline/README.md](zeldareco/pipeline/README.md) and [examples/bao_pipeline_example.yaml](examples/bao_pipeline_example.yaml) for the full workflow.

## Installation

Requires Python >= 3.10.

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

## Documentation map

- `zeldareco/BAOreconstruction/README.md`: orchestrator and density preparation
- `zeldareco/displacement_solver/README.md`: solver layer and lazy properties
- `zeldareco/mesh/README.md`: mesh geometry and helper methods
- `zeldareco/pipeline/README.md`: YAML-driven catalog pipeline and output flow
- `zeldareco/utils/README.md`: formatting, logging, and utility helpers

## License

This package is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
