# Utils

This package groups small helper modules shared across the reconstruction pipeline.

## Responsibilities

- format user-facing inputs into a consistent internal representation
- provide logging setup and reusable utility helpers
- keep non-physics glue code separate from solver and mesh logic

## Main modules

- `formatters.py`: normalization/validation helpers for box size, box centre, weights,
  MAS, nmesh, reconstruction type, and RSD-space flags, plus grid-sizing helpers
  (`round_to_multigrid_friendly`, `nmesh_boxsize_from_cellsize`,
  `set_boxsize_from_positions`, `survey_to_box_frame`).
- `backend.py`: computational backend selection for the FFT solvers — the `FFTBackend`
  container and `get_fft_backend` factory that pick between the CPU (numpy/scipy) and
  GPU (CuPy) backends; `CUPY_AVAILABLE` is the single source of truth for GPU availability.
  Also exposes `PYFFTW_AVAILABLE` / `use_pyfftw()`, which gate the opt-in in-place CPU
  FFT backend on `BAORECON_FFT=pyfftw` (see [../../docs/pyfftw_backend.md](../../docs/pyfftw_backend.md)).
- `coordinates.py`: cosmology construction (`create_cosmology`) and RA/DEC/redshift ↔
  Cartesian conversions built on Astropy.
- `frames.py`: the cone-aligned reference frame (`ConeFrame`) — the rotation taking a
  survey's mean line of sight onto the `z` axis, and back. A pure rotation about the
  origin, so it applies unchanged to positions and to displacement fields; carries a
  working dtype and never casts the catalogue.
- `loggers.py`: logger configuration shared across modules (`setup_logger`).
- `utils.py`: general-purpose numerical helpers (rho/delta conversion, smoothing-radius
  conversions, periodic distances, box splitting).
- `mock_generator.py`: synthetic field/catalog generation (Gaussian and lognormal maps,
  Poisson sampling).

## Aligning a survey cone by hand

`ReconstructionPipeline` does this for you behind `reconstruction.align_cone`.
Outside the pipeline — driving `BAOReconstructor` or `reconstruct_positions`
yourself — it is four lines, and `ConeFrame` is public for exactly that.

Build the frame from the **randoms**: their angular density follows the survey
selection function, so the geometry does not move with the data's cosmic
variance. Data and randoms must share one frame in any case.

```python
import numpy as np
from baorecon import reconstruct_positions
from baorecon.utils.frames import ConeFrame

frame = ConeFrame.from_positions(random_pos)      # or .from_angles(ra, dec)

data_rec, random_rec = reconstruct_positions(
    frame.rotate(data_pos),
    frame.rotate(random_pos),
    f=0.8, bias=1.8, cellsize=8.0, smoothing=15.0,
    los=None,                       # local radial LOS: equivariant under rotation
)

# Rotating back is YOURS to do. Nothing downstream checks the frame, and a
# catalogue left in the cone frame looks entirely normal -- until you convert it
# to RA/DEC and find the footprint sitting on the pole.
#
# .astype() because the reconstructor works at its own precision (float32 by
# default) and hands back arrays in it, while the frame here was built from
# float64 positions. The frame refuses a mismatched array rather than casting a
# full-size one, so move the frame's nine numbers instead.
back = frame.astype(data_rec.dtype)
data_rec = back.unrotate(data_rec)
random_rec = back.unrotate(random_rec)
```

What it buys is a **compact bounding box**, not a narrower cone: the angular
radius is the survey's size on the sky and no axis can shrink it. On a Flagship
z1 cone the radial extent collapses from 2190 to 718 Mpc/h and the box volume
drops 1.87x, so the same cell size needs about half the cells. A local line of
sight is unaffected — it rotates with the tracers — and measuring a 2PCF before
and after a rigid rotation gives identical pair counts.

A few things worth knowing:

- **It is a pure rotation about the origin**, so it applies unchanged to
  positions and to displacement fields: `frame.rotate(psi)` is correct. Contrast
  `survey_to_box_frame`, which also *translates* — applying that to a
  displacement would be wrong.
- **`rotate(a, out=a)` works in place**, which on a 50-million-row random
  catalogue avoids a second multi-hundred-megabyte buffer.
- **The frame carries a working dtype** (default `float32`; `from_positions`
  follows the catalogue) and refuses a mismatched array rather than casting it.
  Watch the round trip: `BAOReconstructor` works at *its* `dtype` — `float32`
  unless you say otherwise — so a frame built from a float64 catalogue will not
  accept what comes back. `frame.astype(...)` moves the frame's nine numbers,
  which is the cheap side of the exchange.
- **After the rotation the axis is `+z`**, which in RA/DEC is the *pole*:
  `DEC = +90`, with RA degenerate. It does not land on `(0, 0)` — that would be
  `+x`. Sanity-check the alignment on the Cartesian positions (the mean direction
  should have `z ≈ 1`), not on the angles.
- **`los='z'` after aligning** means plane-parallel along the cone axis. That is
  a poor approximation for a cone tens of degrees wide; `frame.angular_radius(pos)`
  reports the half-opening angle to judge it by.
- **`to_dict()`/`from_dict()`** carry the matrix into a run's metadata. Anything
  defined on the solver mesh — density, potential, displacement grids — stays in
  the cone frame whatever you do with the catalogues, and cannot be put back on
  the sky without it.

## Notes

- These helpers are meant to keep the public API consistent and reduce repeated input handling in the higher-level layers.