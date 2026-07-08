# reconstruction

The core of BAO reconstruction: builds the overdensity field, solves for the
displacement field, and shifts the data/random catalogues.

- `DensityManager` (`density.py`) — paints the catalogues onto the mesh and
  builds the overdensity field `delta`. It is solver-agnostic and centralizes
  box/position/weight handling.
- `BAOReconstructor` (`bao_reconstructor.py`) — the orchestrator. It resolves
  the device (CPU/GPU), the solver class (iFFT or multigrid) and the
  line-of-sight strategy once, then runs the reconstruction.

```python
from baorecon import BAOReconstructor

recon = BAOReconstructor(
    data_pos=data_pos,        # (N, 3) Cartesian
    random_pos=random_pos,    # (M, 3) Cartesian
    nmesh=256,
    boxsize=None,             # if None, derived from the random extent
    los=None,                 # None -> local radial LOS; 'x'/'y'/'z' -> plane-parallel
    f=0.8, bias=1.8,
    RSDspace="RedshiftSpace",
    solver_type="ifft",       # "ifft" (Burden) or "multigrid"
    device="cpu",
)
data_rec, random_rec = recon.run_reconstruction()
```

For a hands-on, end-to-end walkthrough — building a mock catalogue, running
`BAOReconstructor`, and inspecting the overdensity, potential and displacement
fields — see the annotated notebook
[../../examples/bao_reconstructor_walkthrough.ipynb](../../examples/bao_reconstructor_walkthrough.ipynb).

## Positions must be in Cartesian coordinates

`data_pos` and `random_pos` are `(N, 3)` arrays of **Cartesian coordinates**
`(x, y, z)`, with the observer at the **origin** `(0, 0, 0)` of the reference
frame. Units must be consistent with each other and with `boxsize`,
`boxcentre`, `padding`, `R_sm` (typically **Mpc/h**).

The code does **not** accept `RA / DEC / REDSHIFT` directly: those must first be
converted to Cartesian (see [Working with RA/DEC/REDSHIFT](#working-with-radecredshift)).

Weights (`data_weights`, `random_weights`) are optional, of shape `(N,)`; if
omitted they are taken to be unity.

## Line of sight (LOS)

The line of sight sets the direction along which RSD effects are applied. There
are two strategies, selected by the `los` parameter:

- **`los='x'`, `'y'` or `'z'`** → fixed **plane-parallel** LOS along a Cartesian
  axis (`FixedAxisLOS`). All tracers share the same direction.
- **`los=None`** → **local radial** LOS (`LocalLOS`): at every point the
  direction is the radial versor pointing **from the origin (observer)** towards
  that point.

> **The local LOS is radial with respect to the origin.** For it to be computed
> correctly, every tracer must sit at its **true comoving distance** from the
> observer at `(0, 0, 0)`. For a survey catalogue converted from
> `RA/DEC/REDSHIFT` this is automatic: the observer is already at the origin.
> For a **snapshot** it is not — see below.

## Fixed-redshift snapshot: the box must be shifted

A snapshot is a periodic box at fixed redshift, with positions typically in
`[0, L]³`, i.e. **right next to the origin**. If the local radial LOS
(`los=None`) is used with the box as-is, the radial versors from the origin fan
out across the box and the LOS becomes **physically meaningless**.

For the LOS to be computed correctly, **the box must be translated so that its
centre lands at the comoving distance `d_c` corresponding to the mean redshift
of the snapshot**. Then the radial versors across the box become nearly parallel
(plane-parallel limit) and correctly represent the LOS.

```python
import numpy as np
from baorecon import BAOReconstructor
from baorecon.utils.coordinates import create_cosmology, comoving_distance

cosmo = create_cosmology(H0=67.11, Om0=0.3175)

BOXSIZE = 1000.0     # Mpc/h
Z_MEAN  = 1.0        # snapshot redshift

# comoving distance of the mean redshift, in the same units as the positions.
# comoving_distance() returns Mpc; multiply by h to get Mpc/h.
d_c = comoving_distance(Z_MEAN, cosmo) * cosmo.h        # Mpc/h

# box in [0, L]^3 (centre at L/2): move the centre to d_c along the LOS axis (z).
shift = np.array([0.0, 0.0, d_c - BOXSIZE / 2.0], dtype=data_pos.dtype)
data_pos_shifted   = data_pos   + shift
random_pos_shifted = random_pos + shift

recon = BAOReconstructor(
    data_pos=data_pos_shifted,
    random_pos=random_pos_shifted,
    nmesh=256,
    boxsize=BOXSIZE,
    los=None,                 # local radial LOS: now correct
    f=0.8, bias=1.8,
    RSDspace="RedshiftSpace",
    pbc=True,
)
data_rec, random_rec = recon.run_reconstruction()

# the reconstructed positions are in the shifted frame: subtract the same shift
# to bring them back to the original snapshot frame.
data_rec   -= shift
random_rec -= shift
```

> **Alternative without a shift.** For a snapshot one often uses the fixed
> **plane-parallel** LOS directly (`los='z'`): it is the natural choice for a
> fixed-redshift box and requires **no** translation. The shift described above
> is only needed when using the local radial LOS (`los=None`) on a snapshot.

## Working with RA/DEC/REDSHIFT

If the catalogues are in `RA / DEC / REDSHIFT` coordinates there are two routes.

### 1. Use the pipeline (recommended for survey workflows)

`baorecon.pipeline.ReconstructionPipeline` is YAML-driven and handles I/O, the
`RA/DEC/z → xyz` conversion, reconstruction, and the `xyz → RA/DEC/z`
back-conversion for you. See [../pipeline/README.md](../pipeline/README.md).

### 2. Convert manually with `utils/coordinates`

Alternatively, convert the coordinates explicitly — in the same style as the
pipeline — and call `BAOReconstructor` directly. In a survey catalogue the
observer is already at the origin, so **no shift is needed**.

```python
from baorecon import BAOReconstructor
from baorecon.utils.coordinates import (
    create_cosmology, radec_z_to_xyz, xyz_to_radec_z,
)

cosmo = create_cosmology(H0=67.11, Om0=0.3175)

# RA/DEC/REDSHIFT -> Cartesian (Mpc/h). The observer is at the origin.
data_xyz, _   = radec_z_to_xyz(data_ra,   data_dec,   data_z,
                               cosmo=cosmo, distance_unit="Mpc/h")
random_xyz, _ = radec_z_to_xyz(random_ra, random_dec, random_z,
                               cosmo=cosmo, distance_unit="Mpc/h")

recon = BAOReconstructor(
    data_pos=data_xyz,
    random_pos=random_xyz,
    nmesh=256,
    los=None,                 # local radial LOS: correct without a shift
    f=0.8, bias=1.8,
    RSDspace="RedshiftSpace",
    solver_type="ifft",
    pbc=False,                # survey geometry (non-periodic)
)
data_rec_xyz, random_rec_xyz = recon.run_reconstruction()

# Cartesian -> RA/DEC/REDSHIFT
data_ra_rec, data_dec_rec, data_z_rec, _ = xyz_to_radec_z(
    data_rec_xyz, cosmo=cosmo, distance_unit="Mpc/h",
)
```

## Notes

- `boxsize` / `boxcentre`: if not provided, they are derived from the random
  catalogue extent (`boxcentre` = midpoint, `boxsize` = extent + `padding`).
  As an alternative to `nmesh`, pass `cellsize` (mutually exclusive with
  `nmesh`/`boxsize`) to fix the cell resolution.
- `pbc`: `True` for a periodic box (snapshot); `False` for a survey geometry
  (enforces non-periodic `padding` and validates that positions fall inside the box).
- `rectype`: `"rec-sym"` (symmetric shift of data and randoms) or `"rec-iso"`.
- `dtype`: working precision, `float32` by default (`float64` optional). The GPU
  path always works in `float32`.
- The reconstruction physics stays in this package; `baorecon.pipeline` is only a
  thin wrapper that adds I/O and coordinate conversions.
