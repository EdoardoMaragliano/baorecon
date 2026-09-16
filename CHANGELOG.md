# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Test coverage for three things that had none. `rec-iso` was never constructed
  anywhere in the suite although it takes its own branch in `_shift_randoms`;
  it is now covered by what *distinguishes* it from `rec-sym` — in redshift space
  the two must give identical data and differing randoms, in real space identical
  randoms — so a dead branch would fail rather than pass. `reconstruct_positions`,
  the one-line API the README leads with, is checked against an explicitly built
  `BAOReconstructor`. And the two pipeline implementations are pinned to each
  other stage by stage and on their written products; they had drifted to 86% vs
  39% coverage, which matters because every wiring change has to be made twice.
- `assign` and `readout` coverage across NGP/CIC/TSC, periodic and non-periodic,
  serial and chunked. The suite had leaned on CIC with `pbc=True` while survey
  runs are `pbc=False`; mass conservation is now asserted for particles sitting
  on the box faces, where the non-periodic clamp actually applies.

### Fixed
- `n_iterations` was accepted everywhere and ingested nowhere. `BAOReconstructor`
  declared `**kwargs` and never stored them, so the keyword — passed by
  `reconstruct_positions`, by `ReconstructionPipeline.build_reconstructor`, and
  documented as a parfile key in `examples/bao_pipeline_parfile.ini` — reached the
  constructor and stopped there. The iterative FFT solver reads `solver_args`, so
  it ran at its internal default of 3 whatever the caller or the parfile asked
  for. `n_iterations` is now an explicit `BAOReconstructor` argument merged into
  `solver_args` (an explicit `solver_args` entry still wins), and all three routes
  take effect. Results are unchanged for anyone who left it at the conventional 3.

  `BAOReconstructor` now **rejects** keywords it does not recognise instead of
  dropping them, which is what let this hide. The error names them and, where it
  can tell, says where they belong: solver options in `solver_args`, `align_cone`
  in `ReconstructionPipeline`. Callers passing extra keywords that never did
  anything will start seeing a `TypeError` — which is the point, since silently
  ignoring `align_cone=True` let a caller believe the catalogue had been aligned.

### Changed
- `reconstruct_positions` names `solver_type` (default `"multigrid"`, unchanged —
  note the example parfiles configure `"ifft"`, so a call here and a pipeline run
  do not pick the same solver unless one of them says so). `n_iterations` moves to
  `**kwargs`, where it now works; the signature had promoted a solver-specific
  option above the choice of solver itself.

### Fixed
- `interpolate_cic_vector` and `interpolate_tsc_vector` (CPU) read a single mesh
  size off axis 0 (`nmesh = field.shape[0]`) and used it for all three axes. The
  per-axis `boxsize` was honoured, the per-axis `nmesh` was not, so on any grid
  whose axes differ the cell size was wrong on two of them and the node indices
  were wrong with it — and where axis 0 is the longest, the non-periodic clamp
  ran past the end of a shorter axis, reading out of bounds (njit, no bounds
  checking). Both now take `nx`, `ny`, `nz` from `field.shape`.

  These are the read-out for the FFT solvers (`FFTSolverCPU.read_displacement_at`),
  so **every FFT reconstruction on a non-cubic mesh returned wrong displacements
  at the tracers**. Measured on a Flagship z1 run at `nmesh = (224, 320, 320)`:
  the median |Psi| went from 2.385 to 4.362 Mpc/h, and the FFT and multigrid
  read-outs went from a correlation of 0.04 to 1.00. Runs with a scalar `nmesh`
  are unaffected — the mesh is cubic then, and `field.shape[0]` is right for
  every axis — which is why this survived: a non-cubic `nmesh` only arises from
  `cellsize` or an explicit length-3 `nmesh`. The GPU kernels already derived the
  three sizes separately and were never affected.

  The existing rectangular-box tests varied `boxsize` while keeping `nmesh`
  cubic, the one rectangular case that cannot catch this. `tests/test_rectangular_box.py`
  now covers per-axis `nmesh`: interpolation of an exact analytic field on four
  grid shapes, and agreement between the FFT and multigrid *read-outs* at tracer
  positions (independent implementations — the FFT interpolates its displacement
  grid, the multigrid differentiates the potential).
- `tests/test_solver_equivalence.py` compared two zero fields. Its
  `_gaussian_bump` used `sigma = 0.05` on a unit-spaced grid, centred between
  nodes: sampled on the grid it peaked at 7e-66, both solvers returned a zero
  displacement, and `np.allclose(0, 0)` passed without comparing anything. The
  bump is now three cells wide and centred on a node, and a `_assert_not_vacuous`
  guard fails if either displacement collapses to zero again.

### Added
- `reconstruction.align_cone` (YAML and INI, default `false`) wires `ConeFrame`
  into both pipelines. The rotation lives in the coordinate layer, paired with
  `convert_to_xyz`/`convert_back` -- already the symmetric pair that puts a
  catalogue into the frame the solver works in and takes it back out -- so
  `BAOReconstructor` is untouched and its `boxsize` and `los` keep meaning what
  they meant, in the one frame it ever sees. The box is measured after the
  rotation, so it is the tighter one. The frame is built from the *randoms*, by
  whichever route the input path allows: from the angles for `ra_dec_z` (while
  they are still alive, before `convert_to_xyz` releases them) and from the
  positions for `cartesian`, which has no angles at all. `convert_back` also
  un-rotates the *pre*-reconstruction arrays while they live, since
  `_save_catalogs` forms the tracer displacements as `pos_xyz - rec_xyz` and
  that subtraction would otherwise straddle two frames. The matrix is written
  into the metadata sidecar, without which the saved grids -- which stay in the
  cone frame -- cannot be put back on the sky. Alignment is logged at INFO with
  the footprint centre in RA/DEC, and at DEBUG before and after: after the
  rotation the centre is at `DEC = +90` (the z axis is the pole; RA is degenerate
  there, and the centre does *not* land on RA/DEC = 0,0), reported with the
  residual misalignment.
- `baorecon.utils.frames.ConeFrame`: the rotation that aligns a survey cone's
  mean line of sight with the `z` axis, so an off-axis lightcone (up to ~150
  degrees from `z`) is left with only its own angular radius. The payoff is a
  compact bounding box -- the same mesh resolves the cone with fewer cells --
  while a local line of sight is unaffected, being equivariant under rotation.
  Built from sky coordinates (`from_angles`), from Cartesian positions
  (`from_positions`, the only route for catalogues that are already Cartesian),
  or from an explicit axis (`from_direction`, e.g. to share one geometry across
  redshift bins). Being a pure rotation about the origin it applies unchanged to
  positions and to displacement fields; `rotate(a, out=a)` works in place, which
  avoids a second multi-hundred-megabyte buffer on a large random catalogue.
  `to_dict`/`from_dict` carry the matrix into the run metadata, without which
  saved grids -- which live in the cone frame -- cannot be put back on the sky.
  `angular_radius` reports the cone's half-opening angle, the scale at which a
  fixed-axis line of sight stops being a good approximation.
  The frame carries a working `dtype` like the rest of the package (default
  `float32`; `from_positions` follows the catalogue) and never casts the
  catalogue: `rotate` refuses a mismatched array rather than allocating a
  full-size copy or silently changing the pipeline's precision, and `astype`
  moves the frame's nine numbers instead. Strided arrays are read in place, so
  passing a view does not materialise it.
  Storage precision and accumulation precision are kept distinct: the direction
  sum and the matrix construction are always float64 (three scalars and nine
  numbers, so no memory cost) before being cast down. Summing float32 unit
  vectors along axis 0 of an `(N, 3)` array skips numpy's pairwise summation and
  put the axis 23 degrees off in testing, silently.
  The matrix matches ZAtools' `Rotate_cone` elementwise, so replacing that
  implementation is a no-op. Not yet wired into `ReconstructionPipeline`.
- `CatalogConfig.from_ini` reads a 2PCF/Euclid-style INI parameter file, and
  `CatalogConfig.from_file` dispatches on the extension (`.ini`/`.par`/`.parfile`
  go to the INI loader, anything else to the YAML one), so
  `ReconstructionPipeline` takes either format with no call-site change. The
  `[Cosmology]` block keeps the 2PCF spelling verbatim so one block can serve
  both codes, and the loader translates. `examples/bao_pipeline_parfile.ini` is
  the annotated template; `tests/test_config_ini.py` asserts it stays equivalent
  to the YAML example.
- The pipeline accepts catalogues whose columns already are Cartesian x/y/z, via
  `coordinate_system.input = "cartesian"` (`CARTESIAN` in the parfile). The sky
  conversion is skipped in both directions, so no cosmology enters that path and
  the reconstructed positions are the output as-is.

### Removed
- The dead `frame` parameter (`frame="icrs"`) is gone from `radec_z_to_xyz`,
  `xyz_to_radec_z` and the pipeline call sites, together with the unused
  `SkyCoord` import. These transforms have always been direct
  spherical <-> Cartesian numpy operations with no frame handling; the parameter
  promised behaviour that did not exist.

## [0.7.0] - 2026-07-27

### Added
- `read_displacement_at` is now implemented on the FFT solvers
  (`FFTSolverCPU`/`FFTSolverGPU`): it interpolates the spectral displacement grid
  at the requested positions and returns a host `(N, 3)` array, matching the
  multigrid solver. The GPU solver keeps the field on the device and brings only
  the per-particle result back to host.
- The FFT solvers accept `pbc` as a constructor argument, so the periodic-wrap
  behaviour of the displacement read-out is configured on the solver instead of
  the caller.
- `BAOReconstructor.interpolate_displacement` (real-space displacement at the
  tracer positions) and `BAOReconstructor.get_rsd_displacement` (the RSD
  contribution to the displacement) are now public, documented methods.

### Changed
- `BAOReconstructor` reads every solver out through the single
  `read_displacement_at` interface instead of branching on `solver_type`; the
  FFT/multigrid difference (interpolate a spectral Psi grid vs. differentiate the
  potential on the fly) is now hidden behind the solver.
- Mass-assignment-scheme (`MAS`) case handling is standardized through
  `format_mas` across the FFT and multigrid solvers.

## [0.6.0] - 2026-07-20

### Added
- `grid_density` pipeline output option: saves the overdensity field
  (`delta_on_mesh`) as a FITS image, alongside the existing `grid_potential` and
  `grid_displacement` grid outputs. Follows the same host-copy-then-release
  pattern on the GPU path, and is skipped (kept intact) when
  `reconstructor_object` is also requested.
- `ReconstructionPipelineInteractive` (exported as
  `baorecon.pipeline.ReconstructionPipelineInteractive`): a memory-retaining
  mirror of `ReconstructionPipeline` for interactive/notebook use, replacing the
  old, unexported `bao_pipeline_straight.py`. Every intermediate array (raw and
  Cartesian positions, reconstructed positions, solver grids) stays reachable on
  the object instead of being released mid-run.
- Packaging: adopted a PEP 621 `pyproject.toml` with `test`/`notebook`/`gpu`/`docs`
  optional-dependency extras and a version dynamically single-sourced from
  `baorecon.__version__`; `setup.py` and the `requirements/*.txt` files are
  removed. CI now installs via these extras and tests Python 3.10-3.12, and the
  Sphinx docs version is single-sourced the same way.

### Fixed
- `examples/run_bao_pipeline.py` now prints every path returned by
  `pipeline.run()` instead of assuming a fixed `(data_path, random_path)`
  two-tuple, which broke once `run()` started returning the full `saved_paths`
  dict.

## [0.5.0] - 2026-07-08

### Added
- Streamed radial (`LocalLOS`) projection promoted to shared infrastructure in
  `baorecon/solvers/fft/_radial_stream.py`: the per-cell radial versor
  `n̂ = x/|x|` is evaluated on the fly and the potential gradient is projected and
  scattered one component at a time, so the full `(N, N, N, 3)` gradient / versor /
  parallel-field grids are never held in memory at once. It is now used by the
  **default scipy CPU solver** and the **GPU solver** (via CuPy `ElementwiseKernel`
  twins), not only the opt-in pyfftw backend — lowering peak memory on the radial
  line of sight for every backend. Cross-implementation parity (numba vs numpy vs
  CuPy) is guarded by `tests/test_radial_kernels.py`.
- Benchmark: `bench_bao_reconstructor.py` gained `--smoother {jacobi,mcgs}`
  (multigrid smoother: Jacobi V-cycle or multicolor Gauss–Seidel) and
  `--fft {scipy,pyfftw}` (selects the CPU ifft FFT backend via `BAORECON_FFT`),
  and now warns when `--fft pyfftw` cannot take effect (multigrid / GPU).
- Benchmark: all CSVs gained a `vram_peak_mb` column — the CuPy memory-pool
  high-water mark (`total_bytes()`) on GPU, `0.0` on CPU / pyrecon — tracked once
  in the shared `bench_common.measure()` primitive.

### Changed
- `divergence_inplace` → `divergence_from_components` in
  `baorecon/solvers/fft/_common.py`: it now takes a `get_component(i)` callback so
  each gradient component can be synthesised lazily by the caller.
- `Mesh` stores `boxsize` / `boxcentre` (and the derived `cell_size` / `min_corner`)
  at the mesh's working precision (`dtype`) instead of pinning them to float32, so a
  float64 mesh keeps float64 geometry. The default float32 mode is unchanged, and
  `LocalLOS` still exposes a float32 `cell_size` so the on-the-fly radial versor
  matches the cached versor grid cell for cell in float32.
- Mass assignment / read-out now handle the non-periodic (`pbc=False`) box edge
  uniformly by **clamping** out-of-range stencil cells to the nearest boundary cell
  (mass-conserving) across every scheme (NGP / CIC / TSC), the serial and parallel
  CPU kernels, and the GPU kernels. Previously the parallel CPU CIC/TSC kernels and
  `tsc_read` *dropped* those contributions and the GPU kernels ignored `pbc` and
  always wrapped, so `pbc=False` results near the box edge change slightly: mass is
  now conserved and paint/read stay mutual adjoints at the boundary, identically on
  CPU and GPU.

### Fixed
- CPU mass assignment / read-out are now type-neutral instead of forcing float32:
  the interface allocates the grid at the mesh's working precision and casts
  positions / weights to match, so a `dtype=float64` run is honoured end-to-end
  rather than silently downcast during painting (the accumulation happened in
  float32 even when the surrounding pipeline was float64). The GPU path stays
  float32 by design.
- The iterative FFT (iFFT / Burden) CPU solver honours the working precision: the
  wavevectors (scipy and pyfftw paths) and the pyfftw in-place buffers / outputs now
  follow the delta dtype (float32/complex64 or float64/complex128), so a float64
  reconstruction stays float64 through the solver instead of round-tripping through
  float32. `prepare_k_components` also accepts a `numpy.dtype` instance, not only a
  type. (The radial `LocalLOS` versor geometry remains float32.)
- `tsc_assign_serial` silently ignored its `pbc` argument and always wrapped
  periodically; it now honours `pbc` (periodic wrap when true, boundary clamp when
  false), matching the other kernels and `tsc_read`.
- `cic_assign_serial` and `cic_read` used truncation (`int()`) for the cell index,
  which produced negative CIC weights for negative positions under PBC; they now use
  `floor`. Not reachable through the pipeline (positions are pre-wrapped to
  `[0, boxsize)`), but wrong for direct `assign` / `readout` calls.
- GPU mass assignment / read-out kernels now honour `pbc`. They previously took no
  `pbc` argument and **always wrapped periodically**, so a `device="gpu", pbc=False`
  run wrapped box-edge mass to the far side instead of clamping (diverging from the
  CPU kernels). The four `numba.cuda` kernels now thread `pbc` (wrap when true,
  boundary clamp when false) and use `floor` for the CIC cell index, matching
  `baorecon.mas.cpu`. `tests/test_density_manager.py` is parametrized over
  `pbc ∈ {True, False}` on both backends.
- Benchmark: the `bench_bao_reconstructor.py` `smoother` argument was passed as a
  bare kwarg that `BAOReconstructor(**kwargs)` silently swallowed and never reached
  the solver; it is now threaded through `solver_args`. GPU VRAM tracking also
  called a non-existent `MemoryPool.max_bytes()` (crashing the GPU worker) and
  stored the result in a column `save_csv` dropped; both are resolved.

## [0.4.0] - 2026-07-03

### Added
- Pluggable catalog I/O backends with a Parquet backend alongside FITS; the
  compute layer only ever sees NumPy arrays, decoupled from the on-disk format.
- Configurable working precision via `reconstruction.dtype` (default `float32`,
  `float64` to opt into double precision), propagated end-to-end from catalog
  load through reconstruction to the saved coordinate/displacement columns.
- `keep_cols` column pruning so only the columns the pipeline needs (plus any
  explicitly kept) are read; FITS column-subset reads use `fitsio` when
  available, otherwise prune in memory.

### Changed
- Catalogs are downcast to the working precision on load, and positions are held
  at that precision throughout the pipeline; the coordinate helpers now preserve
  their input dtype rather than forcing `float32`.
- Substantially reduced pipeline memory footprint: raw RA/DEC/z are released
  after coordinate conversion, and `run()` releases the heavy arrays
  progressively — potential/displacement grids are written and freed one at a
  time, and catalogues are written one tracer at a time with their position
  arrays dropped immediately after (on GPU the freed CuPy buffers are returned
  to the device memory pool).

### Fixed
- `bench_pipeline_class` reported roughly double the true total memory because it
  reused a previous pipeline allocation.
- `resolve_format` now raises clear errors for an unknown format or extension.

[Unreleased]: https://github.com/EdoardoMaragliano/baorecon/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/EdoardoMaragliano/baorecon/releases/tag/v0.6.0
[0.5.0]: https://github.com/EdoardoMaragliano/baorecon/releases/tag/v0.5.0
[0.4.0]: https://github.com/EdoardoMaragliano/baorecon/releases/tag/v0.4.0
