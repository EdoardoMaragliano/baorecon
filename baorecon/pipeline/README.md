# Pipeline

This package provides the YAML-driven, end-to-end catalog pipeline that wraps the core BAO reconstruction code.

## What it does

The pipeline is designed for survey-style workflows where inputs are FITS catalogs and the reconstruction needs to keep track of catalog metadata.

It handles:

1. loading data and random catalogs from FITS paths
2. reading a YAML configuration file
3. selecting the coordinate, weight, and ID columns
4. converting RA/DEC/redshift to Cartesian coordinates using Astropy cosmology helpers
5. running `BAOReconstructor`
6. preserving IDs through optional masking/filtering steps
7. converting reconstructed coordinates back to RA/DEC/redshift
8. saving a flexible set of outputs (catalogs, fields, etc.) with tokenized filenames

### Output Control

The `output` section of the YAML configuration file allows fine-grained control over what gets saved. By default, only the reconstructed catalogs are written to disk. You can specify other artifacts using the `save` key:

```yaml
output:
  folder: "./output/my_run"
  # ...
  save:
    - catalogs              # (Default) Reconstructed FITS catalogs.
    - tracer_displacements  # Adds S_X, S_Y, S_Z columns to the FITS catalogs.
    - grid_potential        # Scalar potential (phi) on the grid (FITS image).
    - grid_displacement     # Displacement field (psi) on the grid (FITS image).
    - reconstructor_object  # The full BAOReconstructor object (pickle file for debugging).
```

If the `save` key is omitted, the pipeline defaults to `['catalogs']`.

### Compute backend (CPU/GPU)

The `reconstruction` section accepts a `device` key controlling where the
density assignment, FFT-based displacement/potential solver, and field
interpolation run:

```yaml
reconstruction:
  # ...
  device: "cpu"  # "cpu" (default) or "gpu"
```

`device: "gpu"` requires CuPy and a CUDA-enabled GPU (see the top-level
[README.md](../../README.md#gpu-support-optional)). The `multigrid` solver
type always runs on CPU regardless of this setting. The reconstructed
catalogs and grid outputs are returned as host (NumPy) arrays either way.

## Main entry point

- [bao_pipeline.py](bao_pipeline.py): the `ReconstructionPipeline` orchestrator — this is the current, supported implementation and the one exported as `baorecon.pipeline.ReconstructionPipeline`.
- [bao_pipeline_straight.py](bao_pipeline_straight.py): **legacy** version, kept for reference only. It is the original monolithic pipeline (saves everything at the end, no GPU host-transfer handling and no progressive memory release) and is not exported by the package. Use `bao_pipeline.py` for all new work.

## Example usage

Use the example configuration in [examples/bao_pipeline_example.yaml](../../examples/bao_pipeline_example.yaml) together with the CLI script in [examples/run_bao_pipeline.py](../../examples/run_bao_pipeline.py).

Typical execution flow:

1. load the YAML config
2. instantiate `ReconstructionPipeline`
3. call `run()`
4. collect the specified outputs in the configured folder

## Notes

- The pipeline is intentionally thin: it orchestrates I/O and coordinate transforms, but keeps the reconstruction physics inside `baorecon.reconstruction`.
- The config file is the preferred way to define column mappings, coordinate conventions, cosmology parameters, and output naming.
- `run()` saves each output as soon as its inputs are final and releases the heavy solver/grid arrays mid-run (on GPU it also returns the freed CuPy buffers to the device memory pool), keeping peak memory low for large meshes. The step-by-step methods (`load_catalogs`, `convert_to_xyz`, `reconstruct`, `convert_back`, `save_outputs`) — and `save_outputs()`, which writes everything at the end without the progressive release — remain available for callers that drive the pipeline manually.