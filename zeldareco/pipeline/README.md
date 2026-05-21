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
8. saving outputs with tokenized filenames

## Main entry point

- [bao_pipeline.py](bao_pipeline.py): the `ReconstructionPipeline` orchestrator

## Example usage

Use the example configuration in [examples/bao_pipeline_example.yaml](../../examples/bao_pipeline_example.yaml) together with the CLI script in [examples/run_bao_pipeline.py](../../examples/run_bao_pipeline.py).

Typical execution flow:

1. load the YAML config
2. instantiate `ReconstructionPipeline`
3. call `run()`
4. collect the reconstructed FITS outputs in the configured folder

## Notes

- The pipeline is intentionally thin: it orchestrates I/O and coordinate transforms, but keeps the reconstruction physics inside `zeldareco.BAOreconstruction`.
- The config file is the preferred way to define column mappings, coordinate conventions, cosmology parameters, and output naming.