# BAOreconstruction

This package contains the high-level reconstruction orchestration and the density-field preparation layer.

## Responsibilities

- normalize user inputs such as positions, weights, box size, and mass-assignment choice
- build the mesh overdensity field through `DensityManager`
- coordinate the reconstruction flow in `BAOReconstructor`
- keep the public API focused on catalog-level reconstruction, not on low-level mesh details

## Data flow

1. Raw data and random catalogs enter the orchestrator.
2. `DensityManager` prepares inputs and generates `delta_on_mesh`.
3. The selected solver consumes `delta_on_mesh` and returns the displacement field.
4. The orchestrator applies the shift to data and random catalogs.


## Public entry points

- `density_manager.py`: mesh preparation and mass assignment
- `bao_reconstructor.py`: high-level reconstruction orchestration

## Notes

- This layer is the right place for catalog formatting and reconstruction policy.
- Solver details belong in `zeldareco.displacement_solver`.
