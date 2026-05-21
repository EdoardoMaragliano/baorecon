import numpy as np
from unittest.mock import patch
from zeldareco.displacement_solver.fft_solver import FFTSolver


def test_fft_solver_lazy():
    # Create tiny delta and mesh via simple mocks
    delta = np.zeros((4, 4, 4), dtype=np.float64)
    # Use Mesh from real module
    from zeldareco.mesh.mesh import Mesh
    mesh = Mesh(4, 100.0, np.array([50.0, 50.0, 50.0]))

    solver = FFTSolver(delta, mesh)
    # Accessing potential/displacement should trigger _compute exactly once
    with patch.object(solver, '_compute', wraps=solver._compute) as mocked:
        _ = solver.displacement
        _ = solver.displacement
        assert mocked.call_count == 1
