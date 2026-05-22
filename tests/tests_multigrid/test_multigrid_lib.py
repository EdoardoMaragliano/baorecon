import numpy as np
import pytest

from zeldareco.displacement_solver.multigrid_lib import (
    prolong_jit,
    reduce_jit,
    residual_jit,
)


def _prolong_wrapper_from_coarse(coarse_grid_3d):
    """Wrapper for the prolong method."""
    nx_c, ny_c, nz_c = coarse_grid_3d.shape
    nx_f, ny_f, nz_f = nx_c * 2, ny_c * 2, nz_c * 2
    nmesh_dims = np.array([nx_f, ny_f, nz_f], dtype=np.int32)
    fine_flat = np.zeros(nx_f * ny_f * nz_f, dtype=np.float64)
    coarse_flat = coarse_grid_3d.ravel().astype(np.float64)
    localnmeshx = nx_f
    offsetx = 0
    prolong_jit(coarse_flat, fine_flat, nmesh_dims, localnmeshx, offsetx)
    return fine_flat.reshape((nx_f, ny_f, nz_f))


def _reduce_wrapper_from_fine(fine_grid_3d):
    """Wrapper for the reduce method."""
    nx, ny, nz = fine_grid_3d.shape
    nx_c, ny_c, nz_c = nx // 2, ny // 2, nz // 2
    fine_flat = fine_grid_3d.ravel().astype(np.float64)
    coarse_flat = np.zeros(nx_c * ny_c * nz_c, dtype=np.float64)
    nmesh_dims_fine = np.array([nx, ny, nz], dtype=np.int32)
    localnmeshx_c = nx_c
    offsetx_c = 0
    reduce_jit(fine_flat, coarse_flat, nmesh_dims_fine, localnmeshx_c, offsetx_c)
    return coarse_flat.reshape((nx_c, ny_c, nz_c))


def _residual_wrapper(v_3d, f_3d, boxsize_val=1.0):
    """Wrapper for the residual method."""
    N = v_3d.shape[0]
    v_flat = v_3d.ravel().astype(np.float64)
    f_flat = f_3d.ravel().astype(np.float64)
    res_out = np.zeros_like(v_flat)
    nmesh_dims = np.array([N, N, N], dtype=np.int32)
    localnmeshx = N
    offsetx = 0
    boxsize = np.array([boxsize_val, boxsize_val, boxsize_val], dtype=np.float64)
    boxcenter = boxsize / 2.0
    beta = 0.0
    los = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    use_plane_parallel = True
    residual_jit(v_flat, f_flat, res_out, nmesh_dims, localnmeshx, offsetx, boxsize, boxcenter, beta, los, use_plane_parallel)
    return res_out.reshape((N, N, N))


def test_prolong_constant_field():
    """Test the prolong method on a constant field.
    
    Ensures that interpolating a uniformly valued grid onto a grid with 
    double the resolution maintains exactly the same constant value across 
    all nodes and results in the correct final dimensions.
    """

    c = np.full((4, 4, 4), 10.0, dtype=np.float64)
    f = _prolong_wrapper_from_coarse(c)
    assert f.shape == (8, 8, 8)
    assert np.allclose(f, 10.0)


def test_prolong_gradient_z():
    """
    Tests the prolongation operator on a field with a linear gradient.
    
    Initializes a field that varies linearly along the Z-axis. Verifies that 
    the operator's linear interpolation correctly computes the expected 
    intermediate values along the direction of variation.
    """
    nx, ny, nz = 4, 4, 4
    c = np.zeros((nx, ny, nz), dtype=np.float64)
    for z in range(nz):
        c[:, :, z] = z * 10.0
    f = _prolong_wrapper_from_coarse(c)
    # check a few expected values along z at (0,0,:6)
    vals = f[0, 0, :6]
    expected = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    assert np.allclose(vals, expected)


def test_prolong_coincidence_nodes():
    """
    Verifies node coincidence between the fine and coarse grids.
    
    During prolongation, the fine grid nodes that spatially overlap with 
    the original coarse grid nodes must keep their original values intact. 
    Extracting nodes with a stride of 2 must yield the exact original input.
    """
    c = np.random.RandomState(0).rand(8, 8, 8).astype(np.float64)
    f = _prolong_wrapper_from_coarse(c)
    extracted = f[::2, ::2, ::2]
    assert extracted.shape == c.shape
    assert np.allclose(extracted, c)


def test_prolong_non_cubic_shape_and_max_principle():
    """
    Tests prolongation on asymmetric domains and the Maximum Principle.
    
    Ensures correct behavior on non-cubic grids (different dimensions per axis) 
    and guarantees that interpolation does not create numerical artifacts 
    (overshoots/undershoots). The fine grid values must be strictly confined 
    between the minimum and maximum values present in the coarse grid.
    """

    # Non cubic shape
    c = np.random.RandomState(1).rand(3, 4, 2).astype(np.float64)
    f = _prolong_wrapper_from_coarse(c)
    assert f.shape == (6, 8, 4)
    # Maximum principle: output within input range (allow tiny eps)
    eps = 1e-12
    assert f.min() >= c.min() - eps
    assert f.max() <= c.max() + eps


def test_prolong_periodicity_sinusoid():
    """
    Verifies the handling of Periodic Boundary Conditions (PBC).
    
    Uses a sinusoidal field to check behavior at the boundaries of the spatial 
    domain. Ensures that interpolation (and difference calculation) at the box 
    edges is as smooth and continuous as in the internal nodes.
    """
    nx = 16
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    c = np.sin(x).reshape((nx, 1, 1)).astype(np.float64)
    f = _prolong_wrapper_from_coarse(c)
    diff_boundary = abs(f[-1, 0, 0] - f[0, 0, 0])
    diff_internal = abs(f[10, 0, 0] - f[11, 0, 0])
    assert abs(diff_boundary - diff_internal) < 0.1


def test_reduce_constant_and_delta_center():
    """
    Tests the restriction (reduction) operator on basic fields.
    
    Evaluates the projection from a fine grid to a coarse grid in two scenarios:
    1. A constant field must return a constant coarse grid.
    2. A single impulse (delta function) at the center must distribute its 
       value to adjacent coarse nodes following the correct interpolation 
       filter weights (converging to an expected center value of ~0.125).
    """

    # constant test
    f = np.ones((8, 8, 8), dtype=np.float64) * 10.0
    c = _reduce_wrapper_from_fine(f)
    assert np.allclose(c, 10.0)

    # delta test: single impulse
    f2 = np.zeros((8, 8, 8), dtype=np.float64)
    f2[2, 2, 2] = 1.0
    c2 = _reduce_wrapper_from_fine(f2)
    # expected center value ~ 0.125
    assert np.isclose(c2[1, 1, 1], 0.125, rtol=1e-8, atol=1e-12)


def test_residual_identity_and_flat_and_exact_cancellation():
    """
    Verifies the accuracy of the residual calculation (r = f - A*v).
    
    Evaluates the internal differential operator (discrete Laplacian) 
    in three regimes:
    - Identity: if field v is zero, the residual must equal the source f.
    - Flat: if v is constant and f is zero, the residual must be near zero.
    - Exact cancellation: if f is set equal to the analytically calculated 
      discrete Laplacian of v, the residual must vanish, proving the 
      correctness of operator A.
    """
    # small N for speed
    N = 8
    # identity: v=0 -> residual == f
    v0 = np.zeros((N, N, N), dtype=np.float64)
    f_rand = np.random.RandomState(2).rand(N, N, N).astype(np.float64)
    r = _residual_wrapper(v0, f_rand)
    assert np.allclose(r, f_rand)

    # flat: v constant, f=0 -> residual ~ 0
    v_const = np.ones((N, N, N), dtype=np.float64) * 100.0
    f_zero = np.zeros((N, N, N), dtype=np.float64)
    r2 = _residual_wrapper(v_const, f_zero)
    assert np.max(np.abs(r2)) < 1e-10

    # exact cancellation: f = Laplacian(v) -> residual ~ 0
    v_rand = np.random.RandomState(3).rand(N, N, N).astype(np.float64)
    # discrete Laplacian using periodic rolls
    term_x = (np.roll(v_rand, -1, axis=0) + np.roll(v_rand, 1, axis=0))
    term_y = (np.roll(v_rand, -1, axis=1) + np.roll(v_rand, 1, axis=1))
    term_z = (np.roll(v_rand, -1, axis=2) + np.roll(v_rand, 1, axis=2))
    # include cell-size normalization as used in the JIT implementation
    boxsize_val = 1.0
    cellsize = boxsize_val / N
    icellsize2 = 1.0 / (cellsize * cellsize)
    lap = (term_x + term_y + term_z - 6.0 * v_rand) * icellsize2
    f_input = lap
    r3 = _residual_wrapper(v_rand, f_input)
    assert np.max(np.abs(r3)) < 1e-8


def test_fmg_vs_fft_and_vcycle():
    """
    End-to-End test of the Multigrid Solver against the analytical FFT solution.
    
    Solves the Poisson equation for a sinusoidal source field using three 
    different methods to verify convergence:
    1. Analytical calculation in Fourier space (FFT) as the reference solution.
    2. Resolution via Full Multigrid (FMG).
    3. Resolution via a single V-Cycle.
    
    Physically validates the results by ensuring the correlation with the 
    FFT solution exceeds strict thresholds (>0.98 for FMG and >0.8 for V-Cycle).
    """
   
    N = 32
    L = 1.0
    x = np.linspace(0, L, N, endpoint=False)
    boxcenter = np.array([L / 2.0, L / 2.0, L / 2.0], dtype=np.float64)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    k = 2.0 * np.pi / L
    source = np.sin(k * X)
    source -= np.mean(source)

    # reference solution via FFT: solve -k^2 phi = source -> phi = -source/k^2
    delta_k = np.fft.fftn(source)
    k_idx = np.fft.fftfreq(N) * N * (2 * np.pi / L)
    kx, ky, kz = np.meshgrid(k_idx, k_idx, k_idx, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0
    phi_k = -delta_k / k2
    phi_k[0, 0, 0] = 0.0
    phi_fft = np.fft.ifftn(phi_k).real

    from zeldareco.displacement_solver.multigrid_solver import _RawMultigrid as MultigridSolver

    solver = MultigridSolver(N_fine=N, boxsize=L, boxcenter=boxcenter, use_plane_parallel=True)

    v_vcycle = solver.solve(source, beta=0.0, n_cycles=10, n_smooth=2, damping=0.4)
    v_fmg = solver.solve_fmg(source, beta=0.0, v_cycles=2, n_smooth=2, damping=0.4)

    # compare shapes and finite
    assert v_vcycle.shape == (N, N, N)
    assert v_fmg.shape == (N, N, N)
    assert np.isfinite(v_vcycle).all()
    assert np.isfinite(v_fmg).all()

    # correlation with FFT solution
    corr_fmg = np.corrcoef(phi_fft.ravel(), v_fmg.ravel())[0, 1]
    corr_vc = np.corrcoef(phi_fft.ravel(), v_vcycle.ravel())[0, 1]

    assert corr_fmg > 0.98
    assert corr_vc > 0.8


def test_vcycle_convergence_rate():
    """
    Measures and validates the asymptotic convergence factor of the V-Cycle.
    
    Theory:
    In a well-conditioned Multigrid solver for the Poisson equation,
    the ratio between the L2 norm of the residual at iteration k+1 and 
    the residual at iteration k (called the asymptotic convergence factor, rho) 
    must be strictly less than 1. It typically settles at stable and very 
    low values (e.g., rho < 0.5 for a 3D Poisson problem).
    """
    # 1. setup of domain and source field
    N = 32
    L = 1.0
    x = np.linspace(0, L, N, endpoint=False)
    boxcenter = np.array([L / 2.0, L / 2.0, L / 2.0], dtype=np.float64)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    k_wave = 2.0 * np.pi / L
    source = np.sin(k_wave * X) * np.cos(k_wave * Y) # mixed 3d source
    source -= np.mean(source) # Ensure average is zero

    # Initialize the solver
    from zeldareco.displacement_solver.multigrid_solver import _RawMultigrid as MultigridSolver
    solver = MultigridSolver(N_fine=N, boxsize=L, boxcenter=boxcenter, use_plane_parallel=True)

    # 2. Norm of the initial residual (before each V-cycle, v=0 -> r=f)
    norm_r_initial = np.linalg.norm(source.ravel())
    residual_norms = [norm_r_initial]

    # 3.Execute increasing number of V-cycles and compute residual
    max_cycles = 5
    for n_cyc in range(1, max_cycles + 1):
        # solve using n_cyc cycles
        v_k = solver.solve(source, beta=0.0, n_cycles=n_cyc, n_smooth=2, damping=0.4)
        
        # Compute the residual r = f - A*v
        r_k = _residual_wrapper(v_k, source, boxsize_val=L)
        
        # Save the L2 norm of the residual
        norm_r = np.linalg.norm(r_k.ravel())
        residual_norms.append(norm_r)

    # 4. Compute the convergence factor rho = ||r_{k}|| / ||r_{k-1}||
    convergence_factors = [
        residual_norms[i] / residual_norms[i-1] 
        for i in range(1, len(residual_norms))
    ]

    # --- ASSERTIONS ---
    
    # A. Strict monotony: each V-cycle MUST reduce the residual
    for i, rho in enumerate(convergence_factors):
        assert rho < 1.0, (
            f"Il residuo non sta diminuendo al ciclo {i+1}! "
            f"rho = {rho:.4f}. Il solver potrebbe essere divergente."
        )

    # B. Asymptotic convergence rate
    # Should be rho constant and < 0.5.
    # Take the mean of the last two cycle as an estimate of the asymptotic value
    asymptotic_rho = np.mean(convergence_factors[-2:])
    
    assert asymptotic_rho < 0.6, (
        f"Convergence is too slow for a V-cycle Multigrid. "
        f"Asymptotic convergence rho = {asymptotic_rho:.4f} (expected < 0.6)."
    )
    
    # Opzionale: stampa i tassi se esegui pytest con la flag -s
    print(f"\nConvergence factors (for each cycle): {[round(r, 3) for r in convergence_factors]}")