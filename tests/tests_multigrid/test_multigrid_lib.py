import numpy as np
import pytest

from zeldareco.displacement_solver.multigrid_lib import (
    prolong_jit,
    reduce_jit,
    residual_jit,
)


def _prolong_wrapper_from_coarse(coarse_grid_3d):
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
    c = np.full((4, 4, 4), 10.0, dtype=np.float64)
    f = _prolong_wrapper_from_coarse(c)
    assert f.shape == (8, 8, 8)
    assert np.allclose(f, 10.0)


def test_prolong_gradient_z():
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
    c = np.random.RandomState(0).rand(8, 8, 8).astype(np.float64)
    f = _prolong_wrapper_from_coarse(c)
    extracted = f[::2, ::2, ::2]
    assert extracted.shape == c.shape
    assert np.allclose(extracted, c)


def test_prolong_non_cubic_shape_and_max_principle():
    # Non cubic shape
    c = np.random.RandomState(1).rand(3, 4, 2).astype(np.float64)
    f = _prolong_wrapper_from_coarse(c)
    assert f.shape == (6, 8, 4)
    # Maximum principle: output within input range (allow tiny eps)
    eps = 1e-12
    assert f.min() >= c.min() - eps
    assert f.max() <= c.max() + eps


def test_prolong_periodicity_sinusoid():
    nx = 16
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    c = np.sin(x).reshape((nx, 1, 1)).astype(np.float64)
    f = _prolong_wrapper_from_coarse(c)
    diff_boundary = abs(f[-1, 0, 0] - f[0, 0, 0])
    diff_internal = abs(f[10, 0, 0] - f[11, 0, 0])
    assert abs(diff_boundary - diff_internal) < 0.1


def test_reduce_constant_and_delta_center():
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
    # small sinusoidal source to compare FMG against FFT solution
    N = 32
    L = 1.0
    x = np.linspace(0, L, N, endpoint=False)
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

    solver = MultigridSolver(N, L, use_plane_parallel=True)

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
