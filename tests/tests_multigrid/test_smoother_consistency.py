"""Self-consistency tests between the two multigrid smoothers: ``jacobi`` and ``mcgs``.

``jacobi`` (damped Jacobi) and ``mcgs`` (8-colour Gauss-Seidel) are the two
smoother branches of the *same* FMG/V-cycle in
:mod:`baorecon.solvers.multigrid.solver`. Both relax the identical discrete
operator ``L(v) = f`` (the RSD-modified anisotropic Laplacian implemented by
``residual_jit``), so the solve must produce the same potential -> displacement
-> shifted positions regardless of smoother. These tests assert two independent
things:

* **cross-agreement**: ``jacobi`` and ``mcgs`` agree with each other;
* **anchor**: both agree with an independent analytic/FFT reference, so they are
  right, not just consistent with a shared bug.

Everything runs through ``solve_fmg`` at the **production defaults**
(``v_cycles=6, n_smooth=5, damping=0.4``, ``dtype=float32``) -- exactly what
:class:`~baorecon.solvers.multigrid.solver.MultigridSolver._compute_potential_mesh`
uses -- so the tests validate the configuration that actually ships, not an
idealised over-converged one. Full Multigrid is efficient enough that both
smoothers already agree to ~1e-5 (relative) at 6 cycles; the one slow case is the
**radial (local LOS) redshift-space** operator, whose wide-angle terms converge
much more slowly, so its tolerance is looser. Multigrid is CPU-only, so there is
no device parametrisation; float32 matches the production working precision and
the smoother agreement is convergence-limited (not precision-limited), so float64
would only double the numba compile cost without tightening anything.
"""

import numpy as np
import pytest

from baorecon.solvers.multigrid.solver import _RawMultigrid, MultigridSolver
from baorecon.solvers.multigrid._kernels import residual_jit
from baorecon.mesh.mesh import Mesh
from baorecon.mesh.los import FixedAxisLOS


# ---------------------------------------------------------------------------
# Production solve_fmg defaults (see MultigridSolver._compute_potential_mesh)
# ---------------------------------------------------------------------------
V_CYCLES = 6
N_SMOOTH = 5
DAMPING = 0.4
DTYPE = np.float32

BOX = 1000.0
LOS_Z = np.array([0.0, 0.0, 1.0], dtype=DTYPE)
# Plane-parallel box straddling the origin; radial box pushed into the positive
# octant so the observer-at-origin radial versor is well defined over the grid.
BC_PP = np.array([BOX / 2, BOX / 2, BOX / 2], dtype=DTYPE)
BC_RADIAL = np.array([1.5 * BOX, 1.5 * BOX, 1.5 * BOX], dtype=DTYPE)


def make_source(N, L=BOX, dtype=DTYPE):
    """A smooth, mean-zero source field (a few low Fourier modes)."""
    dims = (N, N, N) if np.isscalar(N) else tuple(int(n) for n in N)
    xs = [np.linspace(0, L, d, endpoint=False) for d in dims]
    X, Y, Z = np.meshgrid(*xs, indexing="ij")
    k = 2.0 * np.pi / L
    src = np.sin(k * X) + 0.5 * np.cos(k * Y) + 0.3 * np.sin(k * Z)
    src = src.astype(dtype)
    src -= src.mean()
    return src


def _box3(boxsize, dtype=DTYPE):
    box = np.asarray(boxsize, dtype=dtype)
    return np.full(3, box, dtype=dtype) if box.ndim == 0 else box


def solve_prod(source, smoother, boxsize, boxcenter, beta, ppl,
               v_cycles=V_CYCLES, n_smooth=N_SMOOTH, damping=DAMPING, dtype=DTYPE):
    """Run a fresh ``_RawMultigrid.solve_fmg`` with the given smoother, using the
    production FMG defaults. This is the same driver MultigridSolver uses."""
    dims = np.asarray(source.shape, dtype=np.int32)
    solver = _RawMultigrid(N_fine=dims, boxsize=_box3(boxsize, dtype),
                           boxcenter=np.asarray(boxcenter, dtype),
                           use_plane_parallel=ppl, dtype=dtype, smoother=smoother)
    return solver.solve_fmg(source, beta=beta, los=np.asarray(LOS_Z, dtype),
                            v_cycles=v_cycles, n_smooth=n_smooth, damping=damping)


def residual_rel_norm(v, source, boxsize, boxcenter, beta, ppl):
    """``||f - L(v)|| / ||f||`` for the discrete multigrid operator."""
    dtype = v.dtype
    dims = np.asarray(v.shape, dtype=np.int32)
    out = np.zeros(v.size, dtype=dtype)
    residual_jit(v.ravel(), source.ravel().astype(dtype), out, dims, int(dims[0]), 0,
                 _box3(boxsize, dtype), np.asarray(boxcenter, dtype), beta,
                 np.asarray(LOS_Z, dtype), ppl)
    return float(np.linalg.norm(out) / np.linalg.norm(source.ravel()))


def rel_l2(a, b, demean=True):
    """Relative L2 difference. The Poisson operator is defined up to an additive
    constant (its DC null space), so potentials are compared mean-subtracted."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if demean:
        a = a - a.mean()
        b = b - b.mean()
    denom = np.linalg.norm(a)
    return float(np.linalg.norm(a - b) / denom) if denom else float(np.linalg.norm(a - b))


def corr(a, b):
    return float(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1])


# (beta, use_plane_parallel, boxcenter, relL2 tolerance). Tolerances are the
# measured jacobi-vs-mcgs agreement at production FMG defaults, with ~10x margin:
# plane-parallel / radial-realspace agree to ~1e-5; radial redshift-space, whose
# operator converges slowly, agrees to ~2e-4.
AGREEMENT_CASES = [
    pytest.param(0.0, True, BC_PP, 1e-4, id="realspace-planeparallel"),
    pytest.param(0.4, True, BC_PP, 1e-4, id="redshift-planeparallel"),
    pytest.param(0.0, False, BC_RADIAL, 1e-4, id="realspace-radial"),
    pytest.param(0.4, False, BC_RADIAL, 2e-3, id="redshift-radial"),
]


# ---------------------------------------------------------------------------
# Group A -- solution agreement at production defaults (the core self-consistency)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("beta,ppl,boxcenter,tol", AGREEMENT_CASES)
def test_potentials_agree(beta, ppl, boxcenter, tol):
    """jacobi and mcgs produce the same potential (up to the DC constant)."""
    src = make_source(32)
    vj = solve_prod(src, "jacobi", BOX, boxcenter, beta, ppl)
    vm = solve_prod(src, "mcgs", BOX, boxcenter, beta, ppl)

    assert np.isfinite(vj).all() and np.isfinite(vm).all()
    assert rel_l2(vj, vm) < tol
    assert corr(vj - vj.mean(), vm - vm.mean()) > 1.0 - 1e-3


def test_both_match_fft_reference():
    """Anchor: both smoothers reproduce the analytic FFT Poisson solution.

    Guards against jacobi and mcgs agreeing on a *wrong* answer. The discrete
    7-point Laplacian differs from ``-k^2`` by an O(h^2) eigenvalue shift, so the
    match is judged by correlation (tight) plus a loose relative L2 that tolerates
    that discretisation gap.
    """
    N, L = 32, BOX
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    k = 2.0 * np.pi / L
    source = (np.sin(k * X)).astype(DTYPE)
    source -= source.mean()

    # FFT reference: laplacian(phi) = source  ->  phi_k = source_k / (-k^2).
    delta_k = np.fft.fftn(source)
    kk = np.fft.fftfreq(N) * N * (2 * np.pi / L)
    kx, ky, kz = np.meshgrid(kk, kk, kk, indexing="ij")
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k2[0, 0, 0] = 1.0
    phi_k = delta_k / (-k2)
    phi_k[0, 0, 0] = 0.0
    ref = np.fft.ifftn(phi_k).real   # solver solves laplacian(v)=source, so v ~ ref

    for smoother in ("jacobi", "mcgs"):
        v = solve_prod(source, smoother, BOX, BC_PP, 0.0, True)
        assert corr(v - v.mean(), ref - ref.mean()) > 0.999
        assert rel_l2(v, ref) < 2e-2


# ---------------------------------------------------------------------------
# Group B -- residual / convergence self-consistency at production defaults
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smoother", ["jacobi", "mcgs"])
def test_residual_small_at_production_defaults(smoother):
    """L(v) ~ f after the default 6 FMG cycles (plane-parallel: fast convergence)."""
    src = make_source(32)
    v = solve_prod(src, smoother, BOX, BC_PP, 0.4, True)
    assert residual_rel_norm(v, src, BOX, BC_PP, 0.4, True) < 1e-3


def test_mcgs_converges_no_slower_than_jacobi():
    """At the same cycle budget, Gauss-Seidel reaches at least as low a residual."""
    src = make_source(32)
    rj = residual_rel_norm(solve_prod(src, "jacobi", BOX, BC_PP, 0.0, True),
                           src, BOX, BC_PP, 0.0, True)
    rm = residual_rel_norm(solve_prod(src, "mcgs", BOX, BC_PP, 0.0, True),
                           src, BOX, BC_PP, 0.0, True)
    assert rm <= 1.5 * rj, f"mcgs residual {rm:.2e} worse than jacobi {rj:.2e}"


def test_extra_cycles_barely_change_solution():
    """The default 6 cycles are converged: doubling them barely moves the answer,
    for either smoother (a per-smoother stability / idempotence check)."""
    src = make_source(32)
    for smoother in ("jacobi", "mcgs"):
        v6 = solve_prod(src, smoother, BOX, BC_PP, 0.0, True, v_cycles=6)
        v12 = solve_prod(src, smoother, BOX, BC_PP, 0.0, True, v_cycles=12)
        assert rel_l2(v6, v12) < 1e-3


@pytest.mark.parametrize("smoother", ["jacobi", "mcgs"])
def test_deterministic(smoother):
    """Repeated solves are bit-reproducible (no numba parallel nondeterminism)."""
    src = make_source(32)
    a = solve_prod(src, smoother, BOX, BC_PP, 0.4, True)
    b = solve_prod(src, smoother, BOX, BC_PP, 0.4, True)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Group C -- public MultigridSolver: potential and displacement agree
# ---------------------------------------------------------------------------
def _public_solver(delta, mesh, smoother):
    return MultigridSolver(
        delta, mesh, f=0.8, bias=2.0, los=FixedAxisLOS(2),
        RSDspace="RedshiftSpace", use_plane_parallel=True, smoother=smoother,
    )


def test_public_potential_and_displacement_agree():
    """Through the public API (the production solve_fmg path), potential and
    displacement match between smoothers."""
    N = 32
    mesh = Mesh(nmesh=N, boxsize=BOX, boxcentre=BC_PP)
    delta = make_source(N)

    pj = _public_solver(delta, mesh, "jacobi").potential
    pm = _public_solver(delta, mesh, "mcgs").potential
    assert rel_l2(pj, pm) < 1e-3
    assert corr(pj - pj.mean(), pm - pm.mean()) > 0.999

    dj = _public_solver(delta, mesh, "jacobi").displacement
    dm = _public_solver(delta, mesh, "mcgs").displacement
    # displacement = -grad(phi): the DC constant drops out, so compare directly.
    assert rel_l2(dj, dm, demean=False) < 1e-2


# ---------------------------------------------------------------------------
# Group D -- end-to-end reconstruction (user-facing)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("los,tol", [("z", 0.1), (None, 0.6)])
def test_reconstruction_shifts_agree(los, tol):
    """BAOReconstructor shifted positions agree between smoothers, and the
    disagreement is tiny compared with the R_sm=15 Mpc/h smoothing scale."""
    from baorecon import BAOReconstructor

    rng = np.random.default_rng(0)
    n = 3000
    data = rng.uniform(100, 900, size=(n, 3)).astype(np.float32)
    rand = rng.uniform(100, 900, size=(n, 3)).astype(np.float32)
    w = np.ones(n, np.float32)

    def recon(smoother):
        r = BAOReconstructor(
            data_pos=data.copy(), random_pos=rand.copy(),
            data_weights=w, random_weights=w, RSDspace="RedshiftSpace",
            nmesh=64, boxsize=BOX, boxcentre=BC_PP, padding=0.0, los=los,
            R_sm=15.0, pbc=False, rectype="rec-sym", f=0.8, bias=2.0, MAS="CIC",
            dtype=np.float32, solver_type="multigrid", device="cpu",
            solver_args={"smoother": smoother},
        )
        return r.run_reconstruction()

    dj, rj = recon("jacobi")
    dm, rm = recon("mcgs")
    d_data = np.linalg.norm(dj - dm, axis=1).max()
    d_rand = np.linalg.norm(rj - rm, axis=1).max()
    assert d_data < tol, f"data shift disagreement {d_data:.3f} Mpc/h"
    assert d_rand < tol, f"random shift disagreement {d_rand:.3f} Mpc/h"
    # sanity: the shifts themselves are order R_sm, so tol << typical shift
    assert d_data < 0.1 * np.linalg.norm(dj - data, axis=1).mean()


# ---------------------------------------------------------------------------
# Group E -- grid variants and the mcgs odd-bottom-level fallback
# ---------------------------------------------------------------------------
def test_anisotropic_grid_agree():
    """Rectangular (balanced) grid: both smoothers still agree."""
    dims = np.array([32, 32, 64], dtype=np.int32)
    src = make_source(dims)
    box = np.array([BOX, BOX, 2 * BOX])
    vj = solve_prod(src, "jacobi", box, BC_PP, 0.4, True)
    vm = solve_prod(src, "mcgs", box, BC_PP, 0.4, True)
    assert rel_l2(vj, vm) < 1e-3


def test_mcgs_odd_bottom_level_fallback():
    """nmesh=20 (=5*2^2) coarsens 20->10->5; the odd bottom forces mcgs to fall
    back to Jacobi there. It must still converge and match a pure-jacobi solve."""
    N = 20
    src = make_source(N)
    solver = _RawMultigrid(N_fine=N, boxsize=_box3(BOX), boxcenter=BC_PP,
                           use_plane_parallel=True, dtype=DTYPE, smoother="mcgs")
    assert solver.levels[-1].shape == (5, 5, 5)          # odd bottom level
    assert solver._bottom_smoother == "jacobi"           # fallback engaged

    vm = solver.solve_fmg(src, beta=0.0, los=np.asarray(LOS_Z, DTYPE),
                          v_cycles=V_CYCLES, n_smooth=N_SMOOTH, damping=DAMPING)
    vj = solve_prod(src, "jacobi", BOX, BC_PP, 0.0, True)
    assert rel_l2(vj, vm) < 1e-3


@pytest.mark.parametrize("smoother", ["jacobi", "mcgs"])
def test_both_reject_non_multigrid_friendly(smoother):
    """Grid-validation is identical for both smoothers."""
    with pytest.raises(ValueError):
        _RawMultigrid(N_fine=30, boxsize=_box3(BOX), boxcenter=BC_PP,
                      use_plane_parallel=True, smoother=smoother)
