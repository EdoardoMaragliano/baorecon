"""Both reconstruction conventions, and the one-line public API.

``rec-sym`` and ``rec-iso`` differ in exactly one place — whether the randoms
carry the redshift-space term — and until now only ``rec-sym`` was ever
constructed in the suite. The tests here pin the *difference* rather than just
running both: a smoke test would pass even if the branch were dead.

``reconstruct_positions`` is the entry point the README leads with and had no
test at all.
"""

import numpy as np
import pytest

from baorecon import reconstruct_positions
from baorecon.reconstruction.bao_reconstructor import BAOReconstructor

BOXSIZE = 200.0
NMESH = 32
RECTYPES = ["rec-sym", "rec-iso"]


def _catalogs(seed=5, n_data=3000, n_random=12000):
    """A clustered data sample inside a uniform random sample."""
    rng = np.random.RandomState(seed)
    centre = np.array([BOXSIZE / 2] * 3)
    data = np.clip(rng.normal(centre, BOXSIZE / 8, size=(n_data, 3)), 0, BOXSIZE)
    random = rng.uniform(0, BOXSIZE, size=(n_random, 3))
    return data.astype(np.float64), random.astype(np.float64)


def _reconstructor(rectype, rsd_space="RedshiftSpace", **kwargs):
    data, random = _catalogs()
    return BAOReconstructor(
        data_pos=data, random_pos=random, nmesh=NMESH, boxsize=BOXSIZE,
        boxcentre=[BOXSIZE / 2] * 3, padding=0.0, f=0.8, bias=1.5, R_sm=10.0,
        rectype=rectype, RSDspace=rsd_space, los=None, pbc=False,
        solver_type="ifft", dtype=np.float64, **kwargs)


# ---------------------------------------------------------------------------
# rectype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rectype", RECTYPES)
@pytest.mark.parametrize("rsd_space", ["RealSpace", "RedshiftSpace"])
def test_reconstruction_runs_for_both_rectypes(rectype, rsd_space):
    data_rec, random_rec = _reconstructor(rectype, rsd_space).run_reconstruction()
    data, random = _catalogs()

    assert data_rec.shape == data.shape and random_rec.shape == random.shape
    assert np.all(np.isfinite(data_rec)) and np.all(np.isfinite(random_rec))
    # Something actually moved: a no-op would satisfy every assertion above.
    assert np.abs(data_rec - data).max() > 1e-3
    assert np.abs(random_rec - random).max() > 1e-3


def test_rectypes_differ_only_in_the_randoms():
    """In redshift space the data shift is shared; the randoms are the difference.

    ``rec-sym`` displaces the randoms by ``Psi + f (Psi.n) n`` and ``rec-iso`` by
    ``Psi`` alone, so the two random sets must differ while the two data sets
    stay identical. This is the assertion a smoke test cannot make.
    """
    data_sym, random_sym = _reconstructor("rec-sym").run_reconstruction()
    data_iso, random_iso = _reconstructor("rec-iso").run_reconstruction()

    np.testing.assert_allclose(data_sym, data_iso, rtol=0, atol=1e-10)
    assert np.abs(random_sym - random_iso).max() > 1e-3, (
        "rec-sym and rec-iso gave identical randoms: the RSD term is not "
        "reaching the randoms, or the rectype branch is dead")


def test_rectypes_agree_on_the_randoms_in_real_space():
    """With no RSD to add, the two conventions coincide by construction."""
    _, random_sym = _reconstructor("rec-sym", "RealSpace").run_reconstruction()
    _, random_iso = _reconstructor("rec-iso", "RealSpace").run_reconstruction()
    np.testing.assert_allclose(random_sym, random_iso, rtol=0, atol=1e-10)


def test_rec_iso_random_shift_carries_no_line_of_sight_term():
    """The rec-iso random displacement must be the real-space Psi exactly."""
    recon = _reconstructor("rec-iso")
    _, random_rec = recon.run_reconstruction()
    psi = recon.interpolate_displacement(recon.random_pos, survey_frame=True)
    np.testing.assert_allclose(random_rec, recon.random_pos - psi, rtol=0, atol=1e-8)


def test_unknown_rectype_is_rejected():
    with pytest.raises(ValueError):
        _reconstructor("rec-nonsense")


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def test_reconstruct_positions_matches_the_explicit_reconstructor():
    """The one-liner must be a shortcut, not a second implementation."""
    data, random = _catalogs()
    common = dict(nmesh=NMESH, boxsize=BOXSIZE, boxcentre=[BOXSIZE / 2] * 3,
                  padding=0.0, R_sm=10.0, rectype="rec-sym", solver_type="ifft",
                  RSDspace="RedshiftSpace", pbc=False, dtype=np.float64)

    d_api, r_api = reconstruct_positions(
        data, random, f=0.8, bias=1.5, smoothing=10.0, los=None,
        device="cpu", **{k: v for k, v in common.items() if k != "R_sm"})
    d_ref, r_ref = BAOReconstructor(
        data_pos=data, random_pos=random, f=0.8, bias=1.5, los=None,
        **common).run_reconstruction()

    np.testing.assert_allclose(d_api, d_ref, rtol=0, atol=1e-10)
    np.testing.assert_allclose(r_api, r_ref, rtol=0, atol=1e-10)


def test_reconstruct_positions_accepts_cellsize_instead_of_nmesh():
    """``cellsize`` derives the grid, and excludes ``nmesh`` and ``boxsize``.

    The API nulls ``nmesh`` for you when ``cellsize`` is given; ``boxsize`` it
    does not, so passing both is rejected downstream. Asserted below.
    """
    data, random = _catalogs()
    shifted, _ = reconstruct_positions(
        data, random, f=0.8, bias=1.5, cellsize=BOXSIZE / NMESH,
        padding=0.0, smoothing=10.0, los=None, pbc=False,
        solver_type="ifft", dtype=np.float64)
    assert shifted.shape == data.shape
    assert np.all(np.isfinite(shifted))
    assert np.abs(shifted - data).max() > 1e-3

    with pytest.raises(ValueError, match="mutually exclusive"):
        reconstruct_positions(data, random, f=0.8, bias=1.5,
                              cellsize=BOXSIZE / NMESH, boxsize=BOXSIZE,
                              padding=0.0, smoothing=10.0, los=None, pbc=False)


def test_reconstruct_positions_returns_both_catalogues_moved():
    data, random = _catalogs()
    d, r = reconstruct_positions(data, random, f=0.8, bias=1.5, nmesh=NMESH,
                                 boxsize=BOXSIZE, boxcentre=[BOXSIZE / 2] * 3,
                                 padding=0.0, smoothing=10.0, los=None,
                                 pbc=False, solver_type="ifft", dtype=np.float64)
    assert d.shape == data.shape and r.shape == random.shape
    assert np.abs(d - data).max() > 1e-3
    assert np.abs(r - random).max() > 1e-3


def test_api_exposes_solver_type_by_name():
    """``solver_type`` is a named parameter, and it selects the solver."""
    import inspect
    params = inspect.signature(reconstruct_positions).parameters
    assert "solver_type" in params
    assert params["solver_type"].default == "multigrid", (
        "the default is BAOReconstructor's; changing it changes every existing call")

    data, random = _catalogs()
    common = dict(nmesh=NMESH, boxsize=BOXSIZE, boxcentre=[BOXSIZE / 2] * 3,
                  padding=0.0, smoothing=10.0, los=None, pbc=False,
                  dtype=np.float64, f=0.8, bias=1.5)
    ifft, _ = reconstruct_positions(data, random, solver_type="ifft", **common)
    mg, _ = reconstruct_positions(data, random, solver_type="multigrid", **common)
    assert not np.allclose(ifft, mg, atol=1e-8), "solver_type is not reaching the solver"


def test_n_iterations_reaches_the_solver_by_both_routes():
    """Exposed means ingested: the keyword must actually change the solve.

    It used to reach ``BAOReconstructor`` as a loose keyword and stop there --
    ``**kwargs`` was declared and never stored -- so the iterative FFT solver ran
    at its internal default whatever the caller or the parfile asked for. The
    value itself is fixed by the literature and nobody is expected to tune it;
    what matters is that it is not silently dropped.
    """
    data, random = _catalogs()
    common = dict(nmesh=NMESH, boxsize=BOXSIZE, boxcentre=[BOXSIZE / 2] * 3,
                  padding=0.0, smoothing=10.0, los=None, pbc=False,
                  dtype=np.float64, f=0.8, bias=1.5, solver_type="ifft",
                  RSDspace="RedshiftSpace")

    one, _ = reconstruct_positions(data, random, n_iterations=1, **common)
    nine, _ = reconstruct_positions(data, random, n_iterations=9, **common)
    assert not np.allclose(one, nine, atol=1e-8), "n_iterations is being dropped"

    # solver_args is the more specific route and wins over the keyword.
    via_args, _ = reconstruct_positions(
        data, random, n_iterations=1, solver_args={"n_iterations": 9}, **common)
    np.testing.assert_allclose(via_args, nine, rtol=0, atol=0)


def test_n_iterations_default_is_the_conventional_three():
    data, random = _catalogs()
    common = dict(nmesh=NMESH, boxsize=BOXSIZE, boxcentre=[BOXSIZE / 2] * 3,
                  padding=0.0, smoothing=10.0, los=None, pbc=False,
                  dtype=np.float64, f=0.8, bias=1.5, solver_type="ifft",
                  RSDspace="RedshiftSpace")
    implicit, _ = reconstruct_positions(data, random, **common)
    explicit, _ = reconstruct_positions(data, random, n_iterations=3, **common)
    np.testing.assert_allclose(implicit, explicit, rtol=0, atol=0)


def test_unknown_keywords_are_rejected():
    """A keyword nothing reads must fail loudly, not vanish.

    ``**kwargs`` was declared and never stored, so anything unrecognised was
    accepted and dropped -- how ``n_iterations`` came to be exposed in three
    places and ingested in none. The error names the offenders and, where it can
    tell, points at where they were probably meant to go.
    """
    data, random = _catalogs()
    common = dict(nmesh=NMESH, boxsize=BOXSIZE, boxcentre=[BOXSIZE / 2] * 3,
                  padding=0.0, smoothing=10.0, los=None, pbc=False,
                  dtype=np.float64, f=0.8, bias=1.5, solver_type="ifft")

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        reconstruct_positions(data, random, nmsh=32, **common)


def test_align_cone_is_rejected_with_a_pointer_to_the_pipeline():
    """Cone alignment lives in the pipeline, and asking for it here is an error.

    Silently ignoring it would be worse than refusing: a caller would believe the
    catalogue had been aligned.
    """
    data, random = _catalogs()
    common = dict(nmesh=NMESH, boxsize=BOXSIZE, boxcentre=[BOXSIZE / 2] * 3,
                  padding=0.0, smoothing=10.0, los=None, pbc=False,
                  dtype=np.float64, f=0.8, bias=1.5, solver_type="ifft")

    with pytest.raises(TypeError, match="ReconstructionPipeline"):
        reconstruct_positions(data, random, align_cone=True, **common)


def test_solver_options_are_pointed_at_solver_args():
    data, random = _catalogs()
    common = dict(nmesh=NMESH, boxsize=BOXSIZE, boxcentre=[BOXSIZE / 2] * 3,
                  padding=0.0, smoothing=10.0, los=None, pbc=False,
                  dtype=np.float64, f=0.8, bias=1.5, solver_type="multigrid")

    with pytest.raises(TypeError, match=r"solver_args"):
        reconstruct_positions(data, random, smoother="mcgs", **common)

    # ...and through solver_args it is accepted.
    shifted, _ = reconstruct_positions(
        data, random, solver_args={"smoother": "mcgs"}, **common)
    assert np.all(np.isfinite(shifted))
