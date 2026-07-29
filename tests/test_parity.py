"""Numerical and behavioural parity with PyMC's HMC implementation."""

from __future__ import annotations

import numpy as np
import pytest

import mojopymc as mojo
import mojopymc.quadpotential as mojo_quad

from pymc.blocking import RaveledVars
from pymc.step_methods.hmc import integration as py_integration
from pymc.step_methods.hmc import nuts as py_nuts
from pymc.step_methods.hmc import quadpotential as py_quad


class GaussianLogp:
    dtype = np.dtype("float64")
    _raveled_inputs = True

    def __init__(self, precision):
        self.precision = np.ascontiguousarray(precision, dtype=np.float64)
        self._pytensor_function = self.evaluate

    def evaluate(self, q):
        grad = -(self.precision @ q)
        return 0.5 * float(q @ grad), grad


def spd(n, seed=1):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, n))
    return np.ascontiguousarray(matrix @ matrix.T + np.eye(n))


@pytest.mark.parametrize("n", [1, 7, 64, 1003])
def test_diag_velocity_and_energy(n):
    rng = np.random.default_rng(n)
    diag = rng.lognormal(size=n)
    x = rng.normal(size=n)
    ours = mojo.QuadPotentialDiag(diag)
    theirs = py_quad.QuadPotentialDiag(diag)
    assert np.allclose(ours.velocity(x), theirs.velocity(x), rtol=1e-14)
    assert ours.energy(x) == pytest.approx(theirs.energy(x), rel=1e-13)
    ours_out = np.empty_like(x)
    theirs_out = np.empty_like(x)
    assert ours.velocity(x, out=ours_out) is None
    assert theirs.velocity(x, out=theirs_out) is None
    assert np.array_equal(ours_out, theirs_out)
    assert ours.velocity_energy(x, ours_out) == pytest.approx(
        theirs.velocity_energy(x, theirs_out), rel=1e-13
    )


def test_diag_parallel_threshold_and_simd_tail(monkeypatch):
    n = 1_003
    rng = np.random.default_rng(1003)
    diag = rng.lognormal(size=n)
    x = rng.normal(size=n)
    potential = mojo.QuadPotentialDiag(diag)
    expected_velocity = diag * x
    expected_energy = 0.5 * float(x @ expected_velocity)

    monkeypatch.setattr(mojo_quad, "_ELEMENT_PARALLEL_THRESHOLD", n + 1)
    serial_velocity = np.empty_like(x)
    serial_energy = potential.velocity_energy(x, serial_velocity)

    monkeypatch.setattr(mojo_quad, "_ELEMENT_PARALLEL_THRESHOLD", n)
    parallel_velocity = np.empty_like(x)
    parallel_energy = potential.velocity_energy(x, parallel_velocity)

    assert np.array_equal(serial_velocity, expected_velocity)
    assert np.array_equal(parallel_velocity, expected_velocity)
    assert serial_energy == pytest.approx(expected_energy, rel=1e-13)
    assert parallel_energy == pytest.approx(expected_energy, rel=1e-13)


def test_diag_parallel_context_failure_falls_back(monkeypatch):
    n = 19
    rng = np.random.default_rng(1019)
    diag = rng.lognormal(size=n)
    x = rng.normal(size=n)
    potential = mojo.QuadPotentialDiag(diag)
    monkeypatch.setattr(mojo_quad, "_ELEMENT_PARALLEL_THRESHOLD", 1)
    monkeypatch.setattr(mojo_quad, "cpu_context", lambda: 0)
    assert np.array_equal(potential.velocity(x), diag * x)


@pytest.mark.parametrize("potential_name", ["full", "full_inv"])
def test_dense_velocity_energy_and_random(potential_name):
    matrix = spd(13)
    x = np.random.default_rng(3).normal(size=13)
    if potential_name == "full":
        ours = mojo.QuadPotentialFull(matrix, rng=11)
        theirs = py_quad.QuadPotentialFull(matrix, rng=11)
    else:
        ours = mojo.QuadPotentialFullInv(matrix, rng=11)
        theirs = py_quad.QuadPotentialFullInv(matrix, rng=11)
    assert np.allclose(ours.velocity(x), theirs.velocity(x), rtol=2e-13, atol=1e-13)
    assert ours.energy(x) == pytest.approx(theirs.energy(x), rel=2e-13)
    assert np.allclose(ours.random(), theirs.random(), rtol=2e-13, atol=1e-13)


def test_diag_random_matches_upstream_seed():
    diag = np.array([0.2, 1.0, 3.0, 20.0])
    ours = mojo.QuadPotentialDiag(diag, rng=941)
    theirs = py_quad.QuadPotentialDiag(diag, rng=941)
    for _ in range(4):
        assert np.array_equal(ours.random(), theirs.random())


@pytest.mark.parametrize("is_cov", [False, True])
@pytest.mark.parametrize("ndim", [1, 2])
def test_quad_potential_factory(is_cov, ndim):
    matrix = np.array([1.0, 2.0, 4.0]) if ndim == 1 else spd(3)
    ours = mojo.quad_potential(matrix, is_cov, rng=8)
    theirs = py_quad.quad_potential(matrix, is_cov, rng=8)
    x = np.array([0.5, -1.5, 2.0])
    assert np.allclose(ours.velocity(x), theirs.velocity(x), rtol=2e-13)
    assert ours.energy(x) == pytest.approx(theirs.energy(x), rel=2e-13)
    assert mojo.isquadpotential(ours)
    assert not mojo.isquadpotential(matrix)


@pytest.mark.parametrize(
    "bad", [np.array([1.0, 0.0]), np.array([1.0, -2.0]), np.array([1.0, np.nan])]
)
def test_positive_definite_validation(bad):
    with pytest.raises(mojo.PositiveDefiniteError) as ours:
        mojo.quad_potential(bad, True)
    with pytest.raises(py_quad.PositiveDefiniteError) as theirs:
        py_quad.quad_potential(bad, True)
    assert np.array_equal(ours.value.idx, theirs.value.idx)


def test_weighted_variance_matches_upstream():
    rng = np.random.default_rng(12)
    initial_mean = rng.normal(size=257)
    initial_var = rng.lognormal(size=257)
    ours = mojo._WeightedVariance(257, initial_mean, initial_var, 4)
    theirs = py_quad._WeightedVariance(257, initial_mean, initial_var, 4)
    for _ in range(25):
        sample = rng.normal(size=257)
        ours.add_sample(sample)
        theirs.add_sample(sample)
    assert ours.n_samples == theirs.n_samples
    assert np.allclose(ours.mean, theirs.mean, rtol=1e-14, atol=1e-14)
    assert np.allclose(ours.raw_var, theirs.raw_var, rtol=1e-14, atol=1e-14)
    assert np.allclose(ours.current_variance(), theirs.current_variance(), rtol=1e-14)


def test_exp_weighted_variance_matches_upstream():
    rng = np.random.default_rng(14)
    initial_mean = rng.normal(size=129)
    initial_var = rng.random(size=129)
    ours = mojo._ExpWeightedVariance(
        129, init_mean=initial_mean, init_var=initial_var, alpha=0.07
    )
    theirs = py_quad._ExpWeightedVariance(
        129, init_mean=initial_mean.copy(), init_var=initial_var.copy(), alpha=0.07
    )
    for _ in range(20):
        sample = rng.normal(size=129)
        ours.add_sample(sample)
        theirs.add_sample(sample)
    assert np.allclose(ours.current_mean(), theirs.current_mean(), rtol=1e-14)
    assert np.allclose(ours.current_variance(), theirs.current_variance(), rtol=1e-14)


def test_weighted_covariance_matches_upstream():
    rng = np.random.default_rng(15)
    initial_mean = rng.normal(size=11)
    initial_cov = spd(11, seed=16)
    ours = mojo._WeightedCovariance(11, initial_mean, initial_cov, 3)
    theirs = py_quad._WeightedCovariance(11, initial_mean, initial_cov, 3)
    for _ in range(30):
        sample = rng.normal(size=11)
        ours.add_sample(sample)
        theirs.add_sample(sample)
    assert np.allclose(ours.mean, theirs.mean, rtol=1e-14, atol=1e-14)
    assert np.allclose(ours.raw_cov, theirs.raw_cov, rtol=1e-14, atol=1e-14)
    assert np.allclose(
        ours.current_covariance(), theirs.current_covariance(), rtol=1e-14
    )


def test_diag_adaptation_matches_upstream():
    rng = np.random.default_rng(17)
    kwargs = dict(
        n=19,
        initial_mean=np.zeros(19),
        initial_diag=np.ones(19),
        initial_weight=2,
        discard_window=2,
        early_update=True,
        adaptation_window=7,
        adaptation_window_multiplier=2,
    )
    ours = mojo.QuadPotentialDiagAdapt(**kwargs, rng=18)
    theirs = py_quad.QuadPotentialDiagAdapt(**kwargs, rng=18)
    for index in range(35):
        sample = rng.normal(size=19)
        grad = rng.normal(size=19)
        tune = index != 20
        ours.update(sample, grad, tune)
        theirs.update(sample, grad, tune)
        assert np.allclose(ours._var, theirs._var, rtol=1e-13, atol=1e-14)
    assert ours.adaptation_window == theirs.adaptation_window
    assert np.allclose(ours.random(), theirs.random(), rtol=1e-13)


def test_full_adaptation_matches_upstream():
    rng = np.random.default_rng(19)
    kwargs = dict(
        n=6,
        initial_mean=np.zeros(6),
        initial_cov=np.eye(6),
        initial_weight=2,
        adaptation_window=8,
        update_window=3,
    )
    with pytest.warns(UserWarning):
        ours = mojo.QuadPotentialFullAdapt(**kwargs, rng=20)
    with pytest.warns(UserWarning):
        theirs = py_quad.QuadPotentialFullAdapt(**kwargs, rng=20)
    for _ in range(24):
        sample = rng.normal(size=6)
        grad = rng.normal(size=6)
        ours.update(sample, grad, True)
        theirs.update(sample, grad, True)
        assert np.allclose(ours._cov, theirs._cov, rtol=2e-13, atol=1e-14)
    assert np.allclose(ours.velocity(sample), theirs.velocity(sample), rtol=2e-13)


@pytest.mark.parametrize("kind", ["diag", "full", "full_inv"])
def test_leapfrog_step_matches_upstream(kind):
    rng = np.random.default_rng(21)
    n = 17
    mass = rng.lognormal(size=n) if kind == "diag" else spd(n, seed=22)
    if kind == "diag":
        ours_pot = mojo.QuadPotentialDiag(mass)
        theirs_pot = py_quad.QuadPotentialDiag(mass)
    elif kind == "full":
        ours_pot = mojo.QuadPotentialFull(mass)
        theirs_pot = py_quad.QuadPotentialFull(mass)
    else:
        ours_pot = mojo.QuadPotentialFullInv(mass)
        theirs_pot = py_quad.QuadPotentialFullInv(mass)
    logp = GaussianLogp(spd(n, seed=23))
    ours = mojo.CpuLeapfrogIntegrator(ours_pot, logp)
    theirs = py_integration.CpuLeapfrogIntegrator(theirs_pot, logp)
    q = RaveledVars(rng.normal(size=n), ())
    p = rng.normal(size=n)
    ours_state = ours.compute_state(q, p)
    theirs_state = theirs.compute_state(q, p)
    for epsilon in [0.01, 0.01, -0.005, 0.02]:
        ours_state = ours.step(epsilon, ours_state)
        theirs_state = theirs.step(epsilon, theirs_state)
        for ours_value, theirs_value in [
            (ours_state.q.data, theirs_state.q.data),
            (ours_state.p, theirs_state.p),
            (ours_state.v, theirs_state.v),
            (ours_state.q_grad, theirs_state.q_grad),
        ]:
            assert np.allclose(ours_value, theirs_value, rtol=4e-12, atol=2e-12)
        assert ours_state.energy == pytest.approx(theirs_state.energy, rel=3e-12)
        assert ours_state.index_in_trajectory == theirs_state.index_in_trajectory


def test_leapfrog_harmonic_energy_is_bounded():
    n = 128
    logp = GaussianLogp(np.eye(n))
    integrator = mojo.CpuLeapfrogIntegrator(mojo.QuadPotentialDiag(np.ones(n)), logp)
    rng = np.random.default_rng(24)
    state = integrator.compute_state(RaveledVars(rng.normal(size=n), ()), rng.normal(size=n))
    initial_energy = state.energy
    for _ in range(1000):
        state = integrator.step(0.03, state)
    assert abs(state.energy - initial_energy) < 0.02


def test_parallel_diag_leapfrog_matches_upstream(monkeypatch):
    n = 19
    rng = np.random.default_rng(1021)
    diag = rng.lognormal(size=n)
    logp = GaussianLogp(spd(n, seed=1022))
    ours = mojo.CpuLeapfrogIntegrator(mojo.QuadPotentialDiag(diag), logp)
    theirs = py_integration.CpuLeapfrogIntegrator(py_quad.QuadPotentialDiag(diag), logp)
    q = RaveledVars(rng.normal(size=n), ())
    p = rng.normal(size=n)
    ours_state = ours.compute_state(q, p)
    theirs_state = theirs.compute_state(q, p)
    monkeypatch.setattr(mojo_quad, "_ELEMENT_PARALLEL_THRESHOLD", 1)
    ours_state = ours.step(0.01, ours_state)
    theirs_state = theirs.step(0.01, theirs_state)
    assert np.allclose(ours_state.q.data, theirs_state.q.data, rtol=4e-12, atol=2e-12)
    assert np.allclose(ours_state.p, theirs_state.p, rtol=4e-12, atol=2e-12)
    assert ours_state.energy == pytest.approx(theirs_state.energy, rel=3e-12)


@pytest.mark.parametrize(
    "p_sum,left,right,expected",
    [
        ([1.0, 0.0], [1.0, 1.0], [2.0, -1.0], False),
        ([1.0, 0.0], [-1.0, 1.0], [2.0, -1.0], True),
        ([1.0, 1.0], [1.0, 1.0], [-2.0, 1.0], True),
        ([1.0, -1.0], [1.0, 1.0], [1.0, -1.0], True),
    ],
)
def test_nuts_turning_criterion(p_sum, left, right, expected):
    p_sum = np.array(p_sum)
    left = np.array(left)
    right = np.array(right)
    upstream = (p_sum.dot(left) <= 0) or (p_sum.dot(right) <= 0)
    assert mojo.is_turning(p_sum, left, right) == bool(upstream) == expected


@pytest.mark.parametrize("n", [8_191, 8_192, 8_197])
def test_nuts_turning_simd_tail_and_blas_threshold(n):
    rng = np.random.default_rng(n)
    p_sum = rng.normal(size=n)
    left = rng.normal(size=n)
    right = rng.normal(size=n)
    upstream = (p_sum.dot(left) <= 0) or (p_sum.dot(right) <= 0)
    assert mojo.is_turning(p_sum, left, right) == bool(upstream)


def test_nuts_tree_matches_upstream():
    n = 9
    rng = np.random.default_rng(25)
    diag = rng.lognormal(size=n)
    logp = GaussianLogp(spd(n, seed=26))
    ours_integrator = mojo.CpuLeapfrogIntegrator(mojo.QuadPotentialDiag(diag), logp)
    theirs_integrator = py_integration.CpuLeapfrogIntegrator(
        py_quad.QuadPotentialDiag(diag), logp
    )
    q = RaveledVars(rng.normal(size=n), ())
    p = rng.normal(size=n)
    ours_start = ours_integrator.compute_state(q, p)
    theirs_start = theirs_integrator.compute_state(q, p)
    ours = mojo._Tree(
        n, ours_integrator, ours_start, 0.025, 1000.0, np.random.default_rng(27)
    )
    theirs = py_nuts._Tree(
        n, theirs_integrator, theirs_start, 0.025, 1000.0, np.random.default_rng(27)
    )
    for direction in [1, -1, 1, 1, -1]:
        ours_diverging, ours_turning = ours.extend(direction)
        theirs_diverging, theirs_turning = theirs.extend(direction)
        assert bool(ours_diverging) == bool(theirs_diverging)
        assert ours_turning == theirs_turning
        assert np.allclose(ours.p_sum, theirs.p_sum, rtol=3e-12, atol=2e-12)
        assert np.allclose(
            ours.proposal.q.data, theirs.proposal.q.data, rtol=3e-12, atol=2e-12
        )
        if ours_turning:
            break
    ours_stats = ours.stats()
    theirs_stats = theirs.stats()
    assert ours_stats.keys() == theirs_stats.keys()
    for key in ours_stats:
        assert ours_stats[key] == pytest.approx(theirs_stats[key], rel=4e-12, abs=2e-12)


def test_dtype_mismatch_matches_upstream_error():
    class WrongDtypeLogp(GaussianLogp):
        dtype = np.dtype("float32")

    logp = WrongDtypeLogp(np.eye(2))
    with pytest.raises(ValueError, match="dtypes of potential"):
        mojo.CpuLeapfrogIntegrator(mojo.QuadPotentialDiag(np.ones(2)), logp)


@pytest.mark.parametrize(
    "bad",
    [
        np.ones(4, dtype=np.float32),
        np.ones(4, dtype=np.int64),
    ],
)
def test_kernel_inputs_reject_silent_dtype_conversion(bad):
    potential = mojo.QuadPotentialDiag(np.ones(4))
    with pytest.raises(TypeError, match="dtype=float64"):
        potential.velocity(bad)
    with pytest.raises(TypeError, match="dtype=float64"):
        mojo.is_turning(bad, np.ones(4), np.ones(4))


def test_kernel_inputs_reject_noncontiguous_arrays():
    value = np.ones(8)[::2]
    with pytest.raises(TypeError, match="C-contiguous"):
        mojo.QuadPotentialDiag(np.ones(4)).velocity(value)


@pytest.mark.parametrize("kind", ["diag", "full", "full_inv"])
def test_potentials_validate_ffi_lengths_and_outputs(kind):
    matrix = np.ones(4) if kind == "diag" else np.eye(4)
    cls = {
        "diag": mojo.QuadPotentialDiag,
        "full": mojo.QuadPotentialFull,
        "full_inv": mojo.QuadPotentialFullInv,
    }[kind]
    potential = cls(matrix)

    with pytest.raises(ValueError, match="length 4"):
        potential.velocity(np.ones(3))
    with pytest.raises(ValueError, match="shape"):
        potential.velocity(np.ones(4), out=np.empty(3))
    with pytest.raises(TypeError, match="float64"):
        potential.velocity(np.ones(4), out=np.empty(4, dtype=np.float32))

    readonly = np.empty(4)
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        potential.velocity(np.ones(4), out=readonly)


@pytest.mark.parametrize("kind", ["full", "full_inv"])
def test_dense_outputs_cannot_alias_inputs(kind):
    cls = mojo.QuadPotentialFull if kind == "full" else mojo.QuadPotentialFullInv
    potential = cls(np.eye(4))
    value = np.ones(4)
    with pytest.raises(ValueError, match="overlap"):
        potential.velocity(value, out=value)


def test_adaptation_rejects_wrong_sample_length():
    estimators = [
        mojo._WeightedVariance(4),
        mojo._ExpWeightedVariance(
            4, init_mean=np.zeros(4), init_var=np.ones(4), alpha=0.1
        ),
        mojo._WeightedCovariance(4),
    ]
    for estimator in estimators:
        with pytest.raises(ValueError, match="length 4"):
            estimator.add_sample(np.ones(3))


def test_compute_state_validates_momentum_and_gradient_lengths():
    potential = mojo.QuadPotentialDiag(np.ones(4))
    integrator = mojo.CpuLeapfrogIntegrator(potential, GaussianLogp(np.eye(4)))
    q = RaveledVars(np.ones(4), ())
    with pytest.raises(ValueError, match="p and q"):
        integrator.compute_state(q, np.ones(3))

    class BadGradient(GaussianLogp):
        def evaluate(self, q):
            return -1.0, np.ones(3)

    bad_integrator = mojo.CpuLeapfrogIntegrator(potential, BadGradient(np.eye(4)))
    with pytest.raises(ValueError, match="gradient and q"):
        bad_integrator.compute_state(q, np.ones(4))


@pytest.mark.parametrize("kind", ["diag", "full", "full_inv"])
def test_step_rejects_malformed_state_before_ffi(kind):
    matrix = np.ones(4) if kind == "diag" else np.eye(4)
    cls = {
        "diag": mojo.QuadPotentialDiag,
        "full": mojo.QuadPotentialFull,
        "full_inv": mojo.QuadPotentialFullInv,
    }[kind]
    potential = cls(matrix)
    integrator = mojo.CpuLeapfrogIntegrator(potential, GaussianLogp(np.eye(4)))
    state = integrator.compute_state(RaveledVars(np.ones(4), ()), np.ones(4))
    malformed = state._replace(q=RaveledVars(np.ones(3), ()))
    with pytest.raises(ValueError, match="shape|length"):
        integrator.step(0.1, malformed)
