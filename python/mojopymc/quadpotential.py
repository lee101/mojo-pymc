"""PyMC-compatible dense quadratic potentials and adaptation estimators."""

from __future__ import annotations

import warnings

import numpy as np

from ._lib import addr, cpu_context, f64, lib, parallel_lock


_ELEMENT_PARALLEL_THRESHOLD = 262_144
_ELEMENT_SCRATCH = np.empty(8, dtype=np.float64)
_DENSE_NUMPY_THRESHOLD = 1_024
_CBLAS_MAX_DIMENSION = np.iinfo(np.int32).max


class PositiveDefiniteError(ValueError):
    def __init__(self, msg, idx):
        super().__init__(msg)
        self.idx = idx
        self.msg = msg

    def __str__(self):
        return f"Scaling is not positive definite: {self.msg}. Check indexes {self.idx}."


def partial_check_positive_definite(C):
    C = np.asarray(C)
    d = C if C.ndim == 1 else np.diag(C)
    (indices,) = np.nonzero(np.logical_or(np.isnan(d), d <= 0))
    if len(indices):
        raise PositiveDefiniteError(
            "Simple check failed. Diagonal contains negatives", indices
        )


class QuadPotential:
    dtype = np.dtype("float64")

    def __init__(self, rng=None):
        self.rng = np.random.default_rng(rng)

    def update(self, sample, grad, tune):
        pass

    def raise_ok(self, map_info=None):
        return None

    def reset(self):
        pass

    def stats(self):
        return {"largest_eigval": np.nan, "smallest_eigval": np.nan}

    def set_rng(self, rng):
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _array1(value, name):
    result = f64(value)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def _output1(value, name, size):
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.ndim != 1 or value.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if value.dtype != np.dtype("float64") or not value.flags.c_contiguous:
        raise TypeError(f"{name} must be a C-contiguous float64 array")
    if not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def _check_size(value, size, name="x"):
    if value.size != size:
        raise ValueError(f"{name} must have length {size}")


class QuadPotentialDiag(QuadPotential):
    """Quadratic potential represented by a diagonal covariance."""

    def __init__(self, v, dtype=None, rng=None):
        self.dtype = np.dtype(np.float64 if dtype is None else dtype)
        if self.dtype != np.dtype("float64"):
            raise TypeError("mojo-pymc kernels require dtype=float64")
        self.v = _array1(v, "v").astype(self.dtype, copy=False)
        self._n = self.v.size
        self.s = self.v**0.5
        self.inv_s = 1.0 / self.s
        self.rng = np.random.default_rng(rng)

    def _velocity_energy(self, x, destination):
        _check_size(x, self._n)
        destination = _output1(destination, "out", self._n)
        if not self._n:
            return 0.0
        if x.size >= _ELEMENT_PARALLEL_THRESHOLD:
            with parallel_lock:
                ctx = cpu_context()
                if ctx:
                    return lib().mpmc_diag_velocity_energy_parallel(
                        ctx,
                        addr(x),
                        addr(self.v),
                        addr(destination),
                        addr(_ELEMENT_SCRATCH),
                        x.size,
                    )
        return lib().mpmc_diag_velocity_energy(
            addr(x), addr(self.v), addr(destination), x.size
        )

    def velocity(self, x, out=None):
        x = _array1(x, "x")
        destination = np.empty_like(x) if out is None else out
        self._velocity_energy(x, destination)
        if out is None:
            return destination
        return None

    def random(self):
        return self.rng.normal(size=self.s.shape).astype(self.dtype) * self.inv_s

    def energy(self, x, velocity=None):
        x = _array1(x, "x")
        if velocity is None:
            velocity = np.empty_like(x)
            return self._velocity_energy(x, velocity)
        velocity = _array1(velocity, "velocity")
        _check_size(velocity, self._n, "velocity")
        return 0.5 * float(np.dot(x, velocity))

    def velocity_energy(self, x, v_out):
        x = _array1(x, "x")
        return self._velocity_energy(x, v_out)

    def _leapfrog_first(self, q, momentum, grad, velocity, epsilon):
        for value, name in ((q, "q"), (momentum, "momentum"), (velocity, "velocity")):
            _output1(value, name, self._n)
        _check_size(grad, self._n, "grad")
        if q.size >= _ELEMENT_PARALLEL_THRESHOLD:
            with parallel_lock:
                ctx = cpu_context()
                if ctx:
                    lib().mpmc_leapfrog_first_diag_parallel(
                        ctx,
                        addr(q),
                        addr(momentum),
                        addr(grad),
                        addr(self.v),
                        addr(velocity),
                        epsilon,
                        q.size,
                    )
                    return
        lib().mpmc_leapfrog_first_diag(
            addr(q), addr(momentum), addr(grad), addr(self.v), addr(velocity),
            epsilon, q.size,
        )


class QuadPotentialFull(QuadPotential):
    """Quadratic potential represented by a dense covariance."""

    def __init__(self, cov, dtype=None, rng=None):
        self.dtype = np.dtype(np.float64 if dtype is None else dtype)
        if self.dtype != np.dtype("float64"):
            raise TypeError("mojo-pymc kernels require dtype=float64")
        self._cov = f64(cov, copy=True)
        if self._cov.ndim != 2 or self._cov.shape[0] != self._cov.shape[1]:
            raise ValueError("cov must be a square matrix")
        self._chol = np.linalg.cholesky(self._cov)
        self._n = len(self._cov)
        if self._n > _CBLAS_MAX_DIMENSION:
            raise ValueError("dense matrix dimension exceeds the CBLAS int32 limit")
        self.rng = np.random.default_rng(rng)

    def _velocity_energy(self, x, destination):
        _check_size(x, self._n)
        destination = _output1(destination, "out", self._n)
        if np.shares_memory(x, destination):
            raise ValueError("out must not overlap x for dense matrix-vector products")
        if not self._n:
            return 0.0
        if self._n <= _DENSE_NUMPY_THRESHOLD:
            np.dot(self._cov, x, out=destination)
            return 0.5 * float(np.dot(x, destination))
        return lib().mpmc_full_velocity_energy(
            addr(x), addr(self._cov), addr(destination), self._n
        )

    def velocity(self, x, out=None):
        x = _array1(x, "x")
        destination = np.empty_like(x) if out is None else out
        _check_size(x, self._n)
        destination = _output1(destination, "out", self._n)
        if np.shares_memory(x, destination):
            raise ValueError("out must not overlap x for dense matrix-vector products")
        if not self._n:
            return destination if out is None else None
        if self._n <= _DENSE_NUMPY_THRESHOLD:
            np.dot(self._cov, x, out=destination)
        else:
            lib().mpmc_full_velocity(
                addr(x), addr(self._cov), addr(destination), self._n
            )
        if out is None:
            return destination
        return None

    def random(self):
        vals = self.rng.normal(size=self._n).astype(self.dtype)
        return np.linalg.solve(self._chol.T, vals)

    def energy(self, x, velocity=None):
        x = _array1(x, "x")
        if velocity is None:
            velocity = np.empty_like(x)
            return self._velocity_energy(x, velocity)
        velocity = _array1(velocity, "velocity")
        _check_size(x, self._n)
        _check_size(velocity, self._n, "velocity")
        return 0.5 * float(np.dot(x, velocity))

    def velocity_energy(self, x, v_out):
        x = _array1(x, "x")
        return self._velocity_energy(x, v_out)

    def _leapfrog_first(self, q, momentum, grad, velocity, epsilon):
        for value, name in ((q, "q"), (momentum, "momentum"), (velocity, "velocity")):
            _output1(value, name, self._n)
        _check_size(grad, self._n, "grad")
        lib().mpmc_leapfrog_first_full(
            addr(q), addr(momentum), addr(grad), addr(self._cov), addr(velocity),
            epsilon, self._n,
        )

    __call__ = random


class QuadPotentialFullInv(QuadPotential):
    """Quadratic potential constructed from a dense precision matrix."""

    def __init__(self, A, dtype=None, rng=None):
        self.dtype = np.dtype(np.float64 if dtype is None else dtype)
        if self.dtype != np.dtype("float64"):
            raise TypeError("mojo-pymc kernels require dtype=float64")
        A = f64(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix")
        self.L = np.ascontiguousarray(np.linalg.cholesky(A))
        self._n = self.L.shape[0]
        self.rng = np.random.default_rng(rng)

    def velocity(self, x, out=None):
        x = _array1(x, "x")
        destination = np.empty_like(x) if out is None else out
        _check_size(x, self._n)
        destination = _output1(destination, "out", self._n)
        if np.shares_memory(x, destination):
            raise ValueError("out must not overlap x for triangular solves")
        if not self._n:
            return destination if out is None else None
        lib().mpmc_inv_velocity_energy(
            addr(x), addr(self.L), addr(destination), x.size
        )
        if out is None:
            return destination
        return None

    def random(self):
        vals = self.rng.normal(size=self.L.shape[0]).astype(self.dtype)
        return self.L @ vals

    def energy(self, x, velocity=None):
        x = _array1(x, "x")
        if velocity is None:
            velocity = np.empty_like(x)
            _check_size(x, self._n)
            if not self._n:
                return 0.0
            return lib().mpmc_inv_velocity_energy(
                addr(x), addr(self.L), addr(velocity), x.size
            )
        velocity = _array1(velocity, "velocity")
        _check_size(x, self._n)
        _check_size(velocity, self._n, "velocity")
        return 0.5 * float(np.dot(x, velocity))

    def velocity_energy(self, x, v_out):
        x = _array1(x, "x")
        _check_size(x, self._n)
        v_out = _output1(v_out, "v_out", self._n)
        if np.shares_memory(x, v_out):
            raise ValueError("v_out must not overlap x for triangular solves")
        if not self._n:
            return 0.0
        return lib().mpmc_inv_velocity_energy(
            addr(x), addr(self.L), addr(v_out), x.size
        )

    def _leapfrog_first(self, q, momentum, grad, velocity, epsilon):
        for value, name in ((q, "q"), (momentum, "momentum"), (velocity, "velocity")):
            _output1(value, name, self._n)
        _check_size(grad, self._n, "grad")
        lib().mpmc_leapfrog_first_inv(
            addr(q), addr(momentum), addr(grad), addr(self.L), addr(velocity),
            epsilon, self._n,
        )


def quad_potential(C, is_cov, rng=None):
    C = np.asarray(C)
    partial_check_positive_definite(C)
    if C.ndim == 1:
        return QuadPotentialDiag(C if is_cov else 1.0 / C, rng=rng)
    if C.ndim != 2:
        raise ValueError("C must be one- or two-dimensional")
    return QuadPotentialFull(C, rng=rng) if is_cov else QuadPotentialFullInv(C, rng=rng)


def isquadpotential(value):
    return isinstance(value, QuadPotential)


class _WeightedVariance:
    def __init__(
        self, nelem, initial_mean=None, initial_variance=None, initial_weight=0, dtype="d"
    ):
        self._dtype = dtype
        self.n_samples = float(initial_weight)
        self.mean = (
            np.zeros(nelem, dtype="d")
            if initial_mean is None
            else np.array(initial_mean, dtype="d", copy=True)
        )
        self.raw_var = (
            np.zeros(nelem, dtype="d")
            if initial_variance is None
            else np.array(initial_variance, dtype="d", copy=True)
        )
        self.raw_var *= self.n_samples
        if self.raw_var.shape != (nelem,):
            raise ValueError("Invalid shape for initial variance.")
        if self.mean.shape != (nelem,):
            raise ValueError("Invalid shape for initial mean.")

    def add_sample(self, x):
        x = _array1(x, "x")
        _check_size(x, self.mean.size)
        self.n_samples += 1
        lib().mpmc_welford_var_add(
            addr(x), addr(self.mean), addr(self.raw_var), self.n_samples, x.size
        )

    def current_variance(self, out=None):
        if self.n_samples == 0:
            raise ValueError("Can not compute variance without samples.")
        if out is not None:
            return np.divide(self.raw_var, self.n_samples, out=out)
        return (self.raw_var / self.n_samples).astype(self._dtype)

    def current_mean(self):
        return self.mean.astype(self._dtype, copy=True)


class _ExpWeightedVariance:
    def __init__(self, n_vars, *, init_mean, init_var, alpha):
        self._variance = f64(init_var, copy=True)
        self._mean = f64(init_mean, copy=True)
        self._alpha = alpha
        if self._mean.shape != (n_vars,) or self._variance.shape != (n_vars,):
            raise ValueError("Invalid shape")

    def add_sample(self, value):
        value = _array1(value, "value")
        _check_size(value, self._mean.size, "value")
        lib().mpmc_exp_var_add(
            addr(value), addr(self._mean), addr(self._variance), self._alpha, value.size
        )

    def current_variance(self, out=None):
        destination = np.empty_like(self._variance) if out is None else out
        np.copyto(destination, self._variance)
        return destination

    def current_mean(self, out=None):
        destination = np.empty_like(self._mean) if out is None else out
        np.copyto(destination, self._mean)
        return destination


class _WeightedCovariance:
    def __init__(
        self, nelem, initial_mean=None, initial_covariance=None, initial_weight=0,
        dtype="d",
    ):
        self._dtype = dtype
        self.n_samples = float(initial_weight)
        self.mean = (
            np.zeros(nelem, dtype="d")
            if initial_mean is None
            else np.array(initial_mean, dtype="d", copy=True)
        )
        self.raw_cov = (
            np.eye(nelem, dtype="d")
            if initial_covariance is None
            else np.array(initial_covariance, dtype="d", copy=True)
        )
        self.raw_cov *= self.n_samples
        if self.raw_cov.shape != (nelem, nelem):
            raise ValueError("Invalid shape for initial covariance.")
        if self.mean.shape != (nelem,):
            raise ValueError("Invalid shape for initial mean.")
        self._old_diff = np.empty(nelem, dtype="d")
        self._new_diff = np.empty(nelem, dtype="d")

    def add_sample(self, x):
        x = _array1(x, "x")
        _check_size(x, self.mean.size)
        self.n_samples += 1
        lib().mpmc_welford_cov_add(
            addr(x), addr(self.mean), addr(self.raw_cov),
            addr(self._old_diff), addr(self._new_diff), self.n_samples, x.size,
        )

    def current_covariance(self, out=None):
        if self.n_samples == 0:
            raise ValueError("Can not compute covariance without samples.")
        if out is not None:
            return np.divide(self.raw_cov, self.n_samples - 1, out=out)
        return (self.raw_cov / (self.n_samples - 1)).astype(self._dtype)

    def current_mean(self):
        return np.array(self.mean, dtype=self._dtype)


class QuadPotentialDiagAdapt(QuadPotentialDiag):
    """PyMC-compatible windowed diagonal mass-matrix adaptation."""

    def __init__(
        self, n, initial_mean, initial_diag=None, initial_weight=0,
        adaptation_window=101, adaptation_window_multiplier=1, dtype=None,
        discard_window=50, early_update=False, store_mass_matrix_trace=False,
        rng=None,
    ):
        initial_mean = _array1(initial_mean, "initial_mean")
        if len(initial_mean) != n:
            raise ValueError(f"Wrong shape for initial_mean: expected {n} got {len(initial_mean)}")
        if initial_diag is not None:
            initial_diag = _array1(initial_diag, "initial_diag")
            if len(initial_diag) != n:
                raise ValueError(
                    f"Wrong shape for initial_diag: expected {n} got {len(initial_diag)}"
                )
        else:
            initial_diag = np.ones(n)
            initial_weight = 1
        self.dtype = np.dtype(np.float64 if dtype is None else dtype)
        if self.dtype != np.dtype("float64"):
            raise TypeError("mojo-pymc kernels require dtype=float64")
        self._n = n
        self._discard_window = discard_window
        self._early_update = early_update
        self._initial_mean = initial_mean
        self._initial_diag = initial_diag
        self._initial_weight = initial_weight
        self.adaptation_window = adaptation_window
        self.adaptation_window_multiplier = float(adaptation_window_multiplier)
        self._store_mass_matrix_trace = store_mass_matrix_trace
        self._mass_trace = []
        self.rng = np.random.default_rng(rng)
        self.reset()

    def reset(self):
        self._var = np.array(self._initial_diag, dtype=self.dtype, copy=True)
        self.v = self._var
        self._stds = np.sqrt(self._initial_diag)
        self.s = self._stds
        self._inv_stds = 1.0 / self._stds
        self.inv_s = self._inv_stds
        self._foreground_var = _WeightedVariance(
            self._n, self._initial_mean, self._initial_diag, self._initial_weight,
            self.dtype,
        )
        self._background_var = _WeightedVariance(self._n, dtype=self.dtype)
        self._n_samples = 0

    def random(self):
        vals = self.rng.normal(size=self._n).astype(self.dtype)
        return self._inv_stds * vals

    def _update_from_weightvar(self, weightvar):
        weightvar.current_variance(out=self._var)
        np.clip(self._var, 1e-12, 1e12, out=self._var)
        np.sqrt(self._var, out=self._stds)
        np.divide(1, self._stds, out=self._inv_stds)

    def update(self, sample, grad, tune):
        if self._store_mass_matrix_trace:
            self._mass_trace.append(self._stds.copy())
        if not tune:
            return
        if self._n_samples > self._discard_window:
            self._foreground_var.add_sample(sample)
            self._background_var.add_sample(sample)
        if self._early_update or self._n_samples > self.adaptation_window:
            self._update_from_weightvar(self._foreground_var)
        if self._n_samples > 0 and self._n_samples % self.adaptation_window == 0:
            self._foreground_var = self._background_var
            self._background_var = _WeightedVariance(self._n, dtype=self.dtype)
            self.adaptation_window = int(
                self.adaptation_window * self.adaptation_window_multiplier
            )
        self._n_samples += 1


class QuadPotentialFullAdapt(QuadPotentialFull):
    """PyMC-compatible adaptive dense covariance potential."""

    def __init__(
        self, n, initial_mean, initial_cov=None, initial_weight=0,
        adaptation_window=101, adaptation_window_multiplier=2, update_window=1,
        dtype=None, rng=None,
    ):
        warnings.warn("QuadPotentialFullAdapt is an experimental feature")
        initial_mean = _array1(initial_mean, "initial_mean")
        if len(initial_mean) != n:
            raise ValueError(f"Wrong shape for initial_mean: expected {n} got {len(initial_mean)}")
        if initial_cov is None:
            initial_cov = np.eye(n)
            initial_weight = 1
        initial_cov = f64(initial_cov)
        if initial_cov.shape != (n, n):
            raise ValueError(f"Wrong shape for initial_cov: expected {n} got {initial_cov.shape}")
        self.dtype = np.dtype(np.float64 if dtype is None else dtype)
        self._n = n
        self._initial_mean = initial_mean
        self._initial_cov = initial_cov
        self._initial_weight = initial_weight
        self.adaptation_window = int(adaptation_window)
        self.adaptation_window_multiplier = float(adaptation_window_multiplier)
        self._update_window = int(update_window)
        self.rng = np.random.default_rng(rng)
        self.reset()

    def reset(self):
        self._previous_update = 0
        self._cov = np.array(self._initial_cov, dtype=self.dtype, copy=True)
        self._chol = np.linalg.cholesky(self._cov)
        self._chol_error = None
        self._foreground_cov = _WeightedCovariance(
            self._n, self._initial_mean, self._initial_cov, self._initial_weight,
            self.dtype,
        )
        self._background_cov = _WeightedCovariance(self._n, dtype=self.dtype)
        self._n_samples = 0

    def _update_from_weightvar(self, weightvar):
        weightvar.current_covariance(out=self._cov)
        try:
            self._chol = np.linalg.cholesky(self._cov)
            self._chol_error = None
        except (np.linalg.LinAlgError, ValueError) as error:
            self._chol_error = error

    def update(self, sample, grad, tune):
        if not tune:
            return
        delta = self._n_samples - self._previous_update
        self._foreground_cov.add_sample(sample)
        self._background_cov.add_sample(sample)
        if (delta + 1) % self._update_window == 0:
            self._update_from_weightvar(self._foreground_cov)
        if delta >= self.adaptation_window:
            self._foreground_cov = self._background_cov
            self._background_cov = _WeightedCovariance(self._n, dtype=self.dtype)
            self._previous_update = self._n_samples
            self.adaptation_window = int(
                self.adaptation_window * self.adaptation_window_multiplier
            )
        self._n_samples += 1

    def raise_ok(self, vmap):
        if self._chol_error is not None:
            raise ValueError(str(self._chol_error))
