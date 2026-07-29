# mojo-pymc

PyMC's compute-bound Hamiltonian Monte Carlo primitives, implemented in
[Mojo](https://www.modular.com/mojo) and called from Python through a small
ctypes layer.

This is an accelerator for the numerical core of HMC and NUTS, not a second
probabilistic-programming frontend. It keeps PyMC's names and call signatures
for the covered subset, so existing potential and integrator code can usually
change only its import:

```python
import numpy as np
from pymc.blocking import RaveledVars
from mojopymc import CpuLeapfrogIntegrator, QuadPotentialDiag


class StandardNormal:
    dtype = np.dtype("float64")
    _raveled_inputs = True

    def __init__(self):
        self._pytensor_function = self.evaluate

    @staticmethod
    def evaluate(q):
        return -0.5 * q.dot(q), -q


potential = QuadPotentialDiag(np.ones(4), rng=7)
integrator = CpuLeapfrogIntegrator(potential, StandardNormal())
q = RaveledVars(np.array([0.2, -0.1, 0.4, 0.3]), ())
state = integrator.compute_state(q, potential.random())
next_state = integrator.step(0.1, state)
print(next_state.energy, next_state.q.data)
```

## Coverage

The following APIs mirror `pymc.step_methods.hmc`:

| upstream module | covered API |
| --- | --- |
| `quadpotential` | `quad_potential`, `isquadpotential`, `partial_check_positive_definite`, `QuadPotentialDiag`, `QuadPotentialFull`, `QuadPotentialFullInv` |
| `quadpotential` adaptation | `QuadPotentialDiagAdapt`, `QuadPotentialFullAdapt`, `_WeightedVariance`, `_ExpWeightedVariance`, `_WeightedCovariance` |
| `integration` | `State`, `IntegrationError`, `CpuLeapfrogIntegrator.compute_state`, `CpuLeapfrogIntegrator.step` |
| `nuts` | `Proposal`, `Subtree`, `_Tree`, plus the exposed `is_turning` primitive |

This covers diagonal and dense velocity/kinetic-energy evaluation, momentum
updates, signed leapfrog integration, online mass-matrix adaptation, recursive
NUTS tree construction, multinomial proposal selection, divergence accounting,
and the three-part generalized U-turn checks.

Not covered are PyMC's model graph construction, PyTensor log-density and
gradient compilation, dual-averaging step-size adaptation, sampler
multiprocessing, trace storage, sparse CHOLMOD potentials, GPU sampling, and
the public `pm.sample` / `pm.NUTS` orchestration. Those remain in PyMC. Kernels
require contiguous `float64`; PyMC configurations using `float32` are rejected
instead of silently converting sampler state.

## Install and verify

From a checkout, the Pixi environment supplies the pinned Mojo nightly,
Python, PyMC, and test dependencies:

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi run build` produces `dist/libmojo-pymc.so`. Set `MOJOPYMC_LIB` to use a
copy of that library from another location. Run Python examples through
`pixi run python`; the usage example above was executed that way on the
benchmark machine.

## Performance

Measured by the final `pixi run bench` gate on this machine: Intel Xeon
E5-2697 v4, 72 logical CPUs, Linux x86-64, Python 3.13.14, PyMC 5.26.1, and
NumPy 2.5.1. Times are the best of five warmed runs, except the trajectory
(best of three) and dense case (best of ten). Both implementations use
identical arrays.

| case | mojo-pymc | PyMC | result |
| --- | ---: | ---: | ---: |
| diagonal velocity + energy (2M) | 2.732 ms | 6.947 ms | 2.54x faster |
| Welford variance update (2M) | 6.459 ms | 50.756 ms | 7.86x faster |
| NUTS turning criterion (2M) | 0.899 ms | 0.639 ms | 1.41x slower |
| 10 leapfrog steps, diagonal (500k) | 51.187 ms | 63.552 ms | 1.24x faster |
| dense velocity + energy (512x512) | 0.031 ms | 0.016 ms | 1.97x slower |

The adaptation win comes from updating mean and raw variance in one
allocation-free pass, where upstream NumPy creates several full-sized
temporaries. Large diagonal kernels use a fixed worker pool and SIMD within
each aligned chunk; smaller arrays stay serial. The
diagonal leapfrog path also fuses its first momentum half-step, velocity
calculation, and position update.

Dense matrix-vector evaluation and large NUTS dot products call CBLAS directly
from Mojo while retaining the NumPy-owned buffers. Small NUTS inputs use a
four-accumulator SIMD loop with a scalar tail. In this run the two
BLAS-backed cases remain slower because ctypes/Mojo dispatch is visible at
these timings.

No GPU path is included.

## How it works

All kernels live in one Mojo compilation unit linked to the environment's
CBLAS. Python validates shapes and dtypes, owns arrays and scratch space, and
invokes one C-ABI symbol per operation. Since exported Mojo functions cannot
be parametric, every buffer crosses ctypes as a 64-bit integer address and is
reconstructed inside Mojo as
`UnsafePointer[Float64, AnyOrigin[mut=True]]`.

Vectors are contiguous `float64`. Dense mass and precision matrices are
C-contiguous row-major arrays. Dense covariance potentials store the lower
Cholesky factor for momentum draws; inverse potentials use forward and
backward triangular solves in Mojo. Nothing in the shared library allocates or
retains a Python-owned address.

The integrator deliberately leaves the model-specific log-density/gradient
call between the two leapfrog halves in Python. NUTS tree control flow also
remains Python-visible, while its repeated vector reductions and every
integration step enter compiled Mojo. Tests compare values, seeded random
draws, adaptation state, complete leapfrog states, and deterministic tree
extensions directly against installed PyMC.

## License

MIT
