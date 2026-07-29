"""Benchmarks Mojo kernels against the equivalent PyMC 5.26.1 operations."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import pymc

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import mojopymc as mojo  # noqa: E402

from pymc.blocking import RaveledVars  # noqa: E402
from pymc.step_methods.hmc import integration as py_integration  # noqa: E402
from pymc.step_methods.hmc import quadpotential as py_quad  # noqa: E402


def timeit(fn, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def measure(ours, upstream, repeat=5):
    ours()
    upstream()
    return timeit(ours, repeat), timeit(upstream, repeat)


class StandardNormalLogp:
    dtype = np.dtype("float64")
    _raveled_inputs = True

    def __init__(self):
        self._pytensor_function = self.evaluate

    @staticmethod
    def evaluate(q):
        return -0.5 * float(np.dot(q, q)), -q


def run_steps(integrator, initial, count):
    state = initial
    for _ in range(count):
        state = integrator.step(0.01, state)
    return state


def main():
    rng = np.random.default_rng(100)
    results = []

    n = 2_000_000
    diag = np.ascontiguousarray(rng.lognormal(size=n))
    momentum = np.ascontiguousarray(rng.normal(size=n))
    ours_diag = mojo.QuadPotentialDiag(diag)
    py_diag = py_quad.QuadPotentialDiag(diag)
    ours_velocity = np.empty(n)
    py_velocity = np.empty(n)
    results.append(
        (
            "diagonal velocity + energy (2M)",
            *measure(
                lambda: ours_diag.velocity_energy(momentum, ours_velocity),
                lambda: py_diag.velocity_energy(momentum, py_velocity),
            ),
        )
    )

    sample = np.ascontiguousarray(rng.normal(size=n))
    ours_var = mojo._WeightedVariance(n)
    py_var = py_quad._WeightedVariance(n)
    results.append(
        (
            "Welford variance update (2M)",
            *measure(
                lambda: ours_var.add_sample(sample),
                lambda: py_var.add_sample(sample),
            ),
        )
    )

    turn_sum = np.ascontiguousarray(rng.normal(size=n))
    left_v = np.ascontiguousarray(rng.normal(size=n))
    right_v = np.ascontiguousarray(rng.normal(size=n))
    results.append(
        (
            "NUTS turning criterion (2M)",
            *measure(
                lambda: mojo.is_turning(turn_sum, left_v, right_v),
                lambda: (turn_sum.dot(left_v) <= 0) or (turn_sum.dot(right_v) <= 0),
            ),
        )
    )

    step_n = 500_000
    step_diag = np.ascontiguousarray(rng.lognormal(size=step_n))
    q = RaveledVars(np.ascontiguousarray(rng.normal(size=step_n)), ())
    p = np.ascontiguousarray(rng.normal(size=step_n))
    logp = StandardNormalLogp()
    ours_integrator = mojo.CpuLeapfrogIntegrator(
        mojo.QuadPotentialDiag(step_diag), logp
    )
    py_integrator = py_integration.CpuLeapfrogIntegrator(
        py_quad.QuadPotentialDiag(step_diag), logp
    )
    ours_initial = ours_integrator.compute_state(q, p)
    py_initial = py_integrator.compute_state(q, p)
    results.append(
        (
            "10 leapfrog steps, diagonal (500k)",
            *measure(
                lambda: run_steps(ours_integrator, ours_initial, 10),
                lambda: run_steps(py_integrator, py_initial, 10),
                repeat=3,
            ),
        )
    )

    dense_n = 512
    dense = rng.normal(size=(dense_n, dense_n))
    cov = np.ascontiguousarray(dense @ dense.T + np.eye(dense_n))
    dense_p = np.ascontiguousarray(rng.normal(size=dense_n))
    ours_full = mojo.QuadPotentialFull(cov)
    py_full = py_quad.QuadPotentialFull(cov)
    ours_dense_v = np.empty(dense_n)
    py_dense_v = np.empty(dense_n)
    results.append(
        (
            "dense velocity + energy (512x512)",
            *measure(
                lambda: ours_full.velocity_energy(dense_p, ours_dense_v),
                lambda: py_full.velocity_energy(dense_p, py_dense_v),
                repeat=10,
            ),
        )
    )

    cpu = platform.processor()
    if not cpu or cpu.lower() in {"x86_64", "amd64"}:
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
                model_line = next(
                    line for line in cpuinfo if line.startswith("model name")
                )
            cpu = model_line.split(":", 1)[1].strip()
        except (OSError, StopIteration):
            cpu = platform.machine()
    print(
        f"Machine: {cpu}; {os.cpu_count()} logical CPUs; "
        f"{platform.system()} {platform.machine()}; "
        f"Python {platform.python_version()}; PyMC {pymc.__version__}; "
        f"NumPy {np.__version__}"
    )
    print("Method: warmed libraries; best of 5 runs (trajectory: 3; dense: 10).")
    print()
    print("| case | mojo-pymc | PyMC | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours_time, upstream_time in results:
        ratio = upstream_time / ours_time
        label = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
        print(
            f"| {name} | {ours_time * 1e3:.3f} ms | "
            f"{upstream_time * 1e3:.3f} ms | {label} |"
        )


if __name__ == "__main__":
    main()
