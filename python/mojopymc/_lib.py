"""ctypes loader for the Mojo HMC kernels."""

from __future__ import annotations

import atexit
import ctypes
import os
import threading

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJOPYMC_LIB") or os.path.join(ROOT, "dist", "libmojo-pymc.so")

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mpmc_create_cpu_context": ([], I),
    "mpmc_destroy_cpu_context": ([I], None),
    "mpmc_diag_velocity_energy": ([I, I, I, I], F),
    "mpmc_diag_velocity_energy_parallel": ([I, I, I, I, I, I], F),
    "mpmc_full_velocity": ([I, I, I, I], None),
    "mpmc_full_velocity_energy": ([I, I, I, I], F),
    "mpmc_inv_velocity_energy": ([I, I, I, I], F),
    "mpmc_add_scaled": ([I, I, F, I], None),
    "mpmc_leapfrog_first_diag": ([I, I, I, I, I, F, I], None),
    "mpmc_leapfrog_first_diag_parallel": ([I, I, I, I, I, I, F, I], None),
    "mpmc_leapfrog_first_full": ([I, I, I, I, I, F, I], None),
    "mpmc_leapfrog_first_inv": ([I, I, I, I, I, F, I], None),
    "mpmc_welford_var_add": ([I, I, I, F, I], None),
    "mpmc_exp_var_add": ([I, I, I, F, I], None),
    "mpmc_welford_cov_add": ([I, I, I, I, I, F, I], None),
    "mpmc_is_turning": ([I, I, I, I], I),
}

_library: ctypes.CDLL | None = None
_cpu_context: int | None = None
_library_lock = threading.Lock()
_context_lock = threading.Lock()
parallel_lock = threading.Lock()


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        with _library_lock:
            if _library is None:
                if not os.path.isfile(LIB):
                    raise RuntimeError("Mojo library is missing; run `pixi run build`")
                loaded = ctypes.CDLL(LIB)
                for name, (argtypes, restype) in _SIGNATURES.items():
                    fn = getattr(loaded, name)
                    fn.argtypes = argtypes
                    fn.restype = restype
                _library = loaded
    return _library


def f64(value, *, copy: bool = False) -> np.ndarray:
    original = np.asarray(value)
    if isinstance(value, np.ndarray):
        if original.dtype != np.dtype("float64"):
            raise TypeError("arrays passed to mojo-pymc kernels must have dtype=float64")
        if not original.flags.c_contiguous:
            raise TypeError("arrays passed to mojo-pymc kernels must be C-contiguous")
    if copy:
        return np.array(value, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(value, dtype=np.float64)


def addr(value: np.ndarray) -> int:
    pointer = int(value.ctypes.data)
    if value.size and pointer == 0:
        raise ValueError("non-empty arrays must have a non-null data pointer")
    return pointer


def cpu_context() -> int:
    global _cpu_context
    if _cpu_context is None:
        with _context_lock:
            if _cpu_context is None:
                _cpu_context = int(lib().mpmc_create_cpu_context())
    return _cpu_context


def close_cpu_context() -> None:
    global _cpu_context
    if _cpu_context:
        lib().mpmc_destroy_cpu_context(_cpu_context)
        _cpu_context = 0


atexit.register(close_cpu_context)
