"""PyMC-compatible CPU leapfrog integration with Mojo vector kernels."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

try:
    from pymc.blocking import DictToArrayBijection, RaveledVars
except ImportError:
    DictToArrayBijection = None

    class RaveledVars(NamedTuple):
        data: np.ndarray
        point_map_info: object

from ._lib import addr, f64, lib


class State(NamedTuple):
    q: RaveledVars
    p: np.ndarray
    v: np.ndarray
    q_grad: np.ndarray
    energy: float
    model_logp: float
    index_in_trajectory: int


class IntegrationError(RuntimeError):
    pass


class CpuLeapfrogIntegrator:
    def __init__(self, potential, logp_dlogp_func):
        self._potential = potential
        pytensor_function = logp_dlogp_func._pytensor_function
        if logp_dlogp_func._raveled_inputs:

            def func(q, _):
                return pytensor_function(q)

        else:

            def func(q, point_map_info):
                if DictToArrayBijection is None:
                    raise RuntimeError("PyMC is required for unraveled model inputs")
                unraveled_q = DictToArrayBijection.rmap(
                    RaveledVars(q, point_map_info)
                ).values()
                return pytensor_function(*unraveled_q)

        self._logp_dlogp_func = func
        self._dtype = np.dtype(logp_dlogp_func.dtype)
        if np.dtype(self._potential.dtype) != self._dtype:
            raise ValueError(
                f"dtypes of potential ({self._potential.dtype}) and logp function "
                f"({self._dtype})don't match."
            )

    def compute_state(self, q, p):
        q_data = f64(q.data)
        logp, dlogp = self._logp_dlogp_func(q_data, q.point_map_info)
        p = f64(p)
        dlogp = f64(dlogp)
        if p.ndim != 1 or p.shape != q_data.shape:
            raise ValueError("p and q must be one-dimensional arrays with equal shape")
        if dlogp.ndim != 1 or dlogp.shape != q_data.shape:
            raise ValueError("gradient and q must be one-dimensional arrays with equal shape")
        v = self._potential.velocity(p, out=None)
        kinetic = self._potential.energy(p, velocity=v)
        return State(
            RaveledVars(q_data, q.point_map_info),
            p,
            v,
            dlogp,
            kinetic - logp,
            logp,
            0,
        )

    def step(self, epsilon, state):
        try:
            return self._step(epsilon, state)
        except np.linalg.LinAlgError as err:
            raise IntegrationError("LinAlgError during leapfrog step.") from err
        except ValueError as err:
            scipy_msg = "array must not contain infs or nans"
            if err.args and scipy_msg in str(err.args[0]).lower():
                raise IntegrationError(
                    "Infs or nans in scipy.linalg during leapfrog step."
                ) from err
            raise

    def _step(self, epsilon, state):
        q_new = f64(state.q.data, copy=True)
        p_new = f64(state.p, copy=True)
        v_new = np.empty_like(q_new)
        q_grad = f64(state.q_grad)
        pot = self._potential
        if hasattr(pot, "_leapfrog_first"):
            pot._leapfrog_first(q_new, p_new, q_grad, v_new, float(epsilon))
        else:
            lib().mpmc_add_scaled(addr(p_new), addr(q_grad), 0.5 * epsilon, q_new.size)
            pot.velocity(p_new, out=v_new)
            lib().mpmc_add_scaled(addr(q_new), addr(v_new), epsilon, q_new.size)

        logp, q_new_grad = self._logp_dlogp_func(q_new, state.q.point_map_info)
        q_new_grad = f64(q_new_grad)
        lib().mpmc_add_scaled(
            addr(p_new), addr(q_new_grad), 0.5 * epsilon, q_new.size
        )
        kinetic = pot.velocity_energy(p_new, v_new)
        return State(
            RaveledVars(q_new, state.q.point_map_info),
            p_new,
            v_new,
            q_new_grad,
            kinetic - logp,
            logp,
            state.index_in_trajectory + int(np.sign(epsilon)),
        )
