"""NUTS tree construction compatible with PyMC's internal tree contract."""

from __future__ import annotations

from collections import namedtuple

import numpy as np

try:
    from pymc.step_methods.hmc.base_hmc import DivergenceInfo
except ImportError:
    DivergenceInfo = namedtuple("DivergenceInfo", "message exec_info state state_div")

from ._lib import addr, f64, lib
from .integration import IntegrationError

Proposal = namedtuple("Proposal", "q, q_grad, energy, logp, index_in_trajectory")
Subtree = namedtuple("Subtree", "left, right, p_sum, proposal, log_size")


def is_turning(momentum_sum, left_velocity, right_velocity):
    momentum_sum = f64(momentum_sum)
    left_velocity = f64(left_velocity)
    right_velocity = f64(right_velocity)
    if not (
        momentum_sum.shape == left_velocity.shape == right_velocity.shape
        and momentum_sum.ndim == 1
    ):
        raise ValueError("turning vectors must be one-dimensional with equal shape")
    return bool(
        lib().mpmc_is_turning(
            addr(momentum_sum),
            addr(left_velocity),
            addr(right_velocity),
            momentum_sum.size,
        )
    )


class _Tree:
    def __init__(self, ndim, integrator, start, step_size, Emax, rng):
        self.ndim = ndim
        self.integrator = integrator
        self.start = start
        self.step_size = step_size
        self.Emax = Emax
        self.start_energy = start.energy
        self.rng = rng
        self.left = self.right = start
        self.proposal = Proposal(
            start.q, start.q_grad, start.energy, start.model_logp, 0
        )
        self.depth = 0
        self.log_size = 0.0
        self.log_accept_sum = -np.inf
        self.mean_tree_accept = 0.0
        self.n_proposals = 0
        self.p_sum = start.p.copy()
        self.max_energy_change = 0.0

    def extend(self, direction):
        if direction > 0:
            tree, diverging, turning = self._build_subtree(
                self.right, self.depth, float(self.step_size)
            )
            leftmost_begin, leftmost_end = self.left, self.right
            rightmost_begin, rightmost_end = tree.left, tree.right
            leftmost_p_sum = self.p_sum.copy()
            rightmost_p_sum = tree.p_sum
            self.right = tree.right
        else:
            tree, diverging, turning = self._build_subtree(
                self.left, self.depth, float(-self.step_size)
            )
            leftmost_begin, leftmost_end = tree.right, tree.left
            rightmost_begin, rightmost_end = self.left, self.right
            leftmost_p_sum = tree.p_sum
            rightmost_p_sum = self.p_sum.copy()
            self.left = tree.right
        self.depth += 1
        if diverging or turning:
            return diverging, turning
        self_log_size, tree_log_size = self.log_size, tree.log_size
        if np.log(self.rng.random()) < tree_log_size - self_log_size:
            self.proposal = tree.proposal
        self.log_size = np.logaddexp(tree_log_size, self_log_size)
        self.p_sum += tree.p_sum
        if self.depth > 0:
            turning = is_turning(self.p_sum, self.left.v, self.right.v)
            if not turning:
                p_sum1 = leftmost_p_sum + rightmost_begin.p
                turning = is_turning(
                    p_sum1, leftmost_begin.v, rightmost_begin.v
                )
            if not turning:
                p_sum2 = leftmost_end.p + rightmost_p_sum
                turning = is_turning(p_sum2, leftmost_end.v, rightmost_end.v)
        return diverging, turning

    def _single_step(self, left, epsilon):
        right = None
        error = None
        error_msg = None
        try:
            right = self.integrator.step(epsilon, left)
        except IntegrationError as err:
            error_msg = str(err)
            error = err
        else:
            energy_change = right.energy - self.start_energy
            if np.isnan(energy_change):
                energy_change = np.inf
            self.log_accept_sum = np.logaddexp(
                self.log_accept_sum, -energy_change if energy_change > 0 else 0
            )
            if abs(energy_change) > abs(self.max_energy_change):
                self.max_energy_change = energy_change
            if energy_change < self.Emax:
                proposal = Proposal(
                    right.q, right.q_grad, right.energy, right.model_logp,
                    right.index_in_trajectory,
                )
                return Subtree(right, right, right.p, proposal, -energy_change), None, False
            error_msg = f"Energy change in leapfrog step is too large: {energy_change}."
        finally:
            self.n_proposals += 1
        tree = Subtree(None, None, None, None, -np.inf)
        return tree, DivergenceInfo(error_msg, error, left, right), False

    def _build_subtree(self, left, depth, epsilon):
        if depth == 0:
            return self._single_step(left, epsilon)
        tree1, diverging, turning = self._build_subtree(left, depth - 1, epsilon)
        if diverging or turning:
            return tree1, diverging, turning
        tree2, diverging, turning = self._build_subtree(
            tree1.right, depth - 1, epsilon
        )
        left, right = tree1.left, tree2.right
        if not (diverging or turning):
            p_sum = tree1.p_sum + tree2.p_sum
            turning = is_turning(p_sum, left.v, right.v)
            if not turning and depth - 1 > 0:
                p_sum1 = tree1.p_sum + tree2.left.p
                turning = is_turning(
                    p_sum1, tree1.left.v, tree2.left.v
                )
                if not turning:
                    p_sum2 = tree1.right.p + tree2.p_sum
                    turning = is_turning(
                        p_sum2, tree1.right.v, tree2.right.v
                    )
            log_size = np.logaddexp(tree1.log_size, tree2.log_size)
            proposal = (
                tree2.proposal
                if np.log(self.rng.random()) < tree2.log_size - log_size
                else tree1.proposal
            )
        else:
            p_sum = tree1.p_sum
            log_size = tree1.log_size
            proposal = tree1.proposal
        return Subtree(left, right, p_sum, proposal, log_size), diverging, turning

    def stats(self):
        self.mean_tree_accept = np.exp(self.log_accept_sum) / self.n_proposals
        return {
            "depth": self.depth,
            "mean_tree_accept": self.mean_tree_accept,
            "energy_error": self.proposal.energy - self.start.energy,
            "energy": self.proposal.energy,
            "tree_size": self.n_proposals,
            "max_energy_error": self.max_energy_change,
            "model_logp": self.proposal.logp,
            "index_in_trajectory": self.proposal.index_in_trajectory,
        }
