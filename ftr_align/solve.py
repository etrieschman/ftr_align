"""The LP machinery: one *network solve* as the primitive.

the **dual** support problem is,

    minimize    b^T mu
    subject to  A^T mu + 1 s = A^T y      (this equality system *is* Lambda(y))
                mu >= 0

stated so the dual multipliers ``mu`` -- the thing attribution cares about --
are variables.  ``SupportProblem.solve`` returns a ``SupportSolution`` value
object so results compose over sets of ``y`` (multi-interval, ex-ante).

The assembly functions (:func:`dual_feasible`, :func:`support_objective`,
:func:`network_constraints`) accept either numpy arrays or cvxpy expressions, so
the same pieces build the ex-post LP here and the ex-ante bilinear program
later.

Solver seam: ``solve(solver=...)`` passes a CVXPY solver name straight through
(off-the-shelf and commercial backends).  Any other object is treated as a
custom solver and called as ``solver.solve(problem) -> SupportSolution``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import cvxpy as cp
import numpy as np

from .network import NetworkModel, outage_elements

ZERO_TOL = 1e-7


# ----------------------------------------------------------------------------
# Problem data (typed, solver-free numpy bundle)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class SupportData:
    """The math of one support problem, no solver attached.  ``b``, ``y`` (and
    later ``mu``) are co-indexed full-length vectors over the system rows."""

    A: np.ndarray  # (n_rows, n)
    b: np.ndarray  # (n_rows,) limits; +inf on inactive rows
    y: np.ndarray  # (n_rows,) stacked-nonneg certificate

    @property
    def active(self) -> np.ndarray:
        return np.isfinite(self.b)

    @property
    def direction(self) -> np.ndarray:
        """``A^T y`` in node space -- the RHS that defines ``Lambda(y)``."""
        return self.A.T @ self.y


class SupportSolution(NamedTuple):
    value: float
    mu: np.ndarray  # (n_rows,) dual certificate, 0 on inactive rows
    s: float  # balance multiplier
    status: str
    q: np.ndarray | None = None  # primal optimizer (node injections), if requested
    binding: np.ndarray | None = None  # (n_rows,) bool: mu > tol  (the support I(b;y))


# ----------------------------------------------------------------------------
# Composable assembly  (numpy- or cvxpy-valued args)
# ----------------------------------------------------------------------------
def dual_feasible(A, mu, s, direction):
    """Constraints defining ``Lambda(y)``: ``A^T mu + 1 s = direction``, ``mu >= 0``."""
    return [A.T @ mu + s * np.ones(A.shape[1]) == direction, mu >= 0]


def support_objective(b, mu):
    """Dual support objective ``b^T mu``."""
    return b @ mu


def network_constraints(A, b, q):
    """Primal network feasibility: ``A q <= b``, ``1^T q = 0``."""
    return [A @ q <= b, cp.sum(q) == 0]


# ----------------------------------------------------------------------------
# Support problem
# ----------------------------------------------------------------------------
class SupportProblem:
    """Dual-form support problem built from a model and a certificate ``y``."""

    def __init__(self, model: NetworkModel, y: np.ndarray):
        self.model = model
        self.data = SupportData(
            A=model.system.A, b=model.b, y=np.asarray(y, dtype=float)
        )

    @property
    def direction(self) -> np.ndarray:
        return self.data.direction

    def solve(self, solver=None, want_primal: bool = False) -> SupportSolution:
        # custom solver: duck-typed, must return a SupportSolution
        if solver is not None and not isinstance(solver, str):
            return solver.solve(self)

        data = self.data
        active = data.active
        n_rows = data.A.shape[0]

        mu = cp.Variable(n_rows, nonneg=True, name="mu")
        s = cp.Variable(name="s")
        constraints = [data.A.T @ mu + s * np.ones(data.A.shape[1]) == data.direction]
        if (~active).any():
            constraints.append(mu[~active] == 0)
        objective = cp.Minimize(support_objective(data.b[active], mu[active]))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=solver)

        mu_value = np.asarray(mu.value, dtype=float)
        mu_value[~active] = 0.0

        return SupportSolution(
            value=float(problem.value),
            mu=mu_value,
            s=float(s.value),
            status=str(problem.status),
            q=self._solve_primal(solver) if want_primal else None,
            binding=mu_value > ZERO_TOL,
        )

    def _solve_primal(self, solver=None) -> np.ndarray:
        """Primal support: ``max d^T q  s.t.  A_active q <= b_active, 1^T q = 0``."""
        data = self.data
        active = data.active
        q = cp.Variable(data.A.shape[1], name="q")
        constraints = network_constraints(data.A[active], data.b[active], q)
        problem = cp.Problem(cp.Maximize(data.direction @ q), constraints)
        problem.solve(solver=solver)
        return np.asarray(q.value, dtype=float)


# ----------------------------------------------------------------------------
# DAM clearing  (forward map: instance -> certificate y*)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class DamInstance:
    """Bids/offers and fixed loads defining a DAM clearing instance."""

    M_gen: np.ndarray  # (n, n_gen) node x generator
    M_dem: np.ndarray  # (n, n_dem) node x demand
    min_gen: np.ndarray  # (n_gen,)
    max_gen: np.ndarray  # (n_gen,)
    p_gen: np.ndarray  # (n_gen,) marginal cost / bid price
    q_dem: np.ndarray  # (n_dem,) fixed demand


class DamResult(NamedTuple):
    value: float
    q: np.ndarray  # (n,) nodal injections
    y: np.ndarray  # (n_rows,) stacked-nonneg certificate over model.system
    merch_surp: float
    status: str


def clear_dam(model: NetworkModel, inst: DamInstance, solver=None) -> DamResult:
    """Clear the DAM on ``model`` and return its shadow-price certificate ``y*``
    laid out over ``model.system`` rows (zeros where not enforced).

    Note: under dual degeneracy ``y*`` is non-unique; an interior-point solver
    (e.g. ``CLARABEL``) returns the analytic-center certificate.
    """
    system = model.system
    net = system.network

    q_gen = cp.Variable(inst.M_gen.shape[1], name="q_gen")
    q = cp.Variable(net.n_nodes, name="q")
    inj = inst.M_gen @ q_gen - inst.M_dem @ inst.q_dem
    constraints = [
        q == inj,
        cp.sum(q) == 0,
        q_gen >= inst.min_gen,
        q_gen <= inst.max_gen,
    ]

    upper, lower, K = {}, {}, {}
    for key in model.enforced:
        K[key] = net.ptdf(outage_elements(key))
        lim_u = model.b[system.rows_upper(key)]
        lim_l = model.b[system.rows_lower(key)]
        flow = K[key] @ q
        upper[key], lower[key] = flow <= lim_u, -flow <= lim_l
        constraints += [upper[key], lower[key]]

    problem = cp.Problem(cp.Minimize(inst.p_gen @ q_gen), constraints)
    problem.solve(solver=solver)

    q_value = np.asarray(q.value, dtype=float)
    y = np.zeros(system.n_rows)
    merch = 0.0
    for key in model.enforced:
        mu_u = np.asarray(upper[key].dual_value, dtype=float)
        mu_l = np.asarray(lower[key].dual_value, dtype=float)
        y[system.rows_upper(key)] = mu_u
        y[system.rows_lower(key)] = mu_l
        merch += (mu_u - mu_l) @ (K[key] @ q_value)

    return DamResult(
        value=float(problem.value),
        q=q_value,
        y=y,
        merch_surp=float(merch),
        status=str(problem.status),
    )
