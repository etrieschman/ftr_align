"""The LP machinery: one *network solve* as the primitive.

A support function is ``h_Q(d) = max_{q in Q} d^T q`` for a node-space direction
``d in R^n``.  Because ``d`` lives in node space -- shared by every model on the
same network -- DAM and FTR support values and the gap need **no row alignment**:
each solves on its own polytope with the same ``d``.

The canonical form is the **dual**,

    minimize    b^T mu
    subject to  A^T mu + 1 s = d        (this equality system *is* Lambda(d))
                mu >= 0

so the multipliers ``mu`` (what attribution cares about) are variables.

Solver seam: ``solve(solver=...)`` takes ``None | dict | custom solver``.
``None`` or a ``dict`` runs the built-in cvxpy path -- the dict is splatted
straight into ``cp.Problem.solve`` as its options (e.g.
``{"solver": "CLARABEL", "verbose": True}``).  Anything else is a custom solver,
called as ``solver.solve(problem) -> SupportSolution``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import cvxpy as cp
import numpy as np

from .network import NetworkModel

ZERO_TOL = 1e-7


# ----------------------------------------------------------------------------
# Composable assembly  (numpy- or cvxpy-valued args; reused by solve & duality)
# ----------------------------------------------------------------------------
def dual_feasible(A, mu, s, direction):
    """Constraints defining ``Lambda(d)``: ``A^T mu + 1 s = direction``, ``mu >= 0``."""
    return [A.T @ mu + s * np.ones(A.shape[1]) == direction, mu >= 0]


def support_objective(b, mu):
    """Dual support objective ``b^T mu``."""
    return b @ mu


def network_constraints(A, b, q):
    """Primal network feasibility: ``A q <= b``, ``1^T q = 0``."""
    return [A @ q <= b, cp.sum(q) == 0]


# ----------------------------------------------------------------------------
# Problem data (typed, solver-free numpy bundle)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class SupportData:
    """The math of one support problem, no solver attached.  ``b`` and (later)
    ``mu`` are co-indexed full-length vectors over the model's rows."""

    A: np.ndarray  # (n_rows, n)
    b: np.ndarray  # (n_rows,) limits; +inf on unmonitored rows
    direction: np.ndarray  # (n,) node-space support direction d

    @property
    def active(self) -> np.ndarray:
        return np.isfinite(self.b)


class SupportSolution(NamedTuple):
    value: float
    mu: np.ndarray  # (n_rows,) dual certificate, 0 on inactive rows
    s: float  # balance multiplier
    status: str
    q: np.ndarray | None = None  # primal optimizer (node injections), if requested
    binding: np.ndarray | None = None  # (n_rows,) bool: mu > tol  (the support I(b;d))


# ----------------------------------------------------------------------------
# Support problem
# ----------------------------------------------------------------------------
class SupportProblem:
    """Support problem for a model in node-space direction ``direction``."""

    def __init__(self, model: NetworkModel, direction: np.ndarray):
        self.model = model
        self.data = SupportData(
            A=model.A, b=model.b, direction=np.asarray(direction, dtype=float)
        )

    def solve(self, solver=None, want_primal: bool = False) -> SupportSolution:
        if solver is None or isinstance(solver, dict):
            return solve_support_cvxpy(self, solver, want_primal=want_primal)
        return solver.solve(self)  # custom solver: returns a SupportSolution


def solve_support_cvxpy(
    problem: SupportProblem, opts: dict | None = None, want_primal: bool = False
) -> SupportSolution:
    """Built-in dual support solve via cvxpy.  ``opts`` is splatted into
    ``cp.Problem.solve`` (e.g. ``{"solver": "CLARABEL", "verbose": True}``)."""
    opts = opts or {}
    data = problem.data
    active = data.active

    mu = cp.Variable(data.A.shape[0], nonneg=True, name="mu")
    s = cp.Variable(name="s")
    constraints = dual_feasible(data.A, mu, s, data.direction)
    if (~active).any():
        constraints.append(mu[~active] == 0)
    objective = cp.Minimize(support_objective(data.b[active], mu[active]))
    cp.Problem(objective, constraints).solve(**opts)

    mu_value = np.asarray(mu.value, dtype=float)
    mu_value[~active] = 0.0
    return SupportSolution(
        value=float(objective.value),
        mu=mu_value,
        s=float(s.value),
        status="solved",
        q=_solve_primal(problem, opts) if want_primal else None,
        binding=mu_value > ZERO_TOL,
    )


def _solve_primal(problem: SupportProblem, opts: dict | None = None) -> np.ndarray:
    """Primal support: ``max d^T q  s.t.  A_active q <= b_active, 1^T q = 0``."""
    opts = opts or {}
    data = problem.data
    active = data.active
    q = cp.Variable(data.A.shape[1], name="q")
    objective = cp.Maximize(data.direction @ q)
    cp.Problem(objective, network_constraints(data.A[active], data.b[active], q)).solve(**opts)
    return np.asarray(q.value, dtype=float)


# ----------------------------------------------------------------------------
# DAM clearing  (forward map: instance -> certificate y*  ->  direction d)
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
    y: np.ndarray  # (n_rows,) stacked-nonneg certificate over the DAM model's rows
    direction: np.ndarray  # (n,) = A^T y, the node-space congestion direction
    merch_surp: float
    status: str


def clear_dam(model: NetworkModel, inst: DamInstance, solver=None) -> DamResult:
    """Clear the DAM on ``model`` and return its shadow-price certificate ``y*``
    (over the model's rows) and the induced node-space ``direction = A^T y*``.

    ``solver`` is a dict of cvxpy options (or ``None``); under dual degeneracy
    ``y*`` is non-unique and an interior-point solver
    (``{"solver": "CLARABEL"}``) returns the analytic-center certificate.
    """
    net = model.network
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
    for c in model.contingencies:
        if not np.isfinite(c.upper).any() and not np.isfinite(c.lower).any():
            continue  # unmonitored contingency (e.g. inf limits after align)
        K[c.key] = net.ptdf(c.key)
        flow = K[c.key] @ q
        upper[c.key], lower[c.key] = flow <= c.upper, -flow <= c.lower
        constraints += [upper[c.key], lower[c.key]]

    problem = cp.Problem(cp.Minimize(inst.p_gen @ q_gen), constraints)
    problem.solve(**(solver or {}))
    if q.value is None:
        raise ValueError(
            f"DAM clearing did not solve (status={problem.status!r}): no feasible "
            "dispatch for this model and instance -- e.g. contingencies that "
            "cannot be satisfied simultaneously."
        )

    q_value = np.asarray(q.value, dtype=float)
    y = np.zeros(model.n_rows)
    merch = 0.0
    for key in upper:
        mu_u = np.asarray(upper[key].dual_value, dtype=float)
        mu_l = np.asarray(lower[key].dual_value, dtype=float)
        y[model.rows_upper(key)] = mu_u
        y[model.rows_lower(key)] = mu_l
        merch += (mu_u - mu_l) @ (K[key] @ q_value)

    return DamResult(
        value=float(problem.value),
        q=q_value,
        y=y,
        direction=model.A.T @ y,
        merch_surp=float(merch),
        status=str(problem.status),
    )
