"""The support LP and DAM clearing.

``h(b; d) = max_{q in Q(b)} d^T q`` is solved in **dual form**

    min b^T mu   s.t.  K^T mu + 1 s = d,  mu >= 0

so the multipliers are variables and ``mu`` stays stacked-nonnegative: that is
what makes the dual-feasible set a cone, and it lets a row be degenerate on both
sides at once.

``solve(solver=...)`` takes ``None`` or a dict of CVXPY options, splatted
straight into ``Problem.solve``; anything else is a custom solver called as
``solver.solve(problem)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import cvxpy as cp
import numpy as np

from .network import NetworkModel

# Named solver options, so a call site reads as intent rather than configuration.
CENTER = {"solver": "CLARABEL"}  # analytic-centre certificate: J*, blocks
VERTEX = {"solver": "HIGHS"}  # simplex vertex: the face-range LPs

ZERO_TOL = 1e-7


# ----------------------------------------------------------------------------
# Composable assembly  (numpy- or cvxpy-valued args; reused by solve & duality)
# ----------------------------------------------------------------------------
def Lambda(K, mu, s, direction):
    """Dual feasibility: ``K^T mu + 1 s = d``, ``mu >= 0``."""
    return [K.T @ mu + s * np.ones(K.shape[1]) == direction, mu >= 0]


def Lambda_star(K, b, mu, s, direction, value, tol):
    """The optimal dual face: dual feasibility capped by ``b^T mu <= value + tol``.

    One-sided because weak duality already gives ``b^T mu >= value``.  Still a thin
    slab, so problems built on it need simplex and a ``value`` from the same
    engine.
    """
    return Lambda(K, mu, s, direction) + [b @ mu <= value + tol]


def network_constraints(K, b, q):
    """Primal network feasibility: ``K q <= b``, ``1^T q = 0``."""
    return [K @ q <= b, cp.sum(q) == 0]


# ----------------------------------------------------------------------------
# Problem data (typed, solver-free numpy bundle)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class SupportData:
    """The math of one support problem, no solver attached.  ``b`` and (later)
    ``mu`` are co-indexed full-length vectors over the model's rows."""

    K: np.ndarray  # (n_rows, n)
    b: np.ndarray  # (n_rows,) limits; +inf on unmonitored rows
    direction: np.ndarray  # (n,) node-space support direction d = K^T y

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
    engine: str = ""  # solver that produced this `mu`, for messages; "" if unknown
    # Whether `mu` lies in the *relative interior* of the optimal dual face, and
    # so whether `mu > 0` identifies J* (Goldman-Tucker).  Interior, not centre:
    # every relative-interior point has the same maximal support, so that is the
    # whole requirement -- the analytic centre is merely the particular interior
    # point an interior-point method lands on.  A *declaration by whoever solved*,
    # not something looked up from `engine`: a custom solver knows what its own
    # certificate is and says so here.  Defaults to False, so a solver that says
    # nothing cannot stand in for one that does.
    interior: bool = False


# ----------------------------------------------------------------------------
# Support problem
# ----------------------------------------------------------------------------
class SupportProblem:
    """Support problem for a model in node-space direction ``direction``."""

    def __init__(self, model: NetworkModel, direction: np.ndarray):
        self.model = model
        self.data = SupportData(
            K=model.K, b=model.b, direction=np.asarray(direction, dtype=float)
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

    mu = cp.Variable(data.K.shape[0], nonneg=True, name="mu")
    s = cp.Variable(name="s")
    constraints = Lambda(data.K, mu, s, data.direction)
    if (~active).any():
        constraints.append(mu[~active] == 0)
    objective = cp.Minimize(data.b[active] @ mu[active])
    lp = cp.Problem(objective, constraints)
    lp.solve(**opts)

    engine = lp.solver_stats.solver_name
    mu_value = np.asarray(mu.value, dtype=float)
    mu_value[~active] = 0.0
    return SupportSolution(
        value=float(objective.value),
        mu=mu_value,
        s=float(s.value),
        status="solved",
        q=_solve_primal(problem, opts) if want_primal else None,
        binding=mu_value > ZERO_TOL,
        # The engine cvxpy actually ran, not the one that was asked for -- `opts`
        # may name none at all, and a bare solve() defaults to CLARABEL.
        engine=engine,
        # This backend is the one place a cvxpy engine name is turned into the
        # property; a custom solver declares `interior` for itself.
        interior=engine.upper() == CENTER["solver"],
    )


def _solve_primal(problem: SupportProblem, opts: dict | None = None) -> np.ndarray:
    """Primal support: ``max d^T q  s.t.  K_active q <= b_active, 1^T q = 0``."""
    opts = opts or {}
    data = problem.data
    active = data.active
    q = cp.Variable(data.K.shape[1], name="q")
    objective = cp.Maximize(data.direction @ q)
    cp.Problem(objective, network_constraints(data.K[active], data.b[active], q)).solve(**opts)
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
    direction: np.ndarray  # (n,) = K^T y, the node-space congestion direction
    merch_surp: float
    status: str


def clear_dam(model: NetworkModel, inst: DamInstance, solver=None) -> DamResult:
    """Clear the DAM and return the certificate ``y*`` with the node-space direction
    ``d = K^T y*`` it induces.

    ``d`` is what downstream code carries: it lives in node space, shared by every
    model on the network, so support values need no alignment.
    """
    active = model.active
    q_gen = cp.Variable(inst.M_gen.shape[1], name="q_gen")
    q = cp.Variable(model.network.n_nodes, name="q")

    # The same primal network block the support problem uses.  Keeping it stacked
    # (rather than one cvxpy constraint per contingency) means the dual of the
    # single `K q <= b` row block *is* the certificate y, already in stacked row
    # order -- no per-contingency scatter, and no second spelling of feasibility.
    network = network_constraints(model.K[active], model.b[active], q)
    problem = cp.Problem(
        cp.Minimize(inst.p_gen @ q_gen),
        [
            q == inst.M_gen @ q_gen - inst.M_dem @ inst.q_dem,
            q_gen >= inst.min_gen,
            q_gen <= inst.max_gen,
            *network,
        ],
    )
    problem.solve(**(solver or {}))
    if q.value is None:
        raise ValueError(
            f"DAM clearing did not solve (status={problem.status!r}): no feasible "
            "dispatch for this model and instance -- e.g. contingencies that "
            "cannot be satisfied simultaneously."
        )

    q_value = np.asarray(q.value, dtype=float)
    y = np.zeros(model.n_rows)
    y[active] = np.asarray(network[0].dual_value, dtype=float)

    return DamResult(
        value=float(problem.value),
        q=q_value,
        y=y,
        direction=model.K.T @ y,
        # MS = y^T K q; the stacked form folds the usual (mu_up - mu_lo)^T H q,
        # since K = [H; -H] carries the sign.
        merch_surp=float(y @ (model.K @ q_value)),
        status=str(problem.status),
    )
