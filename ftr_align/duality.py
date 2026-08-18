"""Dual and primal faces of a support solve, and the attribution blocks built on
them.

The optimal dual face ``Lambda*(b;y)`` need not be a singleton, so per-row
multipliers are characterised by ranges over it.  Over the dual-optimal support
``J*`` the trade space ``D = ker C`` says which weight shifts change nothing;
its matroid components are the attribution blocks.
"""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import cvxpy as cp
import numpy as np
from scipy.linalg import null_space, orth, qr

from .network import NetworkModel, align
from .solve import (
    CENTER,
    Lambda_star,
    SupportProblem,
    SupportSolution,
    network_constraints,
)

FACE_TOL = 1e-6  # slack on the optimal-value constraint defining the face
SUPPORT_TOL = 1e-4  # mu > tol decides membership of J*; must exceed FACE_TOL leak
RANK_TOL = 1e-7  # numerical zero for rank / nullspace
SPAN_TOL = 1e-8  # relative residual for in_span membership


def robust_bounds(
    problem: SupportProblem,
    solver=None,
    hi_only: bool = False,
    tol: float = FACE_TOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row ``[mu_lo, mu_hi]`` over the optimal dual face.  Rows outside the
    support get ``(0, 0)``.

    ``mu`` is restricted to rows binding at the primal optimum, which is exact:
    complementary slackness pins every other row to 0 across the whole face.  Runs
    on HiGHS whatever ``solver`` names -- the face is a thin slab pinned to the base
    value, so both must share an engine.  ``hi_only`` skips the lower solves.
    """
    # Everything here runs on HiGHS, ignoring the caller's `solver` (we keep any
    # non-solver opts).  Two reasons: (1) the face LPs optimize over a razor-thin
    # slab (b^T mu == value +/- tol) on a low-dimensional candidate face -- simplex
    # handles that exactly, while an interior-point method can't find a strict
    # interior and reports infeasible; (2) the slab is pinned to `value`, so the
    # base solve must use the same engine or the two disagree past tol.  The
    # bounds themselves are solver-invariant.
    opts = solver if isinstance(solver, dict) else {}
    lp_opts = {**opts, "solver": "HIGHS"}
    data = problem.data
    active = data.active
    sol = problem.solve(solver=lp_opts, want_primal=True)
    value = sol.value

    # Candidates: rows binding at the primal optimum.  By complementary slackness
    # every other row has mu == 0 across the *entire* optimal face, so (a) only
    # these can have nonzero bounds and (b) restricting the face LP's mu to them
    # is exact.  Relative bind_tol since b is large at RTS scale.
    bind_tol = tol * np.maximum(1.0, np.abs(data.b))
    candidates = np.where(active & (data.b - data.K @ sol.q <= bind_tol))[0]

    lo = np.zeros(data.K.shape[0])
    hi = np.zeros(data.K.shape[0])
    if candidates.size == 0:
        return lo, hi

    # One compiled problem reused across rows and senses: a Parameter objective
    # selects which mu to extremize, so cvxpy canonicalizes the face once instead
    # of rebuilding an LP per row.  mu ranges only over the candidate rows.
    K_c, b_c = data.K[candidates], data.b[candidates]
    mu = cp.Variable(candidates.size, nonneg=True, name="mu")
    s = cp.Variable(name="s")
    face = Lambda_star(K_c, b_c, mu, s, data.direction, value, tol)
    select = cp.Parameter(candidates.size, name="select")
    face_prob = cp.Problem(cp.Maximize(select @ mu), face)

    e = np.zeros(candidates.size)
    for j, i in enumerate(candidates):
        e[j] = 1.0  # maximize mu_i -> hi
        select.value = e
        face_prob.solve(**lp_opts)
        hi[i] = mu.value[j]
        if not hi_only:
            e[j] = -1.0  # maximize -mu_i == minimize mu_i -> lo
            select.value = e
            face_prob.solve(**lp_opts)
            lo[i] = mu.value[j]
        e[j] = 0.0
    return lo, hi


class PrimalFaceRange(NamedTuple):
    """Range of a functional over the primal optimal face, with an optimizer at each end."""

    lo: float
    hi: float
    q_lo: np.ndarray
    q_hi: np.ndarray

    @property
    def width(self) -> float:
        """``hi - lo``.  Compare against :func:`face_leak`, never against a bare
        absolute tolerance -- the face is constructed with slack, so ``width``
        is never exactly zero."""
        return self.hi - self.lo


def face_leak(value: float, weights: np.ndarray, tol: float = FACE_TOL) -> float:
    """Width :func:`primal_face_range` reports for a face that is really a point.

    The optimal-value cut is relaxed by ``tol * max(1, |value|)``, and a functional
    of size ``||weights||`` reads that slack as range.  Compare widths against this,
    not against a bare tolerance.
    """
    return tol * max(1.0, abs(value)) * float(np.linalg.norm(weights))


def primal_face_range(
    problem: SupportProblem,
    weights: np.ndarray,
    solver=None,
    tol: float = FACE_TOL,
    base: SupportSolution | None = None,
) -> PrimalFaceRange:
    """Range of ``weights^T q`` over the primal optimal face, with an optimizer at
    each end.

    ``q_lo`` and ``q_hi`` are two genuinely distinct optima, which is what lets
    primal invariance be observed rather than assumed.  Runs on HiGHS; pass ``base``
    to reuse one base solve across several weight vectors.
    """
    opts = solver if isinstance(solver, dict) else {}
    lp_opts = {**opts, "solver": "HIGHS"}
    data = problem.data
    active = data.active
    if base is None:
        base = problem.solve(solver=lp_opts)
    if base.engine.upper() != lp_opts["solver"]:
        raise ValueError(
            f"primal_face_range runs its face LPs on {lp_opts['solver']}, but the "
            f"base solution came from {base.engine or 'an unnamed solver'}.  The "
            "optimal-face cut is a razor-thin slab pinned to the base value, so "
            "the two must come from the same engine."
        )
    value = base.value
    slack = tol * max(1.0, abs(value))

    weights = np.asarray(weights, dtype=float)
    q = cp.Variable(data.K.shape[1], name="q")
    face = network_constraints(data.K[active], data.b[active], q) + [
        data.direction @ q >= value - slack
    ]
    w = cp.Parameter(data.K.shape[1], name="w")
    face_prob = cp.Problem(cp.Maximize(w @ q), face)

    ends = []
    for sense in (1.0, -1.0):  # maximize, then minimize
        w.value = sense * weights
        face_prob.solve(**lp_opts)
        q_end = np.asarray(q.value, dtype=float)
        ends.append((float(weights @ q_end), q_end))
    (hi, q_hi), (lo, q_lo) = ends
    return PrimalFaceRange(lo=lo, hi=hi, q_lo=q_lo, q_hi=q_hi)



def in_span(rows: np.ndarray, target: np.ndarray, tol: float = SPAN_TOL) -> bool:
    """Whether ``target`` lies in the span of the rows of ``rows``.

    Projects onto an orthonormal basis of that span and thresholds the relative
    residual, so ``tol`` governs a literal distance to the subspace.
    """
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    target = np.asarray(target, dtype=float)
    norm = float(np.linalg.norm(target))
    if norm == 0.0:
        return True
    if rows.size == 0:
        return False
    basis = orth(rows.T)  # orthonormal basis of the span of the rows
    residual = target - basis @ (basis.T @ target)
    return bool(np.linalg.norm(residual) <= tol * norm)


def J_star(
    problem: SupportProblem,
    sol: SupportSolution | None = None,
    tol: float = SUPPORT_TOL,
) -> np.ndarray:
    """The dual-optimal support ``J*(b;y)``, from one interior-point solve.

    An interior-point method converges to the relative interior of the optimal dual
    face, whose support is exactly ``J*``; a simplex vertex gives a strict subset.
    The certificate is checked, not trusted.  Pass ``sol`` if already solved.
    """
    if sol is None:
        sol = problem.solve(solver=CENTER)
    if not sol.interior:
        raise ValueError(
            "J_star needs a certificate from the relative interior of the optimal "
            f"dual face, but this solution ({sol.engine or 'unnamed solver'}) does "
            "not declare interior=True.  A vertex gives a strict subset of J*, "
            "missing the degenerate rows."
        )
    return np.where(sol.mu > tol)[0]


# ----------------------------------------------------------------------------
# Trade space and attribution blocks
# ----------------------------------------------------------------------------
def trade_matrix(problem: SupportProblem, index: np.ndarray) -> np.ndarray:
    """``C(b;y)``: columns ``[k_i - mean(k_i); b_i]`` for ``i in index``.  Shape
    ``(n+1, |index|)``.
    """
    K, b = problem.data.K, problem.data.b
    cols = [np.concatenate([K[i] - K[i].mean(), [b[i]]]) for i in index]
    return np.array(cols).T if cols else np.zeros((K.shape[1] + 1, 0))


def trade_space(C: np.ndarray, tol: float = RANK_TOL) -> np.ndarray:
    """``D = ker C``: weight trades over the support that change neither the
    aggregate congestion price nor the support value.  Columns are a basis.
    """
    if C.shape[1] == 0:
        return np.zeros((0, 0))
    return null_space(C, rcond=tol)


def connected_blocks(C: np.ndarray, tol: float = RANK_TOL) -> list[list[int]]:
    """Partition the columns of ``C`` into matroid-connectivity blocks -- the finest
    partition along which ``ker C`` splits as a direct sum.

    From the fundamental circuits of one QR-pivoted basis; basis-independent.
    Returns column positions.
    """
    n_cols = C.shape[1]
    if n_cols == 0:
        return []

    _, R, piv = qr(C, pivoting=True, mode="economic")
    diag = np.abs(np.diag(R))
    rank = int((diag > tol * max(diag.max(), 1.0)).sum()) if diag.size else 0
    basis, nonbasis = list(piv[:rank]), list(piv[rank:])

    parent = list(range(n_cols))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    if basis:
        Cb = C[:, basis]
        for j in nonbasis:
            coef, *_ = np.linalg.lstsq(Cb, C[:, j], rcond=None)
            for k, ck in zip(basis, coef):
                if abs(ck) > 1e-6:
                    union(j, k)

    groups: dict[int, list[int]] = {}
    for j in range(n_cols):
        groups.setdefault(find(j), []).append(j)
    return [sorted(g) for g in groups.values()]


def attribution_blocks(
    problem: SupportProblem, index: np.ndarray | None = None
) -> list[np.ndarray]:
    """Attribution blocks as global row indices: the finest partition of ``J*(b;y)``
    along which the trade space splits, hence the finest units carrying an invariant
    attributed value.

    Pass ``index`` to reuse a support you already have.
    """
    if index is None:
        index = J_star(problem)  # CENTER: support via strict complementarity
    cols = connected_blocks(trade_matrix(problem, index))
    return [np.asarray([int(index[c]) for c in group]) for group in cols]


def block_totals(
    b: np.ndarray, mu: np.ndarray, blocks: list[np.ndarray]
) -> np.ndarray:
    """Per-block value ``W_{J_r} = sum_{i in J_r} b_i mu_i``.

    Constant across the optimal dual face, so any optimal ``mu`` gives the same
    answer -- which is what makes these reportable when the individual ``mu_i`` are
    not.
    """
    return np.array([float(b[rows] @ mu[rows]) for rows in blocks])
