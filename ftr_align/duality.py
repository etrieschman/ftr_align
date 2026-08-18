"""Dual-face analysis of a support solve.

Given a direction ``d``, the optimal dual face ``Lambda*(b;y)`` is the set of
certificates attaining the support value.  It need not be a singleton, so
per-constraint multipliers are characterised by *robust ranges* ``[mu_lo,
mu_hi]`` over the face -- invariant to which dual optimum a solver returns.

Everything here returns numpy or plain values.  Anything that formats results for
a reader (labels, tables) lives in ``metrics``.

Over the dual-optimal support ``J*(b;y)`` we build the trade space
``D(b;y) = ker C(b;y)`` (weight shifts that change neither the aggregate
congestion price nor the support value) and partition ``J*`` into
matroid-connectivity attribution blocks with face-invariant totals ``W_{J_r}``.
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
    """Per-row ``[mu_lo, mu_hi]`` over the optimal dual face ``Lambda*(b;y)`` --
    the robust multiplier range, invariant to which dual optimum a solver
    returns.  Rows outside the support get ``(0, 0)``.  ``hi_only`` skips the
    ``mu_lo`` solves; note that the *support* alone is far cheaper from
    :func:`J_star` (one interior-point solve), so reach for this only when the
    ranges themselves are wanted.

    Two exact accelerations: (1) ``mu`` ranges only over rows binding at the
    primal optimum, since by complementary slackness every other row is 0 across
    the whole face; (2) one compiled problem with a Parameter objective is reused
    across rows/senses instead of rebuilding an LP each time.

    **Runs on HiGHS internally, ignoring the caller's ``solver`` name** (non-solver
    opts such as ``verbose`` still pass through).  The face is a razor-thin slab
    that simplex solves exactly while an interior-point method reports infeasible,
    and the slab is pinned to the base solve's value so both must share an engine.
    The bounds are solver-invariant, so this changes no results.
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
    """Range of a linear functional over the primal optimal face, with an
    optimizer at each end."""

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
    """The numerical width :func:`primal_face_range` reports for a face that is really
    a single point: the optimal-value cut is relaxed by ``tol * max(1, |value|)``,
    and a functional of size ``||weights||`` reads that slack as range.

    A caller deciding *invariant vs unidentified* must exceed this, the same way
    ``SUPPORT_TOL`` must exceed ``FACE_TOL`` for the dual-side bounds."""
    return tol * max(1.0, abs(value)) * float(np.linalg.norm(weights))


def primal_face_range(
    problem: SupportProblem,
    weights: np.ndarray,
    solver=None,
    tol: float = FACE_TOL,
    base: SupportSolution | None = None,
) -> PrimalFaceRange:
    """Range of ``weights^T q`` over the **primal** optimal face
    ``argmax_{q in Q(b)} d^T q`` -- the mirror of :func:`robust_bounds`, which
    ranges over the dual face.

    Two uses, both from ``prop:primal_invariance``.  The block share
    ``U_B = sum_{i in B} mu_i [f_i - (Kq)_i]`` is *affine* in the intersection
    optimum ``q^``, with coefficient vector ``w = sum_{i in B} mu_i k_i``, so
    passing that ``w`` gives exactly the range over which ``U_B`` is unidentified
    (``lo == hi`` iff the block share is invariant).  And ``q_lo``/``q_hi`` are
    two genuinely distinct optima, which is what lets the invariance *condition*
    be tested against observed behaviour rather than assumed.

    Like :func:`robust_bounds` this **runs on HiGHS internally**: the optimal-face
    cut is a razor-thin slab that simplex handles exactly while an interior-point
    method finds no strict interior, and the cut is pinned to a value from the
    base solve, so the two must share an engine.  The cut is one-sided
    (``d^T q >= value - tol``) because primal feasibility already gives
    ``d^T q <= value``.  ``tol`` is relative to the support value, which runs to
    ``1e4``-plus even on the toy.

    That relaxation leaks: a point face reports a width of order
    :func:`face_leak`, not zero.  Test ``width`` against that, never against a
    bare absolute tolerance.

    Pass ``base`` to reuse a base solve across several weight vectors on the same
    problem -- the usual case, since a per-block reporting loop asks for a range
    on the same face once per block.  It must be a vertex solution, for the same
    reason the function forces HiGHS on itself.
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

    Computed by projecting onto an orthonormal basis of that span (SVD) and
    measuring the relative residual ``||target - P target|| / ||target||``.
    Least squares would answer the same question, but its tolerance knob
    (``rcond``) controls rank truncation rather than the membership test we
    actually want, so the residual it reports is only implicitly the distance to
    the subspace.  Projecting makes the quantity being thresholded the literal
    distance, which is the number ``tol`` should govern.

    The shared engine for both invariance conditions:

    * ``prop:invariant_subset`` -- the total ``W_S`` is certified iff
      ``b_S in rowspace C(b;y)``, i.e. ``in_span(C, b_S)``, with ``b_S`` indexed
      by *position within* ``J*(b;y)``, not by global row.
    * ``prop:primal_invariance`` -- the block share ``U_B`` is independent of the
      intersection optimum iff ``sum_{i in B} mu_i k_i`` lies in
      ``span{1} + row(K_{J*(f^g;y)})``, i.e. ``in_span(vstack([ones, K_J]), w)``.

    (The first is equivalently ``b_S _|_ ker C``, testable against the
    :func:`trade_space` basis already computed for the blocks; both readings agree
    because ``(ker C)^perp = rowspace C``.)
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
    """``J*(b;y)``, the dual-optimal support, from a single interior-point solve
    -- ~100x cheaper than the :func:`robust_bounds` face-LP loop, which is needed
    only when the lo/hi *ranges* themselves are wanted.

    By Goldman-Tucker strict complementarity an interior-point method converges
    to the analytic center of the optimal dual face, whose support is *exactly*
    ``J*``.  **A relative-interior certificate is required** (``sol.interior``):
    every point of the face's relative interior has the same maximal support, so
    that -- not the analytic centre specifically -- is what this needs.  A simplex
    vertex gives a strict subset, missing the degenerate rows.

    Pass ``sol`` when the problem has already been solved -- which is the usual
    case, since whatever wanted ``J*`` generally wanted ``mu`` too.  It is checked
    rather than trusted: a vertex certificate raises instead of silently
    returning too small a support."""
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
    """``C(b;y)`` with columns ``c_i = [k_bar_i; b_i]`` for ``i in index``, where
    ``k_bar_i`` is row ``i`` of ``K`` with its energy (mean) component removed.
    Shape ``(n+1, |index|)``."""
    K, b = problem.data.K, problem.data.b
    cols = [np.concatenate([K[i] - K[i].mean(), [b[i]]]) for i in index]
    return np.array(cols).T if cols else np.zeros((K.shape[1] + 1, 0))


def trade_space(C: np.ndarray, tol: float = RANK_TOL) -> np.ndarray:
    """``D(b;y) = ker C(b;y)``: weight trades over the support that preserve both
    the aggregate congestion price and the support value.  Columns are a basis."""
    if C.shape[1] == 0:
        return np.zeros((0, 0))
    return null_space(C, rcond=tol)


def connected_blocks(C: np.ndarray, tol: float = RANK_TOL) -> list[list[int]]:
    """Partition the columns of ``C`` into matroid-connectivity blocks -- the
    finest partition along which ``D = ker C`` splits as a direct sum.  Computed
    from fundamental circuits of one (QR-pivoted) basis; basis-independent.
    Returns lists of column positions."""
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
    """Attribution blocks ``{J_r}`` (``prop:blocks``) as **global row indices**:
    the finest partition of ``J*(b;y)`` along which ``D = ker C`` splits as a
    direct sum, and hence the finest units carrying an invariant attributed value.

    Orchestration only -- ``J*`` (one CLARABEL solve), then ``C``, then the
    matroid components of :func:`connected_blocks`, whose column positions are
    mapped back to rows of ``K``.  Pass a precomputed ``index`` to reuse a support
    you already have -- ``J_star(problem, sol)`` off a solution in hand is the
    way to avoid a second solve here.  Reporting lives in
    ``metrics.block_table``."""
    if index is None:
        index = J_star(problem)  # CENTER: support via strict complementarity
    cols = connected_blocks(trade_matrix(problem, index))
    return [np.asarray([int(index[c]) for c in group]) for group in cols]


def block_totals(
    b: np.ndarray, mu: np.ndarray, blocks: list[np.ndarray]
) -> np.ndarray:
    """Per-block attributed value ``W_{J_r} = sum_{i in J_r} b_i mu_i``
    (``cor:block_value_invariance``).

    Constant across the optimal dual face, so any optimal ``mu`` gives the same
    answer -- that invariance is the whole point of grouping to blocks, and it is
    what makes these totals reportable when the individual ``mu_i`` are not."""
    return np.array([float(b[rows] @ mu[rows]) for rows in blocks])
