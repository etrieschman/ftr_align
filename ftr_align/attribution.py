"""Misalignment attribution: splitting the alignment gap and pricing repairs.

The gap ``Delta(f,g;y) = h(f;y) - h(g;y)`` is the only term of the funding-gap
decomposition that can produce underfunding, and it hides structure: it is a
*difference* of two one-signed failure modes,

    U = h(f;y) - h(f^g;y)     value the FTR model loses on adopting the intersection
    V = h(g;y) - h(f^g;y)     value the DAM model loses on adopting it
    Delta = U - V

so a fully funded market can carry underfunding exposure and destroy hedge value
at the same time, and the funding rate reports neither.

Everything here is **arithmetic over solved objects** -- support values, an
optimal certificate, the blocks.  The things that genuinely solve or factor live
in ``solve`` (support LPs) and ``duality`` (dual face, primal face, blocks,
span tests).  Nothing here formats output; tables live in ``metrics``.

One asymmetry worth keeping in view.  Repairs are *counterfactual*: what a
different limit vector would have produced, fixed by ``(f, g, y, S)`` and immune
to primal or dual multiplicity (``rem:repair_fixed``).  The floor/ceiling bounds
and the block split are *attributive*: they divide a realized number, and are
only as well defined as the certificate and intersection optimum they are read
at -- hence the two invariance conditions.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .duality import J_star, attribution_blocks, in_span, primal_face_range
from .network import NetworkModel, align, meet
from .solve import SupportProblem


def _h(model: NetworkModel, direction: np.ndarray, solver=None) -> float:
    return SupportProblem(model, direction).solve(solver=solver).value


# ----------------------------------------------------------------------------
# Failure modes
# ----------------------------------------------------------------------------
def failure_modes(
    f: NetworkModel, g: NetworkModel, direction: np.ndarray, solver=None
) -> dict[str, float]:
    """``U``, ``V`` and ``Delta`` at one direction (``def:failure_modes``,
    ``prop:alignment_gap_decomposition``).

    Three support solves.  Both modes are nonnegative because ``Q(f^g)`` is
    contained in each of ``Q(f)`` and ``Q(g)``; ``Delta > 0`` forces ``U > 0`` and
    ``Delta < 0`` forces ``V > 0``, but both can be strictly positive at once --
    which is exactly what the realized gap conceals."""
    h_meet = _h(meet(f, g), direction, solver)
    h_f = _h(f, direction, solver)
    h_g = _h(g, direction, solver)
    return {
        "h_f": h_f,
        "h_g": h_g,
        "h_meet": h_meet,
        "U": h_f - h_meet,
        "V": h_g - h_meet,
        "Delta": h_f - h_g,
    }


# ----------------------------------------------------------------------------
# Repairs (counterfactual, multiplicity-free)
# ----------------------------------------------------------------------------
def repaired(model: NetworkModel, target: NetworkModel, rows) -> NetworkModel:
    """``model`` with its limits on ``rows`` replaced by ``target``'s
    (``def:repair``).  Both must already be on a common row index."""
    b = model.b.copy()
    rows = np.asarray(rows, dtype=int)
    b[rows] = target.b[rows]
    return replace(model, b=b)


def repair_value(
    f: NetworkModel,
    g: NetworkModel,
    direction: np.ndarray,
    rows,
    mode: str = "U",
    solver=None,
) -> float:
    """``U^(S)`` (or ``V^(S)``): the failure-mode value that disappears when the
    limits on ``S = rows`` are replaced by the **intersection** limits.

    The target is ``f ^ g``, not the other model.  That is what makes the repair
    one-signed and hence monotone in ``S`` (``prop:repair_basic``): repairing
    toward ``g`` would *loosen* ``f`` on rows where ``f`` is already the tighter
    of the two, and ``U^(S)`` could then fall as ``S`` grows.

    Not additive across disjoint sets, in either direction
    (``prop:repair_nonadditive``), which is why constraint-by-constraint repair
    reports do not order upgrades (``rem:linediff``)."""
    f_u, g_u = align(f, g)
    m = meet(f, g)
    model = f_u if mode == "U" else g_u
    return _h(model, direction, solver) - _h(
        repaired(model, m, rows), direction, solver
    )


# ----------------------------------------------------------------------------
# Bounds on repair values
# ----------------------------------------------------------------------------
def floor(
    f: NetworkModel,
    g: NetworkModel,
    mu: np.ndarray,
    rows=None,
    mode: str = "U",
) -> float:
    """``prop:floor`` -- a lower bound on ``U^(S)``, from one certificate and no
    further solves: ``sum_{i in S} mu_i [b_i - (f^g)_i]``.

    Additive in ``S`` and supported only where the models disagree *and* the
    pricing model prices the disagreement.  A coverage difference therefore
    contributes nothing (``cor:diagnosable``): an infinite limit forces
    ``mu_i = 0`` whenever the support value is finite, so its whole contribution
    is displaced value registered at other rows.

    ``rows=None`` means the full index set, i.e. a floor on the whole failure
    mode."""
    f_u, g_u = align(f, g)
    model = f_u if mode == "U" else g_u
    # Unenforced rows carry b = +inf and mu = 0; zero them before the arithmetic
    # rather than after, so `inf - inf` and `0 * inf` never produce a nan.
    active = np.isfinite(model.b)
    gap = np.where(active, model.b - np.where(active, meet(f, g).b, 0.0), 0.0)
    if rows is None:
        return float(mu @ gap)
    rows = np.asarray(rows, dtype=int)
    return float(mu[rows] @ gap[rows])


def ceiling(
    f: NetworkModel,
    g: NetworkModel,
    mu: np.ndarray,
    direction: np.ndarray,
    q: np.ndarray,
    mode: str = "U",
) -> float:
    """``prop:ceiling`` -- an upper bound on ``U^(S)``: ``f^T mu - d^T q`` for any
    dual-feasible ``mu`` and any ``q`` feasible for the repaired model.

    ``q`` is an argument, not a choice made internally, because the choice *is*
    the bound (``rem:injection_nesting``).  The multiplier ranges over
    ``Lambda(y)`` regardless of ``S``; all the ``S``-dependence sits in which
    injections are admissible.  ``q = 0`` is admissible for every ``S`` and
    returns ``h(f;y)`` -- true and useless.  A ``q`` drawn from ``Q(f^g)`` is also
    admissible for every ``S`` but bounds only the *full* failure mode.  Only a
    ``q`` exploiting slack at unrepaired rows bounds ``U^(S)`` strictly below
    ``U``.

    Note ``y^T K q = d^T q``: in node space the certificate enters only through
    the direction, as everywhere else."""
    f_u, g_u = align(f, g)
    model = f_u if mode == "U" else g_u
    active = np.isfinite(model.b)
    return float(model.b[active] @ mu[active] - direction @ q)


def exact_split(
    f: NetworkModel,
    g: NetworkModel,
    mu: np.ndarray,
    q_wedge: np.ndarray,
    rows=None,
    mode: str = "U",
) -> float:
    """``cor:exact_split`` -- the failure mode written out row by row,
    ``sum_i mu_i [b_i - (K q^)_i]``, where ``q^`` attains ``h(f^g;y)``.

    This is the ceiling closed: at ``S`` the full index set the admissible
    injections are exactly ``Q(f^g)``, so an optimal ``q^`` makes the bound
    tight.  Summing over all rows reproduces ``U`` (or ``V``) exactly, which is
    the sharpest available check that the certificate, the intersection optimum
    and the support values are mutually consistent."""
    share = row_shares(f, g, mu, q_wedge, mode=mode)
    if rows is None:
        return float(share.sum())
    return float(share[np.asarray(rows, dtype=int)].sum())


def row_shares(
    f: NetworkModel,
    g: NetworkModel,
    mu: np.ndarray,
    q_wedge: np.ndarray,
    mode: str = "U",
) -> np.ndarray:
    """The per-row terms of :func:`exact_split`, ``mu_i [b_i - (K q^)_i]``, as a
    full-length co-indexed vector (zero off the enforced rows)."""
    f_u, g_u = align(f, g)
    model = f_u if mode == "U" else g_u
    active = np.isfinite(model.b)
    b = np.where(active, model.b, 0.0)  # avoid 0 * inf on unenforced rows
    return np.where(active, mu * (b - model.K @ q_wedge), 0.0)


# ----------------------------------------------------------------------------
# Block-level attribution
# ----------------------------------------------------------------------------
def block_shares(
    f: NetworkModel,
    g: NetworkModel,
    direction: np.ndarray,
    mode: str = "U",
    solver=None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """``prop:block_underfunding`` -- the failure mode split over attribution
    blocks, returning ``(blocks, U_B)``.

    ``U`` decomposes over the blocks of the **FTR** support problem and ``V`` over
    those of the **DAM** one: each failure mode is attributed on the blocks of the
    model that loses the value, not of its counterpart.  Every ``U_B`` is
    invariant across the optimal dual face at fixed ``q^``
    (``prop:dual_invariance``); invariance across the choice of ``q^`` is the
    separate, conditional claim tested by :func:`primal_invariant`."""
    f_u, g_u = align(f, g)
    model = f_u if mode == "U" else g_u
    m = meet(f, g)

    problem = SupportProblem(model, direction)
    mu = problem.solve(solver=solver).mu
    q_wedge = SupportProblem(m, direction).solve(solver=solver, want_primal=True).q

    blocks = attribution_blocks(problem)
    share = row_shares(f, g, mu, q_wedge, mode=mode)
    return blocks, np.array([float(share[rows].sum()) for rows in blocks])


def _block_weights(
    f: NetworkModel, g: NetworkModel, mu: np.ndarray, rows, mode: str
) -> tuple[np.ndarray, float]:
    """``w = sum_{i in B} mu_i k_i`` and the constant ``sum_{i in B} mu_i b_i``,
    so that the block share reads ``U_B = const - w^T q^``."""
    f_u, g_u = align(f, g)
    model = f_u if mode == "U" else g_u
    rows = np.asarray(rows, dtype=int)
    return mu[rows] @ model.K[rows], float(mu[rows] @ model.b[rows])


def primal_invariant(
    f: NetworkModel,
    g: NetworkModel,
    direction: np.ndarray,
    mu: np.ndarray,
    rows,
    mode: str = "U",
    solver=None,
) -> bool:
    """``prop:primal_invariance`` -- whether a block's share is the same at every
    intersection optimum, i.e. whether ``sum_{i in B} mu_i k_i`` lies in
    ``span{1} + row(K_{J*(f^g;y)})``.

    Algebraic: one span test on top of one interior-point solve, no face LPs.
    The condition asks that the flows on the block move against one another so
    that the block total does not.  Blocks failing it must be reported at a
    stated ``q^`` and flagged with a range -- :func:`block_share_range`.

    Note it holds *vacuously* whenever ``span{1} + row(K_{J*(f^g;y)})`` is all of
    ``R^n``, i.e. whenever the intersection optimum is a vertex.  It has content
    only when that face has positive dimension."""
    m = meet(f, g)
    w, _ = _block_weights(f, g, mu, rows, mode)
    j_meet = J_star(SupportProblem(m, direction))
    return in_span(np.vstack([np.ones(m.K.shape[1]), m.K[j_meet]]), w)


def block_share_range(
    f: NetworkModel,
    g: NetworkModel,
    direction: np.ndarray,
    mu: np.ndarray,
    rows,
    mode: str = "U",
    solver=None,
) -> tuple[float, float]:
    """The interval a block's share spans as the intersection optimum ``q^``
    ranges over the optimal face (``rem:reporting``: two LPs per flagged block).

    ``U_B`` is affine in ``q^`` with coefficient ``-w``, so this is
    :func:`duality.primal_face_range` at ``w``, re-centred on the block share and
    with the ends swapped by the sign.  Collapses to a point exactly when
    :func:`primal_invariant` holds."""
    w, const = _block_weights(f, g, mu, rows, mode)
    rng = primal_face_range(SupportProblem(meet(f, g), direction), w, solver=solver)
    return const - rng.hi, const - rng.lo


# ----------------------------------------------------------------------------
# Where the models disagree
# ----------------------------------------------------------------------------
def discrepancy(f: NetworkModel, g: NetworkModel) -> dict[str, np.ndarray]:
    """Rows where the FTR model ``f`` and the DAM model ``g`` disagree, split by
    kind and by the failure mode each feeds (``prop:kinds``).  Aligned internally.

    A row disagrees in exactly one of two ways.  A *level* difference has both
    limits finite with ``f_i = alpha_i g_i``, ``alpha_i != 1``; a *coverage*
    difference has one model at ``+inf`` (unmonitored, hence looser than any
    finite limit).  Either way the looser model is the one that loses value on
    adopting the intersection, so ``f_i > g_i`` feeds ``U`` and ``f_i < g_i``
    feeds ``V``.

    The kind decides whether a row can carry a floor at all; the mode is only
    which model is looser.  Note that a row feeding ``U`` does *not* imply
    ``U > 0``: disagreement composes over rows, value does not
    (``prop:composition`` item 3).
    """
    f_u, g_u = align(f, g)
    bf, bg = f_u.b, g_u.b
    both_finite = np.isfinite(bf) & np.isfinite(bg)
    return {
        "level_U": np.where(both_finite & (bf > bg))[0],
        "level_V": np.where(both_finite & (bf < bg))[0],
        "coverage_U": np.where(~both_finite & (bf > bg))[0],
        "coverage_V": np.where(~both_finite & (bf < bg))[0],
    }
