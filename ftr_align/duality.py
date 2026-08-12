"""Dual-face analysis of a support solve.

Given a direction ``d``, the optimal dual face ``Lambda*(b;y)`` is the set of
certificates attaining the support value.  It need not be a singleton, so
per-constraint multipliers are characterised by *robust ranges* ``[mu_lo,
mu_hi]`` over the face -- invariant to which dual optimum a solver returns --
which classify each row binding / degenerate / slack.

Over the dual-optimal support ``J*(b;y)`` we build the trade space
``D(b;y) = ker C(b;y)`` (weight shifts that change neither the aggregate
congestion price nor the support value) and partition ``J*`` into
matroid-connectivity attribution blocks with face-invariant totals ``W_{J_r}``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import cvxpy as cp
import numpy as np
import polars as pl
from scipy.linalg import null_space, qr

from .network import NetworkModel, align, contingency_label, element_label
from .solve import Lambda_star, SupportProblem, support_objective

FACE_TOL = 1e-6  # slack on the optimal-value constraint defining the face
CLASS_TOL = 1e-4  # zero threshold for classification; must exceed FACE_TOL leak
RANK_TOL = 1e-7  # numerical zero for rank / nullspace
NET_DUAL_TOL = 0.5  # drop sub-dollar net duals from the reported table

Classification = Literal["binding", "degenerate", "slack"]


def robust_bounds(
    problem: SupportProblem,
    solver=None,
    hi_only: bool = False,
    tol: float = FACE_TOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row ``[mu_lo, mu_hi]`` over the optimal dual face ``Lambda*(b;y)`` --
    the robust multiplier range, invariant to which dual optimum a solver
    returns.  Rows outside the support get ``(0, 0)``; classify with
    :func:`classify`.  ``hi_only`` skips the ``mu_lo`` solves (e.g. when the
    support alone is wanted -- though :func:`J_star` is the cheaper way there).

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


def classify(
    lo: np.ndarray, hi: np.ndarray, tol: float = CLASS_TOL
) -> list[Classification]:
    """Robust constraint classification (Prop. 4).  ``tol`` must exceed the
    face's numerical leakage (``FACE_TOL``)."""
    out: list[Classification] = []
    for low, high in zip(lo, hi):
        if low > tol:
            out.append("binding")
        elif high > tol:
            out.append("degenerate")
        else:
            out.append("slack")
    return out


def J_star_from_bounds(hi: np.ndarray, tol: float = CLASS_TOL) -> np.ndarray:
    """``J*(b;y)`` read off a :func:`robust_bounds` ``hi`` vector: rows carrying
    positive weight in some optimal certificate (``mu_hi > 0``) -- binding or
    degenerate.  Only these carry attribution."""
    return np.where(hi > tol)[0]


def J_star(problem: SupportProblem, tol: float = CLASS_TOL) -> np.ndarray:
    """``J*(b;y)``, the dual-optimal support, from a single interior-point solve
    -- ~100x cheaper than the :func:`robust_bounds` face-LP loop, which is needed
    only for the lo/hi ranges (:func:`classify`).

    By Goldman-Tucker strict complementarity an interior-point method converges
    to the analytic center of the optimal dual face, whose support is *exactly*
    ``J*``.  **CLARABEL is required and not overridable**: the result is exact
    only for an interior-point solver -- a simplex vertex (e.g. HiGHS) gives a
    strict subset of ``J*`` (it misses degenerate rows)."""
    mu = problem.solve(solver={"solver": "CLARABEL"}).mu
    return np.where(mu > tol)[0]


def net_dual(model: NetworkModel, mu: np.ndarray) -> pl.DataFrame:
    """Collapse stacked ``mu`` to a signed net dual per (contingency, element):
    ``mu_upper - mu_lower``.  Rows with ~zero net are dropped."""
    names = model.network.element_names
    records = []
    for c in model.contingencies:
        net = mu[model.rows_upper(c.key)] - mu[model.rows_lower(c.key)]
        for e in range(model.ell):
            if abs(net[e]) > NET_DUAL_TOL:
                records.append(
                    {
                        "contingency": contingency_label(c.key, names),
                        "element": element_label(names, e),
                        "mu": float(net[e]),
                    }
                )
    return pl.DataFrame(
        records, schema={"contingency": pl.Utf8, "element": pl.Utf8, "mu": pl.Float64}
    )


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
    problem: SupportProblem,
    mu: np.ndarray | None = None,
    index: np.ndarray | None = None,
    solver=None,
) -> pl.DataFrame:
    """Attribution blocks over the support with per-block totals
    ``W_{J_r} = sum_{i in J_r} b_i mu_i`` (invariant across the optimal face).

    The support ``index = J*(b;y)`` defaults to :func:`J_star` (one CLARABEL
    solve); pass a precomputed ``index`` to reuse it.  ``mu`` defaults to a support
    solve on ``solver`` (any optimal dual works -- ``W`` is face-invariant)."""
    if mu is None:
        mu = problem.solve(solver=solver).mu
    if index is None:
        index = J_star(problem)  # CLARABEL: support via strict complementarity
    blocks = connected_blocks(trade_matrix(problem, index))

    labels = problem.model.labels()
    b = problem.data.b
    records = []
    for r, cols in enumerate(blocks):
        rows = [int(index[c]) for c in cols]
        members = []
        for i in rows:
            row = labels.row(i, named=True)
            members.append(f"{row['contingency']}:{row['element']}:{row['side']}")
        W = float(sum(b[i] * mu[i] for i in rows))
        records.append(
            {"block": r, "members": members, "rows": rows, "size": len(rows), "W": W}
        )
    return pl.DataFrame(
        records,
        schema={
            "block": pl.Int64,
            "members": pl.List(pl.Utf8),
            "rows": pl.List(pl.Int64),
            "size": pl.Int64,
            "W": pl.Float64,
        },
    )


def discrepancy(f: NetworkModel, g: NetworkModel) -> dict[str, np.ndarray]:
    """Rows where the FTR model ``f`` and the DAM model ``g`` disagree, split by
    kind and by the failure mode each feeds (``prop:kinds``).  Aligned internally.

    A row disagrees in exactly one of two ways.  A *level* difference has both
    limits finite with ``f_i = alpha_i g_i``, ``alpha_i != 1``; a *coverage*
    difference has one model at ``+inf`` (unmonitored, hence looser than any
    finite limit).  Either way the looser model is the one that loses value on
    adopting the intersection, so ``f_i > g_i`` feeds ``U`` and ``f_i < g_i``
    feeds ``V``.

    Only level differences carry a nonzero floor (``cor:diagnosable``): an
    infinite limit forces ``mu_i = 0``, so a coverage difference's whole
    contribution is displaced value registered at other rows.
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


_REPAIR_SCHEMA = {
    "driver": pl.Utf8,
    "members": pl.List(pl.Utf8),
    "idxs": pl.List(pl.Int64),
    "repair_idxs": pl.List(pl.Int64),
    "repair": pl.Float64,
}


def marginal_repair(
    f: NetworkModel, g: NetworkModel, direction: np.ndarray, solver=None
) -> pl.DataFrame:
    """Standalone block-repair counterfactual (pitch sec. 3.2): for each gap
    block, the change in ``Delta = h(f;y) - h(g;y)`` from repairing **only** that
    block's differing rows (``f`` -> ``g``), measured from the original ``f``.

    Diagnostic, **not additive**: when both failure modes are present they
    interact (one block can mask another), so the marginal repairs do not sum to
    the gap (``prop:repair_nonadditive``).

    NOTE: this repairs toward ``g``, not toward the intersection ``f ^ g``, so it
    is *not* the memo's ``U^(S)``, whose monotonicity relies on the one-signed
    target.  Retargeting waits on ``meet``."""
    f_u, bf, bg, blocks = _repair_blocks(f, g, direction, solver)
    h_f = SupportProblem(f_u, direction).solve(solver=solver).value

    def reduction(rows: list[int]) -> float:
        repaired = replace(f_u, b=_with(bf, rows, bg))
        return h_f - SupportProblem(repaired, direction).solve(solver=solver).value

    records = [{**blk, "repair": reduction(blk["repair_idxs"])} for blk in blocks]
    return pl.DataFrame(records, schema=_REPAIR_SCHEMA)


def _repair_blocks(
    f: NetworkModel, g: NetworkModel, direction: np.ndarray, solver
) -> tuple[NetworkModel, np.ndarray, np.ndarray, list[dict]]:
    """Aligned ``f_u`` (plus its limits ``bf`` and the DAM limits ``bg``) and the
    gap's repair blocks across both failure modes.  Each block is an attribution
    block of the relevant support whose differing rows (``repair_idxs``) move the
    FTR model toward the DAM model.  Blocks with no differing rows are dropped.

    NOTE: ``U`` blocks are taken from the ``g`` support here, but
    ``prop:block_underfunding`` decomposes ``U`` over the blocks of ``f`` (and
    ``V`` over those of ``g``).  Reconciling that is step-3 work, not a rename."""
    f_u, g_u = align(f, g)
    bf, bg = f_u.b, g_u.b
    blocks = []
    for driver, model, differs in (
        ("underfunding", g_u, bf > bg),
        ("hedging", f_u, bf < bg),
    ):
        bl = attribution_blocks(SupportProblem(model, direction), solver=solver)
        for blk in bl.rows(named=True):
            repair_rows = [r for r in blk["rows"] if differs[r]]
            if repair_rows:
                blocks.append(
                    {
                        "driver": driver,
                        "members": blk["members"],
                        "idxs": blk["rows"],
                        "repair_idxs": repair_rows,
                    }
                )
    return f_u, bf, bg, blocks


def _with(b: np.ndarray, rows: list[int], source: np.ndarray) -> np.ndarray:
    """Copy of ``b`` with ``rows`` overwritten by ``source`` (the DAM limits)."""
    out = b.copy()
    out[rows] = source[rows]
    return out
