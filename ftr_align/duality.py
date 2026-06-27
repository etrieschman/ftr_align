"""Dual-face analysis of a support solve.

Given a direction ``d``, the optimal dual face ``Lambda*(b;d)`` is the set of
certificates attaining the support value.  It need not be a singleton, so
per-constraint multipliers are characterised by *robust ranges* ``[mu_lo,
mu_hi]`` over the face -- invariant to which dual optimum a solver returns --
which classify each row binding / degenerate / slack.

Over the support ``I(b;d)`` we build the trade space ``D = ker C`` (weight
shifts that change neither the aggregate congestion price nor the support value)
and partition ``I`` into matroid-connectivity attribution blocks with
face-invariant totals ``W_{G_r}``.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from math import factorial
from typing import Literal

import cvxpy as cp
import numpy as np
import polars as pl
from scipy.linalg import null_space, qr

from .network import NetworkModel, align, contingency_label, element_label
from .solve import SupportProblem, dual_feasible, support_objective

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
    """Per-row ``[mu_lo, mu_hi]`` over the optimal dual face ``Lambda(d)`` cap
    ``{b^T mu == h*}`` -- the robust multiplier range, invariant to which dual
    optimum a solver returns.  Rows outside the support get ``(0, 0)``; classify
    with :func:`classify`.  ``hi_only`` skips the ``mu_lo`` solves (e.g. when the
    support alone is wanted -- though :func:`support_set` is the cheaper way there).

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
    candidates = np.where(active & (data.b - data.A @ sol.q <= bind_tol))[0]

    lo = np.zeros(data.A.shape[0])
    hi = np.zeros(data.A.shape[0])
    if candidates.size == 0:
        return lo, hi

    # One compiled problem reused across rows and senses: a Parameter objective
    # selects which mu to extremize, so cvxpy canonicalizes the face once instead
    # of rebuilding an LP per row.  mu ranges only over the candidate rows.
    A_c, b_c = data.A[candidates], data.b[candidates]
    mu = cp.Variable(candidates.size, nonneg=True, name="mu")
    s = cp.Variable(name="s")
    face = dual_feasible(A_c, mu, s, data.direction) + [
        support_objective(b_c, mu) <= value + tol,
        support_objective(b_c, mu) >= value - tol,
    ]
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


def support_index(hi: np.ndarray, tol: float = CLASS_TOL) -> np.ndarray:
    """``I(b;d)``: rows carrying positive weight in some optimal certificate
    (``mu_hi > 0``) -- binding or degenerate.  Only these carry attribution."""
    return np.where(hi > tol)[0]


def support_set(problem: SupportProblem, tol: float = CLASS_TOL) -> np.ndarray:
    """``I(b;d)`` from a single interior-point solve -- ~100x cheaper than the
    :func:`robust_bounds` face-LP loop, which is needed only for the lo/hi ranges
    (:func:`classify`).

    By Goldman-Tucker strict complementarity an interior-point method converges
    to the analytic center of the optimal dual face, whose support is *exactly*
    ``I``.  **CLARABEL is required and not overridable**: the result is exact only
    for an interior-point solver -- a simplex vertex (e.g. HiGHS) gives a strict
    subset of ``I`` (it misses degenerate rows)."""
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
    """``C`` with columns ``c_i = [a_bar_i; b_i]`` for ``i in index``, where
    ``a_bar_i`` is row ``i`` of ``A`` with its energy (mean) component removed.
    Shape ``(n+1, |index|)``."""
    A, b = problem.data.A, problem.data.b
    cols = [np.concatenate([A[i] - A[i].mean(), [b[i]]]) for i in index]
    return np.array(cols).T if cols else np.zeros((A.shape[1] + 1, 0))


def trade_space(C: np.ndarray, tol: float = RANK_TOL) -> np.ndarray:
    """``D(b;d) = ker C``: weight trades over the support that preserve both the
    aggregate congestion price and the support value.  Columns are a basis."""
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
    ``W_{G_r} = sum_{i in G_r} b_i mu_i`` (invariant across the optimal face).

    The support ``index = I(b;d)`` defaults to :func:`support_set` (one CLARABEL
    solve); pass a precomputed ``index`` to reuse it.  ``mu`` defaults to a support
    solve on ``solver`` (any optimal dual works -- ``W`` is face-invariant)."""
    if mu is None:
        mu = problem.solve(solver=solver).mu
    if index is None:
        index = support_set(problem)  # CLARABEL: support via strict complementarity
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


def discrepancy(dam: NetworkModel, ftr: NetworkModel) -> dict[str, np.ndarray]:
    """Rows where FTR and DAM limits differ (pitch sec. 3.2), on a common index.

    The two models are aligned internally.  ``D_plus`` = rows where FTR is looser
    (``g > f``, underfunding-driving); ``D_minus`` = FTR tighter (``g < f``,
    hedging-inefficiency-driving).  ``+inf`` (unmonitored) is looser than any
    finite limit.
    """
    dam_u, ftr_u = align(dam, ftr)
    f, g = dam_u.b, ftr_u.b
    return {"D_plus": np.where(g > f)[0], "D_minus": np.where(g < f)[0]}


_REPAIR_SCHEMA = {
    "driver": pl.Utf8,
    "members": pl.List(pl.Utf8),
    "idxs": pl.List(pl.Int64),
    "repair_idxs": pl.List(pl.Int64),
    "repair": pl.Float64,
}


def marginal_repair(
    dam: NetworkModel, ftr: NetworkModel, direction: np.ndarray, solver=None
) -> pl.DataFrame:
    """Standalone block-repair counterfactual (pitch sec. 3.2): for each gap
    block, the change in ``h(g) - h(f)`` from repairing **only** that block's
    differing rows (FTR -> DAM), measured from the original FTR.

    Diagnostic, **not additive**: when both underfunding and hedging drivers are
    present they interact (one block can mask another), so the marginal repairs
    do not sum to the gap.  Use :func:`shapley_repair` for an additive split."""
    ftr_u, f, g, blocks = _repair_blocks(dam, ftr, direction, solver)
    h_g = SupportProblem(ftr_u, direction).solve(solver=solver).value

    def reduction(rows: list[int]) -> float:
        repaired = replace(ftr_u, b=_with(g, rows, f))
        return h_g - SupportProblem(repaired, direction).solve(solver=solver).value

    records = [{**blk, "repair": reduction(blk["repair_idxs"])} for blk in blocks]
    return pl.DataFrame(records, schema=_REPAIR_SCHEMA)


def shapley_repair(
    dam: NetworkModel, ftr: NetworkModel, direction: np.ndarray, solver=None
) -> pl.DataFrame:
    """Shapley attribution of the gap ``Delta = h(g) - h(f)`` across gap blocks:
    each block's repair averaged over all orders of repairing the others.

    Unlike :func:`marginal_repair` these are **additive** -- they sum to
    ``Delta`` -- and credit each block for the effect it has once masking blocks
    are already repaired.  Costs ``2**(#blocks)`` support solves (fine at toy /
    block-sparse scale; sample orderings for large block sets)."""
    ftr_u, f, g, blocks = _repair_blocks(dam, ftr, direction, solver)
    h_g = SupportProblem(ftr_u, direction).solve(solver=solver).value
    n = len(blocks)

    cache: dict[frozenset, float] = {}

    def reduction(subset: frozenset) -> float:
        if not subset:
            return 0.0
        if subset not in cache:
            rows = [r for i in subset for r in blocks[i]["repair_idxs"]]
            repaired = replace(ftr_u, b=_with(g, rows, f))
            cache[subset] = (
                h_g - SupportProblem(repaired, direction).solve(solver=solver).value
            )
        return cache[subset]

    phi = [0.0] * n
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for k in range(len(others) + 1):
            weight = factorial(k) * factorial(n - k - 1) / factorial(n)
            for subset in combinations(others, k):
                phi[i] += weight * (
                    reduction(frozenset(subset + (i,))) - reduction(frozenset(subset))
                )

    records = [{**blk, "repair": phi[i]} for i, blk in enumerate(blocks)]
    return pl.DataFrame(records, schema=_REPAIR_SCHEMA)


def _repair_blocks(
    dam: NetworkModel, ftr: NetworkModel, direction: np.ndarray, solver
) -> tuple[NetworkModel, np.ndarray, np.ndarray, list[dict]]:
    """Aligned ``ftr_u`` (plus its limits ``g`` and the DAM limits ``f``) and the
    gap's repair blocks across both drivers.  Each block is an attribution block
    of the relevant support whose differing rows (``repair_rows``) move the FTR
    model toward the DAM model: ``underfunding`` blocks come from the DAM support
    (rows where ``g > f``), ``hedging`` blocks from the FTR support (``g < f``).
    Blocks with no differing rows are dropped."""
    dam_u, ftr_u = align(dam, ftr)
    f, g = dam_u.b, ftr_u.b
    blocks = []
    for driver, model, differs in (
        ("underfunding", dam_u, g > f),
        ("hedging", ftr_u, g < f),
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
    return ftr_u, f, g, blocks


def _with(b: np.ndarray, rows: list[int], source: np.ndarray) -> np.ndarray:
    """Copy of ``b`` with ``rows`` overwritten by ``source`` (the DAM limits)."""
    out = b.copy()
    out[rows] = source[rows]
    return out
