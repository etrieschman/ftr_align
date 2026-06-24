"""Dual-face analysis of a support solve.

Given a certificate ``y``, the optimal dual face ``Lambda*(b;y)`` is the set of
dual certificates that attain the support value.  Because it need not be a
singleton (degeneracy at scale, or by construction in the redundant-element toy
variant), per-constraint multipliers are characterised by *robust ranges*
``[mu_lo, mu_hi]`` over the face, which classify each row as definitely binding,
degenerately binding, or definitely slack -- invariant to which dual optimum a
solver happens to return.

Slice 2 covers the robust ranges, classification, signed net duals (Table III),
and the DAM/FTR limit discrepancy.  Trade space ``D(b;y)`` and attribution
blocks come next.
"""

from __future__ import annotations

from typing import Literal

import cvxpy as cp
import numpy as np
import polars as pl
from scipy.linalg import null_space

from .network import NetworkModel, contingency_label
from .solve import SupportProblem, support_objective

FACE_TOL = 1e-6      # slack on the optimal-value constraint defining the face
CLASS_TOL = 1e-4     # zero threshold for classification; must exceed FACE_TOL leak
RANK_TOL = 1e-7      # numerical zero for rank/nullspace/RREF

Classification = Literal["binding", "degenerate", "slack"]


def robust_bounds(
    problem: SupportProblem,
    value: float | None = None,
    solver=None,
    tol: float = FACE_TOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row ``[mu_lo, mu_hi]`` over the optimal dual face ``Lambda*(b;y)``.

    The face is ``Lambda(y)`` intersected with ``b^T mu == h*``.  Inactive rows
    are pinned to 0; their bounds are ``(0, 0)``.
    """
    data = problem.data
    active = data.active
    n_rows = data.A.shape[0]
    if value is None:
        value = problem.solve(solver=solver).value

    mu = cp.Variable(n_rows, nonneg=True, name="mu")
    s = cp.Variable(name="s")
    obj = support_objective(data.b[active], mu[active])
    face = [
        data.A.T @ mu + s * np.ones(data.A.shape[1]) == data.direction,
        obj <= value + tol,
        obj >= value - tol,
    ]
    if (~active).any():
        face.append(mu[~active] == 0)

    lo = np.zeros(n_rows)
    hi = np.zeros(n_rows)
    for i in np.where(active)[0]:
        cp.Problem(cp.Minimize(mu[i]), face).solve(solver=solver)
        lo[i] = mu.value[i]
        cp.Problem(cp.Maximize(mu[i]), face).solve(solver=solver)
        hi[i] = mu.value[i]
    return lo, hi


def classify(
    lo: np.ndarray, hi: np.ndarray, tol: float = CLASS_TOL
) -> list[Classification]:
    """Robust constraint classification (Prop. 4).

    ``tol`` must exceed the face's numerical leakage (``FACE_TOL``); otherwise a
    solver-tolerance wiggle of an exactly-zero multiplier reads as degenerate.
    """
    out: list[Classification] = []
    for low, high in zip(lo, hi):
        if low > tol:
            out.append("binding")
        elif high > tol:
            out.append("degenerate")
        else:
            out.append("slack")
    return out


def net_dual(model: NetworkModel, mu: np.ndarray) -> pl.DataFrame:
    """Collapse the stacked ``mu`` to a signed net dual per (contingency,
    element): ``mu_upper - mu_lower``.  Rows with ~zero net are dropped."""
    system = model.system
    names = system.network.element_names
    records = []
    for key in system.contingencies:
        net = mu[system.rows_upper(key)] - mu[system.rows_lower(key)]
        for e in range(system.ell):
            if abs(net[e]) > 0.5:
                records.append(
                    {
                        "contingency": contingency_label(key, names),
                        "element": str(names[e]) if names is not None else str(e),
                        "mu": float(net[e]),
                    }
                )
    return pl.DataFrame(
        records, schema={"contingency": pl.Utf8, "element": pl.Utf8, "mu": pl.Float64}
    )


def support_index(hi: np.ndarray, tol: float = CLASS_TOL) -> np.ndarray:
    """``I(b;y)``: rows that carry positive weight in *some* optimal certificate
    (``mu_hi > 0``) -- binding or degenerate.  Only these can carry attribution."""
    return np.where(hi > tol)[0]


def trade_matrix(problem: SupportProblem, index: np.ndarray) -> np.ndarray:
    """``C`` with columns ``c_i = [a_bar_i; b_i]`` for ``i`` in ``index``, where
    ``a_bar_i = a_i - (1/n) 11^T a_i`` is the i-th row of ``A`` with its energy
    (mean) component removed.  Shape ``(n+1, |index|)``."""
    A = problem.data.A
    b = problem.data.b
    cols = []
    for i in index:
        a_bar = A[i] - A[i].mean()
        cols.append(np.concatenate([a_bar, [b[i]]]))
    return np.array(cols).T if len(cols) else np.zeros((A.shape[1] + 1, 0))


def trade_space(C: np.ndarray, tol: float = RANK_TOL) -> np.ndarray:
    """``D(b;y) = ker C``: weight trades over the support that leave both the
    aggregate congestion price (``sum d_i a_bar_i = 0``) and the support value
    (``sum d_i b_i = 0``) unchanged.  Columns are a basis; width = dim D."""
    if C.shape[1] == 0:
        return np.zeros((0, 0))
    return null_space(C, rcond=tol)


def _rref(M: np.ndarray, tol: float = RANK_TOL) -> tuple[np.ndarray, list[int]]:
    """Reduced row echelon form; returns (R, pivot_columns)."""
    M = M.astype(float).copy()
    n_rows, n_cols = M.shape
    pivots: list[int] = []
    r = 0
    for c in range(n_cols):
        if r >= n_rows:
            break
        p = r + int(np.argmax(np.abs(M[r:, c])))
        if abs(M[p, c]) <= tol:
            continue
        M[[r, p]] = M[[p, r]]
        M[r] /= M[r, c]
        for rr in range(n_rows):
            if rr != r:
                M[rr] -= M[rr, c] * M[r]
        pivots.append(c)
        r += 1
    return M, pivots


def connected_blocks(C: np.ndarray, tol: float = RANK_TOL) -> list[list[int]]:
    """Partition the columns of ``C`` into matroid-connectivity blocks: the
    finest partition along which ``D = ker C`` splits as a direct sum, so no
    trade crosses a block boundary.  Returned as lists of *column* positions.

    Computed from fundamental circuits of one basis (RREF); the resulting
    components are basis-independent.
    """
    n_cols = C.shape[1]
    if n_cols == 0:
        return []
    R, pivots = _rref(C, tol)
    pivot_row = {p: k for k, p in enumerate(pivots)}
    pivot_set = set(pivots)

    parent = list(range(n_cols))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    # each non-pivot column j unions with the pivots in its fundamental circuit
    for j in range(n_cols):
        if j in pivot_set:
            continue
        for p in pivots:
            if abs(R[pivot_row[p], j]) > tol:
                union(j, p)

    groups: dict[int, list[int]] = {}
    for j in range(n_cols):
        groups.setdefault(find(j), []).append(j)
    return [sorted(g) for g in groups.values()]


def attribution_blocks(
    problem: SupportProblem,
    value: float | None = None,
    mu: np.ndarray | None = None,
    solver=None,
) -> pl.DataFrame:
    """Attribution blocks over the support ``I(b;y)`` with per-block totals
    ``W_{G_r} = sum_{i in G_r} b_i mu_i`` (invariant across the optimal face).

    Returns one row per block: the member constraints (labelled) and ``W``.  If
    ``mu`` is not given, a support solve provides one (any face point works,
    since ``W`` is face-invariant)."""
    sol = problem.solve(solver=solver)
    if value is None:
        value = sol.value
    if mu is None:
        mu = sol.mu

    _, hi = robust_bounds(problem, value=value, solver=solver)
    index = support_index(hi)
    C = trade_matrix(problem, index)
    blocks = connected_blocks(C)

    labels = problem.model.system.labels()
    b = problem.data.b
    records = []
    for r, cols in enumerate(blocks):
        rows = [int(index[c]) for c in cols]
        members = [
            f"{labels.row(i, named=True)['contingency']}:"
            f"{labels.row(i, named=True)['element']}:"
            f"{labels.row(i, named=True)['side']}"
            for i in rows
        ]
        W = float(sum(b[i] * mu[i] for i in rows))
        records.append({"block": r, "members": members, "size": len(rows), "W": W})
    return pl.DataFrame(
        records,
        schema={"block": pl.Int64, "members": pl.List(pl.Utf8),
                "size": pl.Int64, "W": pl.Float64},
    )


def discrepancy(dam: NetworkModel, ftr: NetworkModel) -> dict[str, np.ndarray]:
    """Rows where the FTR and DAM limit vectors differ (pitch sec. 3.2).

    Requires the two models share a system.  ``D_plus`` = rows where FTR is
    looser (``g > f``, underfunding-driving); ``D_minus`` = FTR tighter
    (``g < f``, hedging-inefficiency-driving).  ``+inf`` (absent) counts as
    looser than any finite limit.
    """
    if dam.system is not ftr.system:
        raise ValueError("models must share a StackedSystem; use shared_system/embed")
    f, g = dam.b, ftr.b
    return {
        "D_plus": np.where(g > f)[0],
        "D_minus": np.where(g < f)[0],
    }
