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

from typing import Literal

import cvxpy as cp
import numpy as np
import polars as pl
from scipy.linalg import null_space, qr

from .network import NetworkModel, align, contingency_label
from .solve import SupportProblem, dual_feasible, support_objective

FACE_TOL = 1e-6  # slack on the optimal-value constraint defining the face
CLASS_TOL = 1e-4  # zero threshold for classification; must exceed FACE_TOL leak
RANK_TOL = 1e-7  # numerical zero for rank / nullspace

Classification = Literal["binding", "degenerate", "slack"]


def robust_bounds(
    problem: SupportProblem, solver=None, tol: float = FACE_TOL
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row ``[mu_lo, mu_hi]`` over the optimal dual face: ``Lambda(d)``
    intersected with ``b^T mu == h*``.  Inactive rows get ``(0, 0)``."""
    data = problem.data
    active = data.active
    value = problem.solve(solver=solver).value

    mu = cp.Variable(data.A.shape[0], nonneg=True, name="mu")
    s = cp.Variable(name="s")
    obj = support_objective(data.b[active], mu[active])
    face = dual_feasible(data.A, mu, s, data.direction) + [
        obj <= value + tol,
        obj >= value - tol,
    ]
    if (~active).any():
        face.append(mu[~active] == 0)

    lo = np.zeros(data.A.shape[0])
    hi = np.zeros(data.A.shape[0])
    for i in np.where(active)[0]:
        cp.Problem(cp.Minimize(mu[i]), face).solve(solver=solver)
        lo[i] = mu.value[i]
        cp.Problem(cp.Maximize(mu[i]), face).solve(solver=solver)
        hi[i] = mu.value[i]
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


def net_dual(model: NetworkModel, mu: np.ndarray) -> pl.DataFrame:
    """Collapse stacked ``mu`` to a signed net dual per (contingency, element):
    ``mu_upper - mu_lower``.  Rows with ~zero net are dropped."""
    names = model.network.element_names
    records = []
    for c in model.contingencies:
        net = mu[model.rows_upper(c.key)] - mu[model.rows_lower(c.key)]
        for e in range(model.ell):
            if abs(net[e]) > 0.5:
                records.append(
                    {
                        "contingency": contingency_label(c.key, names),
                        "element": str(names[e]) if names is not None else str(e),
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
    problem: SupportProblem, mu: np.ndarray | None = None, solver=None
) -> pl.DataFrame:
    """Attribution blocks over the support with per-block totals
    ``W_{G_r} = sum_{i in G_r} b_i mu_i`` (invariant across the optimal face).
    If ``mu`` is omitted a support solve supplies one (any face point works)."""
    if mu is None:
        mu = problem.solve(solver=solver).mu
    _, hi = robust_bounds(problem, solver=solver)
    index = support_index(hi)
    blocks = connected_blocks(trade_matrix(problem, index))

    labels = problem.model.labels()
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
    """Rows where FTR and DAM limits differ (pitch sec. 3.2), on a common index.

    The two models are aligned internally.  ``D_plus`` = rows where FTR is looser
    (``g > f``, underfunding-driving); ``D_minus`` = FTR tighter (``g < f``,
    hedging-inefficiency-driving).  ``+inf`` (unmonitored) is looser than any
    finite limit.
    """
    dam_u, ftr_u = align(dam, ftr)
    f, g = dam_u.b, ftr_u.b
    return {"D_plus": np.where(g > f)[0], "D_minus": np.where(g < f)[0]}
