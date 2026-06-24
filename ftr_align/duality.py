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

from .network import NetworkModel, contingency_label
from .solve import SupportProblem, support_objective

FACE_TOL = 1e-6      # slack on the optimal-value constraint defining the face
CLASS_TOL = 1e-4     # zero threshold for classification; must exceed FACE_TOL leak

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
