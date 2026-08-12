"""Reporting: everything that turns computed results into a table for a reader.

The split this module defines, and the rest of the package respects:

* ``solve`` / ``duality`` / ``attribution`` compute.  They take and return numpy
  arrays and plain floats, carry no labels, and never build a DataFrame.
* ``metrics`` reports.  It is the only place that joins row indices to
  ``(contingency, element, side)`` names, applies display thresholds, and emits
  polars frames.

The reason for the line is that the computed objects are co-indexed vectors over
the rows of ``K`` -- that indexing is the invariant the whole package leans on,
and it survives only if labels stay out of it.  It also keeps the numerical layer
testable against the memos' propositions directly, without a table in between.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import polars as pl

from .attribution import differences, failure_modes, floor, row_shares
from .duality import attribution_blocks
from .network import NetworkModel, align, contingency_label, element_label, meet
from .solve import SupportProblem, SupportSolution

EPS = 1e-9
NET_DUAL_TOL = 0.5  # drop sub-dollar net duals from the reported table


def net_dual(model: NetworkModel, mu: np.ndarray) -> pl.DataFrame:
    """Collapse stacked ``mu`` to a signed net dual per (contingency, element):
    ``mu_upper - mu_lower``.  Rows with ~zero net are dropped.

    This is the *reporting* convention only.  The computation keeps ``mu`` in
    stacked-nonnegative form throughout, because that is what makes ``Lambda(y)``
    a cone, lets a row be degenerate on both sides at once, and keeps upper and
    lower as the distinct columns of ``C`` they are."""
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


def row_labels(model: NetworkModel, rows: Iterable[int]) -> list[str]:
    """``contingency:element:side`` for each row index, for block membership
    listings."""
    labels = model.labels()
    out = []
    for i in rows:
        row = labels.row(int(i), named=True)
        out.append(f"{row['contingency']}:{row['element']}:{row['side']}")
    return out


def alignment_summary(
    runs: Iterable[tuple[dict, SupportSolution, SupportSolution]],
) -> pl.DataFrame:
    """Table II: one row per run with ``MS_DAM``, ``Delta``, ``eta``.

    ``runs`` yields ``(labels, sol_f, sol_g)`` -- the support solutions for the
    FTR model ``f`` and the DAM model ``g``, in that order -- where ``labels`` is
    metadata (e.g. ``{"model_difference": ..., "pattern": ...}``).  Merchandising
    surplus is ``h(g;y)`` and the alignment gap is ``Delta = h(f;y) - h(g;y)``."""
    rows = [
        {
            **labels,
            "MS_DAM": sol_g.value,
            "Delta": sol_f.value - sol_g.value,
            "eta": None if abs(sol_g.value) < EPS else sol_f.value / sol_g.value,
        }
        for labels, sol_f, sol_g in runs
    ]
    return pl.DataFrame(rows)


def dual_summary(
    f: NetworkModel,
    sol_f: SupportSolution,
    g: NetworkModel,
    sol_g: SupportSolution,
    labels: dict | None = None,
) -> pl.DataFrame:
    """Table III: signed net duals ``mu_f`` (FTR model) and ``mu_g`` (DAM model)
    per (contingency, element), joined.  ``labels`` adds constant metadata
    columns."""
    left = net_dual(f, sol_f.mu).rename({"mu": "mu_f"})
    right = net_dual(g, sol_g.mu).rename({"mu": "mu_g"})
    out = left.join(right, on=["contingency", "element"], how="full", coalesce=True)
    if labels:
        out = out.with_columns(**{k: pl.lit(v) for k, v in labels.items()})
    return out


def support_summary(
    problem: SupportProblem, labels: dict | None = None, solver=None
) -> dict:
    """One flat record for a **single** support solve: its value and the shape of
    the attribution it admits.

    The one-model counterpart of :func:`run_row`, for when there is no ``(f, g)``
    pair -- a designed limit vector, or a pattern probed at a posited direction.
    ``n_blocks`` / ``max_block`` / ``dim_trade_space`` are exactly the N1
    quantities ("do blocks bind?"): all-singletons means attribution is
    effectively constraint-level, one giant block means constraint-level
    attribution is not identified at all."""
    sol = problem.solve(solver=solver)
    blocks = attribution_blocks(problem)
    sizes = [len(rows) for rows in blocks]
    return {
        **(labels or {}),
        "h": sol.value,
        "n_priced": sum(sizes),
        "n_blocks": len(blocks),
        "max_block": max(sizes, default=0),
        "dim_trade_space": sum(sizes) - len(blocks),
    }


def run_row(
    f: NetworkModel,
    g: NetworkModel,
    direction: np.ndarray,
    labels: dict | None = None,
    solver=None,
) -> dict:
    """One flat record summarising a single ``(model pair, direction)`` cell.

    Deliberately thin: a dict, so a sweep is ``pl.DataFrame([run_row(...) for
    ...])`` and adding a column is adding a key.  Every table T1-T5 and N1-N6
    wants is a groupby over a frame of these plus :func:`row_table`, so the
    intent is that this grows as the analysis asks for more, not that it is
    complete now."""
    modes = failure_modes(f, g, direction, solver=solver)
    models = dict(zip(("U", "V"), align(f, g)))
    out = {**(labels or {}), **modes}
    # A failure mode is a difference of support values of order 1e4, so "zero"
    # means "below the noise of that subtraction", not below EPS.  Without this
    # a mode that is exactly zero reports a meaningless floor ratio.
    zero = 1e-6 * max(1.0, abs(modes["h_f"]), abs(modes["h_g"]))

    for mode, model in models.items():
        problem = SupportProblem(model, direction)
        mu = problem.solve(solver=solver).mu
        blocks = attribution_blocks(problem)
        sizes = [len(rows) for rows in blocks]
        value = floor(f, g, mu, mode=mode)
        out |= {
            f"floor_{mode}": value,
            # The share of the failure mode the floor explains -- the number that
            # decides whether the floor is an instrument or a footnote (T1).
            # Reported per mode because only *level* differences carry a floor at
            # all, so a case can have a meaningful ratio in one mode and a
            # structural zero in the other.
            f"floor_ratio_{mode}": (
                None if abs(modes[mode]) < zero else value / modes[mode]
            ),
            f"n_blocks_{mode}": len(blocks),
            f"max_block_{mode}": max(sizes, default=0),
            f"dim_trade_space_{mode}": sum(sizes) - len(blocks),
        }
    return out


def row_table(
    f: NetworkModel,
    g: NetworkModel,
    direction: np.ndarray,
    labels: dict | None = None,
    mode: str = "U",
    solver=None,
) -> pl.DataFrame:
    """Per-constraint detail for one cell: limits, the certificate, the row's
    attributed share, and which block it landed in.

    Restricted to rows that either disagree or carry a share -- the full stacked
    index is mostly zeros and unenlightening."""
    f_u, g_u = align(f, g)
    m = meet(f, g)
    model = f_u if mode == "U" else g_u
    problem = SupportProblem(model, direction)
    mu = problem.solve(solver=solver).mu
    q_meet = SupportProblem(m, direction).solve(solver=solver, want_primal=True).q
    share = row_shares(f, g, mu, q_meet, mode=mode)

    block_of = {}
    for r, rows in enumerate(attribution_blocks(problem)):
        for i in rows:
            block_of[int(i)] = r

    kind = {}
    for key, idx in differences(f, g).items():
        for i in idx:
            kind[int(i)] = key

    keep = sorted(set(kind) | set(np.where(np.abs(share) > EPS)[0].tolist()))
    base = model.labels()
    records = [
        {
            **(labels or {}),
            "constraint": i,
            "contingency": base["contingency"][i],
            "element": base["element"][i],
            "side": base["side"][i],
            "f_i": float(f_u.b[i]),
            "g_i": float(g_u.b[i]),
            "meet_i": float(m.b[i]),
            "difference": kind.get(i),
            "mu": float(mu[i]),
            "share": float(share[i]),
            "block": block_of.get(i),
        }
        for i in map(int, keep)
    ]
    return pl.DataFrame(records)


def block_table(
    model: NetworkModel,
    blocks: list[np.ndarray],
    values: np.ndarray,
    value_name: str = "W",
    labels: dict | None = None,
) -> pl.DataFrame:
    """One row per attribution block: its members, size, and attributed value.

    ``values`` is whatever the block carries -- ``W_{J_r}`` from
    ``duality.block_totals`` when reporting a support value, or ``U_B`` from
    ``attribution.block_shares`` when reporting a failure mode."""
    records = [
        {
            "block": r,
            "members": row_labels(model, rows),
            "rows": [int(i) for i in rows],
            "size": len(rows),
            value_name: float(v),
        }
        for r, (rows, v) in enumerate(zip(blocks, values))
    ]
    out = pl.DataFrame(
        records,
        schema={
            "block": pl.Int64,
            "members": pl.List(pl.Utf8),
            "rows": pl.List(pl.Int64),
            "size": pl.Int64,
            value_name: pl.Float64,
        },
    )
    if labels:
        out = out.with_columns(**{k: pl.lit(v) for k, v in labels.items()})
    return out
