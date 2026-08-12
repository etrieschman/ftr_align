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

from .network import NetworkModel, contingency_label, element_label
from .solve import SupportSolution

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
