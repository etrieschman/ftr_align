"""Summary tables for reporting -- the Table II / Table III builders.

Reporting sits on top of the math: ``solve`` gives support solutions, ``duality``
gives per-constraint duals, and these functions assemble them into the tables
used to present results across a set of runs.
"""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from .duality import net_dual
from .network import NetworkModel
from .solve import SupportSolution

EPS = 1e-9


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
    f = net_dual(f, sol_f.mu).rename({"mu": "mu_f"})
    g = net_dual(g, sol_g.mu).rename({"mu": "mu_g"})
    out = f.join(g, on=["contingency", "element"], how="full", coalesce=True)
    if labels:
        out = out.with_columns(**{k: pl.lit(v) for k, v in labels.items()})
    return out
