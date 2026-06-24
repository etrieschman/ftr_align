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


def gap(sol_g: SupportSolution, sol_f: SupportSolution) -> float:
    """Alignment gap ``Delta = h(g) - h(f)`` (>0 underfunding, <0 hedging ineff.)."""
    return sol_g.value - sol_f.value


def ratio(sol_g: SupportSolution, sol_f: SupportSolution) -> float | None:
    """Alignment ratio ``eta = h(g) / h(f)``; ``None`` if ``h(f) == 0``."""
    return None if abs(sol_f.value) < EPS else sol_g.value / sol_f.value


def alignment_summary(
    runs: Iterable[tuple[dict, SupportSolution, SupportSolution]],
) -> pl.DataFrame:
    """Table II: one row per run with ``MS_DAM``, ``Delta``, ``eta``.

    ``runs`` yields ``(labels, sol_f, sol_g)`` where ``labels`` is metadata
    (e.g. ``{"model_difference": ..., "pattern": ...}``)."""
    rows = [
        {
            **labels,
            "MS_DAM": sol_f.value,
            "Delta": gap(sol_g, sol_f),
            "eta": ratio(sol_g, sol_f),
        }
        for labels, sol_f, sol_g in runs
    ]
    return pl.DataFrame(rows)


def dual_summary(
    dam: NetworkModel,
    sol_f: SupportSolution,
    ftr: NetworkModel,
    sol_g: SupportSolution,
    labels: dict | None = None,
) -> pl.DataFrame:
    """Table III: signed net duals ``mu_f`` and ``mu_g`` per (contingency,
    element), joined.  ``labels`` adds constant metadata columns."""
    f = net_dual(dam, sol_f.mu).rename({"mu": "mu_f"})
    g = net_dual(ftr, sol_g.mu).rename({"mu": "mu_g"})
    out = f.join(g, on=["contingency", "element"], how="full", coalesce=True)
    if labels:
        out = out.with_columns(**{k: pl.lit(v) for k, v in labels.items()})
    return out
