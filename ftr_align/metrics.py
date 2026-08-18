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

from .attribution import differences, floor, modes_from_values, row_shares
from .duality import J_star, attribution_blocks
from .network import NetworkModel, align, contingency_label, element_label, meet
from .solve import CENTER, SupportProblem, SupportSolution

EPS = 1e-9
NET_DUAL_TOL = 0.5  # drop sub-dollar net duals from the reported table


def net_dual(model: NetworkModel, mu: np.ndarray) -> pl.DataFrame:
    """Collapse stacked ``mu`` to a signed net dual per (contingency, element):
    ``mu_upper - mu_lower``.  Rows with ~zero net are dropped.

    The column is ``mu_signed`` rather than ``mu`` because the package carries two
    different things under that letter: this one, and the raw stacked multipliers
    that :func:`row_table` reports one row per side.  They are not interchangeable
    and the name should say which you are holding.

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
                        "mu_signed": float(net[e]),
                    }
                )
    return pl.DataFrame(
        records,
        schema={
            "contingency": pl.Utf8,
            "element": pl.Utf8,
            "mu_signed": pl.Float64,
        },
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


def _block_shape(blocks: list[np.ndarray], suffix: str = "") -> dict:
    """The four numbers describing what an attribution admits.

    Blocks *partition* the priced rows ``J*(b;y)``, so ``n_blocks`` counts groups
    rather than ambiguity: two rows that cannot trade give TWO singleton blocks,
    which is the fully-identified case.  Ambiguity is ``dim_trade_space`` (0 =
    every row separately attributable) or equivalently ``max_block`` (1 = the
    same thing).  Read ``n_blocks`` against ``n_priced``: equal means all
    singletons."""
    sizes = [len(rows) for rows in blocks]
    return {
        f"n_priced{suffix}": sum(sizes),
        f"n_blocks{suffix}": len(blocks),
        f"max_block{suffix}": max(sizes, default=0),
        f"dim_trade_space{suffix}": sum(sizes) - len(blocks),
    }


def _by_row(groups) -> dict[int, object]:
    """Invert an iterable of ``(label, row indices)`` into ``{row: label}``."""
    return {int(i): label for label, rows in groups for i in rows}


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
    attribution is not identified at all.

    One solve.  The blocks come off its ``mu`` rather than a second solve, so the
    certificate must be an analytic-centre one -- hence the :data:`CENTER`
    default."""
    sol = problem.solve(solver=CENTER if solver is None else solver)
    blocks = attribution_blocks(problem, J_star(problem, sol))
    return {**(labels or {}), "h": sol.value, **_block_shape(blocks)}


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
    complete now.

    **Three solves**, one per distinct polytope: the aligned ``f``, the aligned
    ``g``, and their intersection.  Everything else -- the failure modes, ``J*``,
    the blocks, the floors -- is read off those three certificates rather than
    re-derived, which is why the models are aligned once at the top: ``f`` and
    ``f_u`` are the same polytope (alignment only adds ``+inf`` rows), so solving
    both would be solving one LP twice.

    :data:`CENTER` by default and effectively required: the block columns need an
    analytic-centre certificate, and using one certificate for the floor and a
    different one for the partition would mix two points of the same optimal
    face."""
    opts = CENTER if solver is None else solver
    f_u, g_u = align(f, g)
    sols = {
        "U": SupportProblem(f_u, direction).solve(solver=opts),
        "V": SupportProblem(g_u, direction).solve(solver=opts),
    }
    h_meet = SupportProblem(meet(f, g), direction).solve(solver=opts).value
    modes = modes_from_values(sols["U"].value, sols["V"].value, h_meet)

    out = {**(labels or {}), **modes}
    # The gap as a fraction of DAM merchandising surplus -- Delta and h_g are both
    # already here, so this is a reading aid rather than a new quantity.
    out["relative_gap"] = (
        None if abs(modes["h_g"]) < EPS else modes["Delta"] / modes["h_g"]
    )
    # A failure mode is a difference of support values of order 1e4, so "zero"
    # means "below the noise of that subtraction", not below EPS.  Without this
    # a mode that is exactly zero reports a meaningless floor ratio.
    zero = 1e-6 * max(1.0, abs(modes["h_f"]), abs(modes["h_g"]))

    for mode, model in (("U", f_u), ("V", g_u)):
        problem = SupportProblem(model, direction)
        sol = sols[mode]
        blocks = attribution_blocks(problem, J_star(problem, sol))
        value = floor(f, g, sol.mu, mode=mode)
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
            **_block_shape(blocks, suffix=f"_{mode}"),
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
    index is mostly zeros and unenlightening.

    Two solves: the mode's model, and the intersection for ``q^``.  ``mu`` here is
    the **raw stacked** certificate, one row per side -- not the signed net that
    :func:`net_dual` reports."""
    f_u, g_u = align(f, g)
    m = meet(f, g)
    model = f_u if mode == "U" else g_u
    problem = SupportProblem(model, direction)
    # CENTER regardless of `solver`: this mu defines J* and so the blocks.
    # `solver` drives the primal solve for q^, which is engine-free.
    sol = problem.solve(solver=CENTER)
    q_meet = SupportProblem(m, direction).solve(solver=solver, want_primal=True).q
    mu = sol.mu
    share = row_shares(f, g, mu, q_meet, mode=mode)

    block_of = _by_row(enumerate(attribution_blocks(problem, J_star(problem, sol))))
    kind = _by_row(differences(f, g).items())

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
