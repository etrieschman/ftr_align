"""Tables: the only layer that labels rows and emits DataFrames.

Everything below returns numpy arrays and plain floats; nothing below builds a
frame or knows a contingency's name.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import polars as pl

from .attribution import (
    _nested_pair,
    block_share_range,
    differences,
    floor,
    primal_invariant,
    row_shares,
)
from .duality import (
    J_star,
    attribution_blocks,
    block_totals,
    trade_matrix,
    trade_space,
)
from .network import NetworkModel, align, meet
from .solve import CENTER, VERTEX, SupportProblem

EPS = 1e-9


def row_labels(model: NetworkModel, rows: Iterable[int]) -> list[str]:
    """``contingency:element:side`` for each row index."""
    labels = model.labels()
    out = []
    for i in rows:
        row = labels.row(int(i), named=True)
        out.append(f"{row['contingency']}:{row['element']}:{row['side']}")
    return out


def _by_row(groups) -> dict[int, object]:
    """Invert ``(label, row indices)`` pairs into ``{row: label}``."""
    return {int(i): label for label, rows in groups for i in rows}


def summary(
    model: NetworkModel,
    direction: np.ndarray,
    target: NetworkModel | None = None,
    labels: dict | None = None,
    solver=None,
) -> dict:
    """One flat record for one model at one direction -- :func:`block_table`
    aggregated, same arguments.

    Always: ``h`` and the shape of the attribution it admits (``n_priced``,
    ``n_blocks``, ``max_block``, ``dim_trade_space``).  All singletons means
    attribution is effectively constraint-level; one large block means it is not
    identified there at all.

    With a ``target`` contained in ``model``: ``h_target`` and the failure mode
    ``loss = h - h_target``, plus its ``floor`` and ``floor_ratio``.

    A dict, so a sweep is ``pl.DataFrame([summary(...) for ...])``."""
    table = block_table(model, direction, target, solver=solver)
    sizes = table["size"]
    out = {
        **(labels or {}),
        "h": float(table["value"].sum()),
        "n_priced": int(sizes.sum()),
        "n_blocks": table.height,
        "max_block": int(sizes.max()) if table.height else 0,
        "dim_trade_space": int(table["dim_trade_space"].sum()),
    }
    if target is None:
        return out

    model_u, target_u = _nested_pair(model, target)
    mu = SupportProblem(model_u, direction).solve(solver=CENTER).mu
    loss = float(table["loss"].sum())
    value = floor(model_u, target_u, mu)
    # "Zero" for a failure mode is below the noise of the subtraction that made
    # it, not below EPS: it is a difference of support values of order 1e4.
    zero = 1e-6 * max(1.0, abs(out["h"]))
    return out | {
        "h_target": out["h"] - loss,
        "loss": loss,
        "floor": value,
        "floor_ratio": None if abs(loss) < zero else value / loss,
    }


def gap_summary(
    f: NetworkModel,
    g: NetworkModel,
    direction: np.ndarray,
    labels: dict | None = None,
    solver=None,
) -> dict:
    """One flat record per ``(model pair, direction)``: both failure modes and
    the gap, with each mode's floor and attribution shape.

        U = h(f) - h(f^g)      value f loses on adopting the intersection
        V = h(g) - h(f^g)      value g loses on adopting it
        Delta = h(f) - h(g) = U - V

    The pair-level composer, and the only thing that reports both modes at once:
    it is :func:`summary` called against ``f ^ g`` from each side, suffixed
    ``_U`` and ``_V``.  ``relative_gap`` is ``Delta / h(g)``."""
    m = meet(f, g)
    per = {
        mode: summary(model, direction, m, solver=solver)
        for mode, model in (("U", f), ("V", g))
    }
    h_f, h_g = per["U"]["h"], per["V"]["h"]
    h_meet = h_f - per["U"]["loss"]

    out = {
        **(labels or {}),
        "h_f": h_f,
        "h_g": h_g,
        "h_meet": h_meet,
        "U": per["U"]["loss"],
        "V": per["V"]["loss"],
        "Delta": h_f - h_g,
    }
    out["relative_gap"] = None if abs(h_g) < EPS else out["Delta"] / h_g
    for mode, row in per.items():
        out |= {
            f"floor_{mode}": row["floor"],
            f"floor_ratio_{mode}": row["floor_ratio"],
            **{
                f"{k}_{mode}": row[k]
                for k in ("n_priced", "n_blocks", "max_block", "dim_trade_space")
            },
        }
    return out


def constraint_table(
    model: NetworkModel,
    direction: np.ndarray,
    target: NetworkModel | None = None,
    labels: dict | None = None,
    solver=None,
) -> pl.DataFrame:
    """One row per priced constraint -- :func:`block_table` without the grouping.

    Same shape and the same two attributions, at row granularity instead of block
    granularity.  Restricted to rows in ``J*(b;y)`` plus, with a ``target``, the
    rows on which the two models disagree.

    Always present, from ``model`` alone:

        value = b_i mu_i        sums to h(model)

    Only with a ``target``, which must be contained in ``model``:

        loss  = mu_i [b_i - (K q)_i]    sums to h(model) - h(target)

    plus ``target_limit`` and ``difference`` -- whether the row disagrees on a
    *level* (both finite) or on *coverage* (one unmonitored), and which mode it
    feeds.

    Neither column is identified row by row where a block has more than one
    member; ``block`` says which rows those are, and :func:`block_table` is the
    honest unit.  ``mu`` is the raw stacked certificate, one row per side."""
    if target is not None:
        model, target = _nested_pair(model, target)

    problem = SupportProblem(model, direction)
    sol = problem.solve(solver=CENTER)
    blocks = attribution_blocks(problem, J_star(problem, sol))
    block_of = _by_row(enumerate(blocks))

    extra: dict[int, dict] = {}
    keep = set(block_of)
    if target is not None:
        q_target = SupportProblem(target, direction).solve(
            solver=solver, want_primal=True
        ).q
        share = row_shares(model, target, sol.mu, q_target)
        kind = _by_row(differences(model, target).items())
        keep |= set(kind) | set(np.flatnonzero(np.abs(share) > EPS).tolist())
        for i in keep:
            extra[i] = {
                "target_limit": float(target.b[i]),
                "difference": kind.get(i),
                "loss": float(share[i]),
            }

    base = model.labels()
    return pl.DataFrame(
        [
            {
                **(labels or {}),
                "block": block_of.get(i),
                "constraint": i,
                "contingency": base["contingency"][i],
                "element": base["element"][i],
                "side": base["side"][i],
                "limit": float(model.b[i]),
                "mu": float(sol.mu[i]),
                "value": float(model.b[i] * sol.mu[i]) if np.isfinite(model.b[i]) else 0.0,
                **extra.get(i, {}),
            }
            for i in sorted(map(int, keep))
        ]
    )


def block_table(
    model: NetworkModel,
    direction: np.ndarray,
    target: NetworkModel | None = None,
    labels: dict | None = None,
    solver=None,
) -> pl.DataFrame:
    """One row per attribution block, carrying two different quantities.

    Always present, from ``model`` alone:

        value = sum_{i in J_r} b_i mu_i        sums to h(model)

    ``value`` has no range column -- it is constant over the whole optimal dual
    face, even where the individual ``mu_i`` are not.

    Only with a ``target``, which must be contained in ``model``:

        loss  = sum_{i in B} mu_i [b_i - (K q)_i]   sums to h(model) - h(target)

    read at ``q``, a maximiser for ``target``.  ``loss_lo``/``loss_hi`` are the
    range as ``q`` moves over ``target``'s optimal face, and ``identified`` says
    whether that range is a point.

    The failure mode is which model you pass first: ``(f, d, f^g)`` gives U,
    ``(g, d, f^g)`` gives V.  ``labels`` adds constant columns.

    One solve without a target, three with, whatever the block count."""
    if target is not None:
        model, target = _nested_pair(model, target)

    problem = SupportProblem(model, direction)
    sol = problem.solve(solver=CENTER)
    blocks = attribution_blocks(problem, J_star(problem, sol))
    W = block_totals(problem.data.b, sol.mu, blocks)

    extra: list[dict] = [{} for _ in blocks]
    if target is not None:
        target_problem = SupportProblem(target, direction)
        q_target = target_problem.solve(solver=CENTER, want_primal=True).q
        share = row_shares(model, target, sol.mu, q_target)
        base = target_problem.solve(solver=VERTEX, want_primal=True)
        j_target = J_star(target_problem)
        for slot, rows in zip(extra, blocks):
            lo, hi = block_share_range(
                model, target, direction, sol.mu, rows, solver=solver, base=base
            )
            slot.update(
                loss=float(share[rows].sum()),
                loss_lo=lo,
                loss_hi=hi,
                identified=primal_invariant(
                    model, target, direction, sol.mu, rows, j_target=j_target
                ),
            )

    out = pl.DataFrame(
        [
            {
                **(labels or {}),
                "block": r,
                "members": row_labels(model, rows),
                "rows": [int(i) for i in rows],
                "size": len(rows),
                "value": float(W[r]),
                "dim_trade_space": int(
                    trade_space(trade_matrix(problem, rows)).shape[1]
                ),
                **slot,
            }
            for r, (rows, slot) in enumerate(zip(blocks, extra))
        ]
    )

    def _frac(column: str) -> pl.Expr:
        # "Zero" for a failure mode is below the noise of the subtraction that
        # produced it, not below EPS.  Typed null so the U and V frames stack.
        total = out[column].sum()
        if abs(total) < 1e-6 * max(1.0, abs(out["value"].sum())):
            return pl.lit(None, dtype=pl.Float64)
        return pl.col(column) / total

    out = out.with_columns(value_frac=_frac("value"))
    return out if target is None else out.with_columns(loss_frac=_frac("loss"))
