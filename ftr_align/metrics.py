"""Tables: the only layer that labels rows and emits DataFrames.

Everything below returns numpy arrays and plain floats; nothing below builds a
frame or knows a contingency's name.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import polars as pl

from .attribution import (
    block_share_range,
    primal_invariant,
    row_shares,
)
from .duality import (
    J_star,
    attribution_blocks,
    block_totals,
    shift_matrix,
    shift_space,
)
from .network import NetworkModel, intersection
from .solve import CENTER, VERTEX, SupportProblem

EPS = 1e-9

_BLOCK_SCHEMA = {
    "block": pl.Int64,
    "members": pl.List(pl.Utf8),
    "rows": pl.List(pl.Int64),
    "size": pl.Int64,
    "value": pl.Float64,
    "dim_shift_space": pl.Int64,
}
_LOSS_SCHEMA = {
    "loss": pl.Float64,
    "loss_lo": pl.Float64,
    "loss_hi": pl.Float64,
    "identified": pl.Boolean,
}


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
    ``n_blocks``, ``max_block``, ``dim_shift_space``).  All singletons means
    attribution is effectively constraint-level; one large block means it is not
    identified there at all.

    With a ``target`` contained in ``model``: ``h_target`` and the failure mode
    ``loss = h - h_target``.

    A dict, so a sweep is ``pl.DataFrame([summary(...) for ...])``."""
    table = block_table(model, direction, target, solver=solver)
    sizes = table["size"]
    out = {
        **(labels or {}),
        "h": float(table["value"].sum()),
        "n_priced": int(sizes.sum()),
        "n_blocks": table.height,
        "max_block": int(sizes.max()) if table.height else 0,
        "dim_shift_space": int(table["dim_shift_space"].sum()),
    }
    if target is None:
        return out

    loss = float(table["loss"].sum())
    return out | {"h_target": out["h"] - loss, "loss": loss}


def gap_summary(
    ftr: NetworkModel,
    dam: NetworkModel,
    direction: np.ndarray,
    labels: dict | None = None,
    solver=None,
) -> dict:
    """One flat record per ``(model pair, direction)``: both failure modes and
    the gap, with each mode's attribution shape.

        U     = h_ftr - h_int      what the FTR model loses on adopting the intersection
        V     = h_dam - h_int      what the DAM model loses on adopting it
        Delta = h_ftr - h_dam = U - V

    The pair-level composer, and the only thing reporting both modes at once: it
    is :func:`summary` called against the intersection from each side, suffixed
    ``_U`` and ``_V``, with one shared intersection solve so ``Delta = U - V`` is
    exact.  ``relative_gap`` is ``Delta / h_dam``."""
    both = intersection(ftr, dam)
    per = {
        mode: summary(model, direction, both, solver=solver)
        for mode, model in (("U", ftr), ("V", dam))
    }
    h_ftr, h_dam = per["U"]["h"], per["V"]["h"]
    # One intersection solve for both modes.  Each `summary` computes its own,
    # and the two agree only to solver precision -- taking them separately would
    # leave `Delta = U - V` holding approximately rather than identically.
    h_int = SupportProblem(both, direction).solve(solver=CENTER).value

    out = {
        **(labels or {}),
        "h_ftr": h_ftr,
        "h_dam": h_dam,
        "h_int": h_int,
        "U": h_ftr - h_int,
        "V": h_dam - h_int,
        "Delta": h_ftr - h_dam,
    }
    out["relative_gap"] = None if abs(h_dam) < EPS else out["Delta"] / h_dam
    for mode, row in per.items():
        out |= {
            f"{k}_{mode}": row[k]
            for k in ("n_priced", "n_blocks", "max_block", "dim_shift_space")
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
    granularity.  Restricted to the rows in ``J*(b;y)``.

    Always present, from ``model`` alone:

        value = b_i mu_i        sums to h(model)

    Only with a ``target``, which must be contained in ``model``:

        loss  = mu_i [b_i - (K q)_i]    sums to h(model) - h(target)

    Only priced rows carry a share: an unpriced row has
    ``mu_i = 0`` at every optimal certificate.  Note priced is stronger than
    binding -- a row can be tight at the primal optimum and still carry
    ``mu_i = 0``.

    Neither column is identified row by row where a block has more than one
    member; ``block`` says which rows those are, and :func:`block_table` is the
    honest unit.  ``mu`` is the raw stacked certificate, one row per side."""
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
        for i in keep:
            extra[i] = {"loss": float(share[i])}

    base = model.labels()
    out = pl.DataFrame(
        [
            {
                **(labels or {}),
                "block": block_of[i],
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
    order = [
        c
        for c in (
            *(labels or {}),
            "block",
            "constraint",
            "contingency",
            "element",
            "side",
            "limit",
            "mu",
            "value",
            "loss",
        )
        if c in out.columns
    ]
    return out.select(order)


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

    The failure mode is which model you pass first: ``(ftr, v, intersection)``
    gives U, ``(dam, v, intersection)`` gives V.  ``labels`` adds constant columns.

    One solve without a target, three with, whatever the block count."""
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

    records = [
        {
            **(labels or {}),
            "block": r,
            "members": row_labels(model, rows),
            "rows": [int(i) for i in rows],
            "size": len(rows),
            "value": float(W[r]),
            "dim_shift_space": int(shift_space(shift_matrix(problem, rows)).shape[1]),
            **slot,
        }
        for r, (rows, slot) in enumerate(zip(blocks, extra))
    ]
    if records:
        out = pl.DataFrame(records)
    else:
        # Nothing priced -- an uncongested interval.  Keep the columns so a sweep
        # over intervals still stacks.
        schema = _BLOCK_SCHEMA | (_LOSS_SCHEMA if target is not None else {})
        out = pl.DataFrame(schema=schema).with_columns(
            **{k: pl.lit(v) for k, v in (labels or {}).items()}
        )
        out = out.select(*(labels or {}), *schema)

    def _frac(column: str) -> pl.Expr:
        # "Zero" for a failure mode is below the noise of the subtraction that
        # produced it, not below EPS.  Typed null so the U and V frames stack.
        total = out[column].sum()
        if abs(total) < 1e-6 * max(1.0, abs(out["value"].sum())):
            return pl.lit(None, dtype=pl.Float64)
        return pl.col(column) / total

    out = out.with_columns(value_frac=_frac("value"))
    return out if target is None else out.with_columns(loss_frac=_frac("loss"))
