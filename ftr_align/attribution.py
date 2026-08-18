"""Failure modes, repairs, and their bounds.

Every attributive function takes a nested pair ``(model, target)`` and measures
what ``model`` loses on adopting ``target``.  Arithmetic over solved objects
only -- the LPs live in ``solve`` and ``duality``, the tables in ``metrics``.
"""

from __future__ import annotations

import numpy as np

from .duality import J_star, attribution_blocks, in_span, primal_face_range
from .network import NetworkModel, align, meet, with_limits
from .solve import CENTER, SupportProblem, SupportSolution


NEST_TOL = 1e-9  # slack on the containment check


def _h(model: NetworkModel, direction: np.ndarray, solver=None) -> float:
    return SupportProblem(model, direction).solve(solver=solver).value


def _nested_pair(
    model: NetworkModel, target: NetworkModel
) -> tuple[NetworkModel, NetworkModel]:
    """Align ``(model, target)`` and require ``Q(target)`` inside ``Q(model)``.

    Containment is ``b_target <= b_model`` row by row after alignment; infinities
    compare correctly, so a target left unmonitored where the model is finite is
    caught.  Raises on a crossing pair, where the quantities are undefined rather
    than merely wrong.
    """
    model_u, target_u = align(model, target)
    crossing = np.where(target_u.b > model_u.b + NEST_TOL)[0]
    if crossing.size:
        raise ValueError(
            "this measures what `model` loses on adopting `target`, so Q(target) "
            f"must be contained in Q(model); it is not at rows {crossing.tolist()}."
            "  For a crossing pair, measure each model against the intersection: "
            "(f, meet(f, g)) gives U and (g, meet(f, g)) gives V."
        )
    return model_u, target_u


# ----------------------------------------------------------------------------
# Repairs (counterfactual, multiplicity-free)
# ----------------------------------------------------------------------------
def repaired(model: NetworkModel, target: NetworkModel, rows) -> NetworkModel:
    """``model`` with its limits on ``rows`` replaced by ``target``'s."""
    b = model.b.copy()
    b[np.asarray(rows, dtype=int)] = target.b[np.asarray(rows, dtype=int)]
    return with_limits(model, b)


def repair_value(
    model: NetworkModel,
    target: NetworkModel,
    direction: np.ndarray,
    rows,
    solver=None,
    base: float | None = None,
) -> float:
    """Failure-mode value that disappears when ``model``'s limits on ``rows`` are
    replaced by ``target``'s.

    Two solves; only the second depends on ``rows``, so pass ``base`` (the
    unrepaired ``h(model)``) when sweeping subsets.
    """
    model_u, target_u = _nested_pair(model, target)
    if base is None:
        base = _h(model_u, direction, solver)
    return base - _h(repaired(model_u, target_u, rows), direction, solver)


# ----------------------------------------------------------------------------
# Bounds on repair values
# ----------------------------------------------------------------------------
def floor(
    model: NetworkModel,
    target: NetworkModel,
    mu: np.ndarray,
    rows=None,
) -> float:
    """Lower bound on the repair value of ``rows``:

        sum_{i in rows} mu_i [b_i - target_i]

    Additive in ``rows``.  An unmonitored row forces ``mu_i = 0`` and so
    contributes nothing.  ``rows=None`` means every row.
    """
    model_u, target_u = _nested_pair(model, target)
    # Unenforced rows carry b = +inf and mu = 0; zero them before the arithmetic
    # rather than after, so `inf - inf` and `0 * inf` never produce a nan.
    active = np.isfinite(model_u.b)
    gap = np.where(active, model_u.b - np.where(active, target_u.b, 0.0), 0.0)
    if rows is None:
        return float(mu @ gap)
    rows = np.asarray(rows, dtype=int)
    return float(mu[rows] @ gap[rows])


def ceiling(
    model: NetworkModel,
    target: NetworkModel,
    mu: np.ndarray,
    direction: np.ndarray,
    q: np.ndarray,
) -> float:
    """Upper bound on a repair value: ``b^T mu - d^T q``, for any dual-feasible
    ``mu`` and any ``q`` feasible for the repaired model.

    ``q`` is an argument because the choice is the bound: ``q = 0`` returns
    ``h(model)``, a ``q`` from ``Q(target)`` bounds the full failure mode, and only
    a ``q`` exploiting slack at unrepaired rows bounds a subset's repair strictly.
    """
    model_u, _ = _nested_pair(model, target)
    active = np.isfinite(model_u.b)
    return float(model_u.b[active] @ mu[active] - direction @ q)


def row_shares(
    model: NetworkModel,
    target: NetworkModel,
    mu: np.ndarray,
    q_target: np.ndarray,
) -> np.ndarray:
    """The failure mode row by row, ``mu_i [b_i - (K q)_i]``, as a full-length
    vector co-indexed with the model's rows.

    ``q`` attains ``h(target)``, which makes the ceiling tight -- so ``.sum()`` is
    the failure mode and ``[rows].sum()`` is any subset's share.
    """
    model_u, _ = _nested_pair(model, target)
    active = np.isfinite(model_u.b)
    b = np.where(active, model_u.b, 0.0)  # avoid 0 * inf on unenforced rows
    return np.where(active, mu * (b - model_u.K @ q_target), 0.0)


# ----------------------------------------------------------------------------
# Block-level attribution
# ----------------------------------------------------------------------------
def _block_weights(
    model: NetworkModel, target: NetworkModel, mu: np.ndarray, rows
) -> tuple[np.ndarray, float]:
    """``w = sum_{i in B} mu_i k_i`` and ``const = sum_{i in B} mu_i b_i``, so that
    a block's share reads ``const - w^T q``.
    """
    model_u, _ = _nested_pair(model, target)
    rows = np.asarray(rows, dtype=int)
    return mu[rows] @ model_u.K[rows], float(mu[rows] @ model_u.b[rows])


def primal_invariant(
    model: NetworkModel,
    target: NetworkModel,
    direction: np.ndarray,
    mu: np.ndarray,
    rows,
    solver=None,
    j_target: np.ndarray | None = None,
) -> bool:
    """Whether a block's share is the same at every ``target`` optimum: whether
    ``sum_{i in B} mu_i k_i`` lies in ``span{1} + row(K_{J*(target)})``.

    One span test, no LPs.  Holds vacuously when that span is all of ``R^n``, i.e.
    when the target optimum is a vertex.  Pass ``j_target`` when looping blocks.
    """
    model_u, target_u = _nested_pair(model, target)
    w, _ = _block_weights(model_u, target_u, mu, rows)
    if j_target is None:
        j_target = J_star(SupportProblem(target_u, direction))
    return in_span(
        np.vstack([np.ones(target_u.K.shape[1]), target_u.K[j_target]]), w
    )


def block_share_range(
    model: NetworkModel,
    target: NetworkModel,
    direction: np.ndarray,
    mu: np.ndarray,
    rows,
    solver=None,
    base: SupportSolution | None = None,
) -> tuple[float, float]:
    """Interval a block's share spans as the ``target`` optimum ``q`` ranges over
    ``target``'s optimal face.

    The share is affine in ``q`` with coefficient ``-w``, so this is
    :func:`duality.primal_face_range` at ``w``, re-centred and with the ends
    swapped.  Collapses to a point exactly when :func:`primal_invariant` holds.
    Pass ``base`` (a HiGHS solve of ``target``) when looping blocks.
    """
    model_u, target_u = _nested_pair(model, target)
    w, const = _block_weights(model_u, target_u, mu, rows)
    rng = primal_face_range(
        SupportProblem(target_u, direction), w, solver=solver, base=base
    )
    return const - rng.hi, const - rng.lo


# ----------------------------------------------------------------------------
# Where the models disagree
# ----------------------------------------------------------------------------
def differences(f: NetworkModel, g: NetworkModel) -> dict[str, np.ndarray]:
    """Rows where ``f`` and ``g`` disagree, sorted into four kinds.

    *level* means both enforce the row at different finite limits; *coverage* means
    one leaves it unmonitored.  ``U``/``V`` names which model is looser there, and
    so which failure mode the row feeds.  Only a level difference can carry a
    floor.
    """
    f_u, g_u = align(f, g)
    bf, bg = f_u.b, g_u.b
    both_finite = np.isfinite(bf) & np.isfinite(bg)
    return {
        "level_U": np.where(both_finite & (bf > bg))[0],
        "level_V": np.where(both_finite & (bf < bg))[0],
        "coverage_U": np.where(~both_finite & (bf > bg))[0],
        "coverage_V": np.where(~both_finite & (bf < bg))[0],
    }
