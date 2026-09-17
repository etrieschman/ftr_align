"""Row- and block-level shares of a failure mode.

Every attributive function takes a nested pair ``(model, target)`` and measures
what ``model`` loses on adopting ``target``.  The two need not share rows: ``mu``
lives on the model's rows and the target enters only through its own solve.  Arithmetic over solved objects
only -- the LPs live in ``solve`` and ``duality``, the tables in ``metrics``.
"""

from __future__ import annotations

import numpy as np

from .duality import J_star, in_span, primal_face_range
from .network import NetworkModel
from .solve import SupportProblem, SupportSolution


FEAS_TOL = 1e-6  # relative slack on the feasibility check


def _check_feasible(model: NetworkModel, q: np.ndarray) -> None:
    """Require ``K q <= b`` on the model's rows.

    This is what prop:exact_split needs of the pair: every share
    ``mu_i [b_i - k_i^T q]`` is nonnegative exactly when the target optimum is
    feasible for the model.  It holds whenever ``Q(target)`` sits inside
    ``Q(model)`` -- e.g. ``target = intersection(model, other)`` -- and needs no
    common row index, so the two models may have different rows entirely.
    """
    active = model.active
    excess = model.K[active] @ q - model.b[active]
    bad = np.flatnonzero(excess > FEAS_TOL * np.maximum(1.0, np.abs(model.b[active])))
    if bad.size:
        raise ValueError(
            "this measures what `model` loses on adopting `target`, so the target's "
            f"optimum must be feasible for `model`; it violates {bad.size} rows.  For a "
            "crossing pair, measure each model against the intersection: "
            "(ftr, intersection(ftr, dam)) gives U and (dam, intersection(ftr, dam)) "
            "gives V."
        )


def row_shares(
    model: NetworkModel,
    target: NetworkModel,
    mu: np.ndarray,
    q_target: np.ndarray,
) -> np.ndarray:
    """The failure mode row by row, ``mu_i [b_i - (K q)_i]``, as a full-length
    vector co-indexed with the model's rows.

    ``q`` attains ``h(target)``, so ``.sum()`` is the failure mode
    (prop:exact_split) and ``[rows].sum()`` is any subset's share.
    """
    _check_feasible(model, q_target)
    active = model.active
    b = np.where(active, model.b, 0.0)  # avoid 0 * inf on unenforced rows
    return np.where(active, mu * (b - model.K @ q_target), 0.0)


# ----------------------------------------------------------------------------
# Block-level attribution
# ----------------------------------------------------------------------------
def _block_weights(
    model: NetworkModel, mu: np.ndarray, rows
) -> tuple[np.ndarray, float]:
    """``w = sum_{i in B} mu_i k_i`` and ``const = sum_{i in B} mu_i b_i``, so that
    a block's share reads ``const - w^T q``.
    """
    rows = np.asarray(rows, dtype=int)
    return mu[rows] @ model.K[rows], float(mu[rows] @ model.b[rows])


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
    w, _ = _block_weights(model, mu, rows)
    if j_target is None:
        j_target = J_star(SupportProblem(target, direction))
    return in_span(np.vstack([np.ones(target.n_nodes), target.K[j_target]]), w)


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
    w, const = _block_weights(model, mu, rows)
    rng = primal_face_range(
        SupportProblem(target, direction), w, solver=solver, base=base
    )
    return const - rng.hi, const - rng.lo
