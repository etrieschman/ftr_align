"""The five-node network of the attribution memo (``fig_texas5``).

Nodes W (western renewables), N (northern renewables), S (southern thermal and
storage), H (eastern load), D (a central data centre / microgrid).  Every
peripheral node connects to D, with additional corridors WN, WS, NH, SH.

Structures the 3-node cannot show:

* the **WN / SH pair** -- priced together while carrying no circulation between
  them, so their attributed values stay individually identified;
* the **SDH triangle** -- a circulation, so only the joint attribution is
  identified.

No bid data: every proposition holds at an arbitrary certificate ``y >= 0``, and
positing a binding pattern *is* positing ``y``.  :func:`solve_limit_design`
turns a desired pattern into the limits that realize it.
"""

# %%
from __future__ import annotations

import numpy as np
import cvxpy as cp

from ..network import PhysicalNetwork

NODE_NAMES = np.array(["W", "N", "S", "D", "H"])
W, N, S, D, H = 0, 1, 2, 3, 4
ELEMENT_NAMES = np.array(["WN", "WS", "WD1", "WD2", "ND", "NH", "SD", "SH", "DH"])
WN, WS, WD1, WD2, ND, NH, SD, SH, DH = 0, 1, 2, 3, 4, 5, 6, 7, 8

# node-branch incidence (node x element), reactances, slack at D
A = np.array(
    [
        [1, 1, 1, 1, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, -1, -1, -1, 0, -1, 0, 1],
        [0, 0, 0, 0, 0, -1, 0, -1, -1],
    ],
    dtype=float,
)
X = np.array([2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0])
BASE_LIMITS = np.full(len(ELEMENT_NAMES), 100.0)
NETWORK = PhysicalNetwork(
    A=A, x=X, slack_idx=D, node_names=NODE_NAMES, element_names=ELEMENT_NAMES
)
n, ell = NETWORK.n_nodes, NETWORK.n_elements
H_base = NETWORK.ptdf()


# %%
# -------------------------------------
# Choose base limits so we can reproduce desired patterns
# -------------------------------------
def solve_limit_design(
    model, patterns, extra: callable | None = None, bounds=(10.0, 500.0), trades=()
):
    """Limits making each pattern bind exactly; ideally with no extra binding constraints.

    ``patterns`` maps a name to global row indices into the stacked system, so a
    pattern may name rows from any contingency; the sign is implicit in which
    half of the stack a row sits in.  :func:`pattern_rows` builds them from
    ``(element, sign)`` pairs on the base case.

    Limits are symmetric -- upper and lower rows of an element share a number.
    One shared ``b`` over all rows serves every pattern, each with its own ``q``.
    Pattern rows are pinned to their limit; every other row is pushed at least
    ``margin`` away, and ``margin`` is maximised.

    ``extra(b, q) -> list`` adds case-specific constraints (limit ties, injection
    signs).  ``trades`` is a sequence of ``(rows, z)`` forcing ``sum z_i b_i = 0``,
    which is the limit-side half of putting those rows in one attribution block --
    the PTDF-side half, ``sum z_i k_i = 0``, is b-free and must already hold.

    Returns ``(problem, b, q)`` as cvxpy variables.
    """
    K, n_rows = model.K, model.n_rows
    q = {name: cp.Variable(model.K.shape[1], name=f"q_{name}") for name in patterns}
    b = cp.Variable(n_rows, name="b")
    margin = cp.Variable(nonneg=True, name="margin")

    lo, hi = bounds
    half = n_rows // 2
    # Lines are symmetric: the upper and lower limit of a row are one number.
    const = [b >= lo, b <= hi, b[:half] == b[half:]]
    for rows, z in trades:
        const.append(
            cp.sum(cp.multiply(np.asarray(z, dtype=float), b[list(rows)])) == 0
        )

    for name, rows in patterns.items():
        rows = list(rows)
        if len(set(rows)) != len(rows):
            raise ValueError(f"{name} contains a repeated row")
        const.append(cp.sum(q[name]) == 0)
        flow = K @ q[name]
        pinned = set(rows)
        const += [flow[i] == b[i] for i in rows]
        const += [flow[i] <= b[i] - margin for i in range(n_rows) if i not in pinned]
        if extra is not None:
            const += list(extra(b, q[name]))

    problem = cp.Problem(cp.Maximize(margin), const)
    problem.solve(solver=cp.HIGHS)
    return problem, b, q


def base_pattern_rows(model, pattern) -> set[int]:
    """Global row indices a pattern asks to bind: the upper row for ``+1``, the
    lower for ``-1``."""
    ell = model.network.n_elements
    return {e if sign == +1 else ell + e for e, sign in pattern}


def base_pattern_direction(model, pattern) -> np.ndarray:
    """``d = K^T y`` for the certificate putting unit weight on the pattern's
    rows -- the inverse of :func:`solve_limit_design`, which finds limits making
    that pattern bind.

    Positing the pattern is positing ``y``, which is why this case carries no bid
    data."""
    y = np.zeros(model.n_rows)
    y[list(base_pattern_rows(model, pattern))] = 1.0
    return model.K.T @ y
