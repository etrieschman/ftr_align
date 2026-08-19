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
_DESIGNS: dict = {}


def _design_problem(model, names, extra, bounds):
    """The design LP for ``model``, with the pinned set left as a parameter.

    Which rows are pinned is the only thing that changes as patterns are swept,
    and it is data, not structure: one ``s`` per pattern, ``s_i = 1`` pinning row
    ``i`` to its limit and ``s_i = 0`` pushing it ``margin`` away.

        flow <= b - (1 - s) * margin        pinned: flow <= b;  else a margin below
        flow >= b - u,  u >= 0,  s * u <= 0 pinned: u = 0, so flow == b;  else free

    So the problem compiles once per ``(model, pattern names, extra, bounds)`` and
    every later sweep is a warm parameter update -- ~25x faster than rebuilding.
    Cached objects are returned, so callers must not mutate them.

    Returns ``(problem, b, q, s)``; ``q`` and ``s`` are dicts keyed by name.
    """
    key = (id(model), tuple(names), id(extra), bounds)
    if key in _DESIGNS:
        return _DESIGNS[key][2:]

    K, n_rows = model.K, model.n_rows
    lo, hi = bounds
    half = n_rows // 2

    b = cp.Variable(n_rows, name="b")
    margin = cp.Variable(nonneg=True, name="margin")
    # Lines are symmetric: the upper and lower limit of an element are one number.
    const = [b >= lo, b <= hi, b[:half] == b[half:]]

    q, s = {}, {}
    for name in names:
        q[name] = cp.Variable(K.shape[1], name=f"q_{name}")
        s[name] = cp.Parameter(n_rows, nonneg=True, name=f"pinned_{name}")
        slack = cp.Variable(n_rows, nonneg=True, name=f"u_{name}")
        flow = K @ q[name]
        const += [
            cp.sum(q[name]) == 0,
            flow <= b - cp.multiply(1 - s[name], margin),
            flow >= b - slack,
            cp.multiply(s[name], slack) <= 0,
        ]
        if extra is not None:
            const += list(extra(model, b, q[name]))

    problem = cp.Problem(cp.Maximize(margin), const)
    _DESIGNS[key] = (model, extra, problem, b, q, s)  # refs keep the ids unique
    return problem, b, q, s


def solve_limit_design(
    model, patterns, extra: callable | None = None, bounds=(10.0, 500.0)
):
    """Limits making each pattern bind exactly; ideally with no extra binding constraints.

    ``patterns`` maps a name to global row indices into the stacked system, so a
    pattern may name rows from any contingency; the sign is implicit in which
    half of the stack a row sits in.  :func:`base_pattern_rows` builds them from
    ``(element, sign)`` pairs on the base case.

    Limits are symmetric -- upper and lower rows of an element share a number.
    One shared ``b`` over all rows serves every pattern, each with its own ``q``.
    Pattern rows are pinned to their limit; every other row is pushed at least
    ``margin`` away, and ``margin`` is maximised.  ``problem.value > 0`` IS the
    check that the pattern is realizable.

    ``extra(model, b, q) -> list`` adds case-specific constraints (limit ties,
    injection signs).

    Sweeping patterns reuses one compiled problem -- see :func:`_design_problem` --
    so ``b`` and ``q`` are shared objects: copy ``b.value`` before the next call.

    Returns ``(problem, b, q)`` as cvxpy variables.
    """
    problem, b, q, s = _design_problem(model, tuple(patterns), extra, bounds)
    for name, rows in patterns.items():
        rows = list(rows)
        if len(set(rows)) != len(rows):
            raise ValueError(f"{name} contains a repeated row")
        pinned = np.zeros(model.n_rows)
        pinned[rows] = 1.0
        s[name].value = pinned
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
