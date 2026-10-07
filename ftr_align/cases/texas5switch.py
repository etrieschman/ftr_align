"""The five-node network with a station switch: the ex-ante design case.

Node-breaker version of :mod:`texas5`.  Station ``H`` is split into two nodes
``H1`` (the ``NH`` side) and ``H2`` (the ``SH``/``DH`` side) joined by a switch.
The parallel ``WD1``/``WD2`` pair is merged into one line ``WD`` (reactance 0.5,
limit 200, electrically identical).

Every element is a row of one incidence matrix ``A = [A_L; A_S]``.  A switch
differs from a line only by its susceptance ``y_j``:

    open      y_j = 0        the element is absent
    relaxed   y_j in (0, inf) an ordinary line with reactance 1/y_j
    closed    y_j = inf      its two nodes share one angle; its flow is free

:func:`switch_ptdf` handles all three at once, so a state (binary or relaxed) is
just a node-space PTDF on the fixed node set, and every downstream object
(``NetworkModel``, ``intersection``, ``gap_summary``, ``faces``) is unchanged.

To add a second switch (say a split of ``D``), rename ``D`` to ``D1``/``D2`` in
``LINES`` and add one entry to ``SWITCHES``; nothing else changes.
"""

from __future__ import annotations

import numpy as np

from ..network import (
    Contingency,
    ContingencyRows,
    NetworkModel,
    PhysicalNetwork,
    compute_ptdf,
    is_connected,
)

# ---------------------------------------------------------------------------
# Static data: nodes, lines (name, from, to, reactance, limit), switches
# ---------------------------------------------------------------------------
NODE_NAMES = ("W", "N", "S", "D", "H1", "H2")
LINES = (
    ("WN", "W", "N", 2.0, 100.0),
    ("WS", "W", "S", 1.0, 100.0),
    ("WD", "W", "D", 0.5, 200.0),  # WD1 || WD2 merged
    ("ND", "N", "D", 2.0, 100.0),
    ("NH", "N", "H1", 2.0, 100.0),
    ("SD", "S", "D", 1.0, 100.0),
    ("SH", "S", "H2", 2.0, 100.0),
    ("DH", "D", "H2", 1.0, 100.0),
)
SWITCHES = (("H", "H1", "H2"),)  # (name, from, to); flow positive from -> to
SLACK = NODE_NAMES.index("D")

LINE_NAMES = tuple(l[0] for l in LINES)
SWITCH_NAMES = tuple(s[0] for s in SWITCHES)
ELEMENT_NAMES = LINE_NAMES + SWITCH_NAMES
n_nodes, n_lines, n_switches = len(NODE_NAMES), len(LINES), len(SWITCHES)


def _incidence(edges) -> np.ndarray:
    """Node x element incidence: +1 at the node an element leaves, -1 where it enters."""
    A = np.zeros((n_nodes, len(edges)))
    for j, (_, u, v, *_) in enumerate(edges):
        A[NODE_NAMES.index(u), j] = 1.0
        A[NODE_NAMES.index(v), j] = -1.0
    return A


A_L = _incidence(LINES)
A_S = _incidence(SWITCHES)
A = np.hstack([A_L, A_S])
X_L = np.array([l[3] for l in LINES])
Y_L = 1.0 / X_L
B_L = np.array([l[4] for l in LINES])


# ---------------------------------------------------------------------------
# One PTDF for every switch regime
# ---------------------------------------------------------------------------
def switch_ptdf(y_switch, y_line=Y_L, slack: int = SLACK) -> np.ndarray:
    """Node-space PTDF ``H`` (elements x nodes), lines first then switches, at
    switch susceptances ``y_switch`` in ``[0, inf]``.

    Finite-``y`` switches are lines.  Closed switches (``inf``) force a zero angle
    difference and carry a free flow, so with ``k`` of them the unknowns are the
    node angles plus ``k`` switch flows, and the equations are

        q = B theta + A_c f_c          (B: Laplacian of the finite-y elements)
        A_c^T theta = 0                (one per closed switch)

    Dropping the slack angle and the slack node's equation (power balance) leaves a
    square saddle system; its inverse gives every flow linearly in ``q``.  Taking
    ``y_switch -> inf`` in the relaxed PTDF recovers this row for row.
    """
    y_switch = np.asarray(y_switch, dtype=float)
    closed = np.isinf(y_switch)
    finite = ~closed & (y_switch > 0)
    open_ = y_switch == 0

    # finite-susceptance elements: all lines plus relaxed switches
    A_fin = np.hstack([A_L, A_S[:, finite]])
    y_fin = np.concatenate([np.asarray(y_line, dtype=float), y_switch[finite]])
    A_c = A_S[:, closed]
    k = A_c.shape[1]

    if not is_connected(np.hstack([A_fin, A_c])):
        raise ValueError("state islands a node: an open switch leaves it unattached")

    keep = [i for i in range(n_nodes) if i != slack]
    B = A_fin @ np.diag(y_fin) @ A_fin.T
    G = np.block([[B[np.ix_(keep, keep)], A_c[keep]], [A_c[keep].T, np.zeros((k, k))]])
    Ginv = np.linalg.inv(G)
    # theta (full, slack = 0) and f_c as linear maps from q_keep
    theta_map = np.zeros((n_nodes, n_nodes - 1))
    theta_map[keep] = Ginv[: n_nodes - 1, : n_nodes - 1]
    f_c_map = Ginv[n_nodes - 1 :, : n_nodes - 1]
    S = np.eye(n_nodes)[keep]  # q -> q_keep

    H_fin = np.diag(y_fin) @ A_fin.T @ theta_map @ S
    H = np.zeros((n_lines + n_switches, n_nodes))
    H[:n_lines] = H_fin[:n_lines]
    H[n_lines + np.where(finite)[0]] = H_fin[n_lines:]
    H[n_lines + np.where(closed)[0]] = f_c_map @ S
    # open switches: zero rows, already
    assert open_.sum() + finite.sum() + closed.sum() == n_switches
    return H


def model(
    y_switch,
    b_line=B_L,
    b_switch=np.inf,
    y_line=Y_L,
    key=None,
    label: str | None = None,
) -> NetworkModel:
    """A base-case ``NetworkModel`` on the six nodes at switch susceptances
    ``y_switch`` with per-element limits.  ``b_switch`` is the (symmetric) limit on
    each switch's flow; ``inf`` is the physical bus-branch model, which is unbounded
    in node space whenever a switch is closed."""
    y_switch = np.atleast_1d(np.asarray(y_switch, dtype=float))
    b_switch = np.broadcast_to(np.asarray(b_switch, dtype=float), (n_switches,))
    upper = np.concatenate([np.asarray(b_line, dtype=float), b_switch])
    rows = ContingencyRows(
        key=key,
        label=label or ("base" if key is None else str(key)),
        elements=ELEMENT_NAMES,
        H=switch_ptdf(y_switch, y_line),
        upper=upper,
        lower=upper.copy(),
    )
    return NetworkModel(nodes=NODE_NAMES, contingencies=(rows,))


# ---------------------------------------------------------------------------
# Bus space for a binary state: where the DAM prices, and where faces() runs
# ---------------------------------------------------------------------------
def bus_collector(state) -> tuple[np.ndarray, tuple[str, ...]]:
    """``M`` (buses x nodes) for a binary switch state: nodes joined by closed
    switches share a bus.  Bus names join their nodes' names."""
    state = np.atleast_1d(np.asarray(state, dtype=bool))
    parent = list(range(n_nodes))

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    for j, (_, u, v) in enumerate(SWITCHES):
        if state[j]:
            parent[find(NODE_NAMES.index(u))] = find(NODE_NAMES.index(v))
    roots = sorted({find(i) for i in range(n_nodes)})
    M = np.zeros((len(roots), n_nodes))
    for i in range(n_nodes):
        M[roots.index(find(i)), i] = 1.0
    names = tuple("".join(NODE_NAMES[i] for i in range(n_nodes) if M[r, i]) for r in range(len(roots)))
    return M, names


def bus_network(state) -> tuple[PhysicalNetwork, np.ndarray]:
    """The bus-branch ``PhysicalNetwork`` of a binary state (lines only, closed
    switches contracted, open ones dropped) and its collector ``M``.  A line with
    both ends in one bus keeps a zero column and carries no flow."""
    M, names = bus_collector(state)
    slack = int(np.where(M[:, SLACK])[0][0])
    net = PhysicalNetwork(
        A=M @ A_L, x=X_L, slack_idx=slack, node_names=np.array(names), element_names=np.array(LINE_NAMES)
    )
    return net, M


def dam_model(state, b_line=B_L) -> NetworkModel:
    """The physical day-ahead model at a binary state, in node space: closed
    switches unlimited (``b_switch = inf``), open ones absent."""
    y = np.where(np.atleast_1d(np.asarray(state, dtype=bool)), np.inf, 0.0)
    return model(y, b_line=b_line, b_switch=np.inf, label=f"dam{tuple(int(s) for s in y > 0)}")


def lift(v_bus: np.ndarray, M: np.ndarray) -> np.ndarray:
    """A bus-space direction to node space: every node takes its bus's value
    (``v = M^T v_bus``), which is exactly what the day-ahead market does to prices."""
    return M.T @ np.asarray(v_bus, dtype=float)


__all__ = [
    "NODE_NAMES", "LINES", "SWITCHES", "ELEMENT_NAMES", "A", "A_L", "A_S", "Y_L", "B_L",
    "switch_ptdf", "model", "bus_collector", "bus_network", "dam_model", "lift",
    "Contingency", "compute_ptdf",
]


# ---------------------------------------------------------------------------
# Scenarios: a physical state, its DAM model, and a realized direction v_t
# ---------------------------------------------------------------------------
import polars as pl
from typing import NamedTuple

from ..network import intersection
from ..polytope import faces
from ..solve import SupportProblem

CENTER = {"solver": "CLARABEL"}


class Scenario(NamedTuple):
    label: str
    state: tuple[int, ...]  # binary switch state of the interval
    dam: NetworkModel  # physical model, node space (closed switches unlimited)
    direction: np.ndarray  # v_t, node space, constant on each DAM bus
    weight: float = 1.0


def dam_faces(state, b_line=B_L) -> tuple[list, pl.DataFrame]:
    """Every vertex of the interval's DAM polytope, enumerated in bus space (where
    it is bounded) and lifted to nodes: ``(faces, table)``.  The table names each
    vertex's binding pattern (``+`` upper, ``-`` lower) and its lifted direction.
    """
    net, M = bus_network(state)
    bus = NetworkModel.build(net, [Contingency(None, b_line)])
    labels = bus.labels()
    fs = faces(bus)
    rows = []
    for i, f in enumerate(fs):
        lab = labels[f.rows.tolist()]
        pattern = " ".join(
            f"{e}{'+' if s == 'upper' else '-'}" for e, s in zip(lab["element"], lab["side"])
        )
        v = lift(f.direction, M)
        rows.append({"face": i, "pattern": pattern, **{f"v_{nm}": v[k] for k, nm in enumerate(NODE_NAMES)}})
    return fs, pl.DataFrame(rows)


def scenario(state, face_index: int, label: str, b_line=B_L, weight: float = 1.0) -> Scenario:
    """The scenario at ``state`` whose direction exposes vertex ``face_index`` of
    the DAM polytope (see :func:`dam_faces`)."""
    fs, _ = dam_faces(state, b_line)
    _, M = bus_network(state)
    state = tuple(int(s) for s in np.atleast_1d(state))
    return Scenario(label, state, dam_model(state, b_line), lift(fs[face_index].direction, M), weight)


# ---------------------------------------------------------------------------
# Evaluation: E[U], E[V] of an FTR model over the scenarios
# ---------------------------------------------------------------------------
def support(model: NetworkModel, direction: np.ndarray) -> float:
    """``h_Q(v)``; ``+inf`` when the polytope is unbounded along ``v`` (the dual is
    then infeasible and cvxpy returns no value)."""
    try:
        return SupportProblem(model, direction).solve(solver=CENTER).value
    except (TypeError, ValueError, IndexError):  # cvxpy leaves .value None when the dual is infeasible
        return np.inf


def evaluate(ftr: NetworkModel, scenarios: list[Scenario]) -> pl.DataFrame:
    """One row per scenario: ``h_ftr``, ``h_dam``, ``h_int``, ``U``, ``V``, and the
    weight.  ``E[U] = (weight * U).sum()``; likewise ``V``."""
    rows = []
    for sc in scenarios:
        h_ftr = support(ftr, sc.direction)
        h_dam = support(sc.dam, sc.direction)
        h_int = support(intersection(ftr, sc.dam), sc.direction)
        rows.append(
            {"scenario": sc.label, "weight": sc.weight, "h_ftr": h_ftr, "h_dam": h_dam,
             "h_int": h_int, "U": h_ftr - h_int, "V": h_dam - h_int}
        )
    return pl.DataFrame(rows)


def expected(table: pl.DataFrame) -> tuple[float, float]:
    """``(E[U], E[V])`` from an :func:`evaluate` table."""
    w = table["weight"].to_numpy()
    return float(w @ table["U"].to_numpy()), float(w @ table["V"].to_numpy())


# ---------------------------------------------------------------------------
# S1: design the limits b at a fixed y, by enumeration over dual vertices
# ---------------------------------------------------------------------------
import itertools

import cvxpy as cp

B_MAX = 1000.0  # cap on a designed limit; keeps unpriced limits finite for reporting


def dual_vertices(model: NetworkModel, direction: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """The vertices of ``Lambda(v) = {mu >= 0 : K^T mu + 1 s = v}`` over all of
    ``model``'s rows, as an array ``(k, n_rows)``.  Independent of ``b``.

    A vertex is a basic feasible solution: ``s`` is free so always basic, leaving
    ``n - 1`` basic ``mu``'s.  Enumerate the ``(n-1)``-subsets of rows, solve the
    square system, keep the nonnegative solutions, dedupe.  Each row is one
    congestion pattern ``direction`` can price on this network.
    """
    K, v = model.K, np.asarray(direction, dtype=float)
    m, n = K.shape
    subsets = np.array(list(itertools.combinations(range(m), n - 1)))
    # batch of (n x n) systems  [K_J^T  1] [mu_J; s] = v
    Mats = np.concatenate([np.transpose(K[subsets], (0, 2, 1)), np.ones((len(subsets), n, 1))], axis=2)
    ok = np.abs(np.linalg.det(Mats)) > tol
    sol = np.linalg.solve(Mats[ok], np.broadcast_to(v, (ok.sum(), n))[..., None])[..., 0]
    mu_J, feas = sol[:, :-1], (sol[:, :-1] >= -tol).all(axis=1)
    out = np.zeros((feas.sum(), m))
    for r, (J, mu) in enumerate(zip(subsets[ok][feas], mu_J[feas])):
        out[r, J] = np.maximum(mu, 0.0)
    return np.unique(np.round(out, 9), axis=0)


class LimitDesign(NamedTuple):
    b: np.ndarray  # designed limits, co-indexed with the FTR model's rows
    objective: float  # sum_w p_w v_w^T q_w  (= sum_w p_w h_int at the design)
    vertices: tuple[int, ...]  # chosen dual vertex per scenario
    n_lp: int


def design_limits(ftr: NetworkModel, scenarios: list[Scenario], tau: float, b_max: float = B_MAX,
                  symmetric_lines: bool = True) -> LimitDesign:
    """Memo eq. design-b-vertex, solved by enumeration over scenario vertex tuples.

    Fixed vertices make the problem an LP in ``(b, q_w)``: maximise the weighted
    intersection value subject to ``q_w`` feasible for the FTR rows at ``b`` and for
    the DAM's rows, and exposure ``sum p_w (mu_w^T b - v_w^T q_w) <= tau``.  The
    vertex ``mu_w`` and ``tau`` are Parameters, so the LP compiles once.
    """
    K, m, n = ftr.K, ftr.n_rows, ftr.n_nodes
    b = cp.Variable(m, nonneg=True)
    qs = [cp.Variable(n) for _ in scenarios]
    mus = [cp.Parameter(m, nonneg=True) for _ in scenarios]
    tau_p = cp.Parameter(nonneg=True, value=tau)
    p, v = [sc.weight for sc in scenarios], [sc.direction for sc in scenarios]

    cons = [b <= np.broadcast_to(np.asarray(b_max, dtype=float), (m,))]  # scalar or per-row fence
    if symmetric_lines:  # a line's rating is one number; the switch cap stays one-directional
        half = m // 2
        cons.append(b[:n_lines] == b[half : half + n_lines])
    for sc, q in zip(scenarios, qs):
        act = sc.dam.active
        cons += [K @ q <= b, cp.sum(q) == 0, sc.dam.K[act] @ q <= sc.dam.b[act]]
    value = sum(pw * (vw @ q) for pw, vw, q in zip(p, v, qs))
    exposure = sum(pw * (mu @ b) for pw, mu in zip(p, mus))
    cons.append(exposure - value <= tau_p)
    lp = cp.Problem(cp.Maximize(value), cons)

    verts = [dual_vertices(ftr, vw) for vw in v]
    best, n_lp = None, 0
    for choice in itertools.product(*(range(len(V)) for V in verts)):
        for mu, V, k in zip(mus, verts, choice):
            mu.value = V[k]
        try:
            lp.solve(solver=cp.HIGHS)
        except (cp.error.SolverError, ValueError):  # HiGHS reports some infeasible pairs as failures
            n_lp += 1
            continue
        n_lp += 1
        if lp.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and (best is None or lp.value > best.objective):
            best = LimitDesign(np.asarray(b.value), float(lp.value), choice, 0)
    if best is None:
        raise ValueError("no vertex choice is feasible at this tau")
    return best._replace(n_lp=n_lp)


def rating_fence(b_line, b_switch=B_MAX) -> np.ndarray:
    """Per-row cap for :func:`design_limits`: no FTR line limit above its physical
    rating (the ISO's rule); the switch, which has no rating, up to ``b_switch``."""
    half = np.concatenate([np.asarray(b_line, dtype=float), np.full(n_switches, float(b_switch))])
    return np.concatenate([half, half])


def with_designed_limits(ftr: NetworkModel, b: np.ndarray) -> NetworkModel:
    """``ftr`` with its (single contingency's) limits replaced by the designed ``b``."""
    (c,) = ftr.contingencies
    r = len(c.upper)
    from dataclasses import replace
    return NetworkModel(nodes=ftr.nodes, contingencies=(replace(c, upper=b[:r].copy(), lower=b[r:].copy()),))


# ---------------------------------------------------------------------------
# Scenarios by design: one physical rating vector, a chosen binding pattern per
# interval (reuses texas5.solve_limit_design on the stacked model of all states)
# ---------------------------------------------------------------------------
from .texas5 import solve_limit_design

# injection sign conventions for the design (W, N renewables inject; D, H load)
INJECT, WITHDRAW = ("W", "N"), ("D", "H1", "H2")


def _sign_extra(model, b, q):
    out = [q[NODE_NAMES.index(nm)] >= 0 for nm in INJECT]
    out += [q[NODE_NAMES.index(nm)] <= 0 for nm in WITHDRAW]
    return out


def design_scenarios(patterns: dict, weights=None, bounds=(10.0, 500.0)):
    """One rating vector ``b_line`` and one :class:`Scenario` per interval.

    ``patterns`` maps a label to ``(state, [(element, sign), ...])``: the lines to
    bind in that interval, ``+1`` upper / ``-1`` lower.  The states' DAM models are
    stacked; each pattern pins its rows in its own state's half, every line's limit
    is tied across halves (one physical rating), and the margin at every other row
    is maximised.  ``problem.value > 0`` is the realizability check.  The direction
    of a scenario is ``K^T 1_pattern`` on its state's rows -- positing the pattern
    is positing ``y``.

    Returns ``(b_line, scenarios, problem)``.
    """
    labels = list(patterns)
    states = [patterns[k][0] for k in labels]
    dams = [dam_model(st) for st in states]
    stack = intersection(*dams)
    m_half = stack.n_rows // 2
    r = n_lines + n_switches  # rows per state in one half

    def rows_of(i, elems):  # global row indices for pattern i
        out = []
        for e, sign in elems:
            j = ELEMENT_NAMES.index(e)
            out.append(i * r + j if sign > 0 else m_half + i * r + j)
        return out

    def extra(model, b, q):
        ties = [b[i * r + j] == b[j] for i in range(1, len(states)) for j in range(n_lines)]
        return ties + _sign_extra(model, b, q)

    pat_rows = {k: rows_of(i, patterns[k][1]) for i, k in enumerate(labels)}
    problem, b, q = solve_limit_design(stack, pat_rows, extra=extra, bounds=bounds)
    if problem.value is None or problem.value <= 0:
        raise ValueError(f"patterns not jointly realizable (margin {problem.value})")
    b_line = np.asarray(b.value)[:n_lines].copy()

    weights = [1.0 / len(labels)] * len(labels) if weights is None else list(weights)
    scenarios = []
    for i, (k, w) in enumerate(zip(labels, weights)):
        ones = np.zeros(stack.n_rows)
        ones[pat_rows[k]] = 1.0
        v = stack.K.T @ ones
        st = tuple(int(s) for s in np.atleast_1d(states[i]))
        scenarios.append(Scenario(k, st, dam_model(st, b_line=b_line), v, w))
    return b_line, scenarios, problem


# ---------------------------------------------------------------------------
# Scenarios from one injection: the states turn it into different flow patterns
# ---------------------------------------------------------------------------
def _state_ptdf(state):
    return switch_ptdf(np.where(np.array(state, dtype=bool), np.inf, 0.0))


def flows_table(intervals: dict) -> pl.DataFrame:
    """Each interval's injection under EVERY interval's state: columns
    ``f[q_k | s_j]``.  The diagonal is what each DAM sees; the off-diagonal is the
    cross-feasibility that decides whether ``V`` can be nonzero."""
    cols = {"element": list(ELEMENT_NAMES)}
    for k, (_, qk, _) in intervals.items():
        for j, (sj, _, _) in intervals.items():
            cols[f"{k} | state {j}"] = _state_ptdf(sj) @ np.asarray(qk, dtype=float)
    return pl.DataFrame(cols)


def scenarios_from_injections(intervals: dict, margin: float = 0.25, floor: float = 10.0, weights=None):
    """One rating vector and one :class:`Scenario` per interval.

    ``intervals`` maps a label to ``(state, q, [binding elements])``.  A binding
    line's rating is its flow in its own interval; every other line's rating is
    ``(1 + margin)`` times its largest flow over the intervals (at least ``floor``).
    A binding line must carry its largest flow in the interval it binds.  The
    direction is the pattern's, ``v = sum_e sign(f_e) h_e`` on the state's rows.
    Returns ``(b_line, scenarios, cross)`` where ``cross[k][j]`` is True when
    interval ``k``'s injection is feasible under interval ``j``'s state.
    """
    labels = list(intervals)
    states = [tuple(int(x) for x in np.atleast_1d(intervals[k][0])) for k in labels]
    qs = [np.asarray(intervals[k][1], dtype=float) for k in labels]
    for q in qs:
        assert abs(q.sum()) < 1e-9, "injection must balance"
    Hs = [_state_ptdf(st) for st in states]
    F = np.array([Hh @ q for Hh, q in zip(Hs, qs)])[:, :n_lines]  # own-state flows
    b_line = np.maximum(floor, (1 + margin) * np.abs(F).max(axis=0))
    for i, k in enumerate(labels):
        for e in intervals[k][2]:
            j = LINE_NAMES.index(e)
            if abs(F[i, j]) < np.abs(F[:, j]).max() - 1e-9:
                raise ValueError(f"{e} cannot bind in {k}: it carries more flow in another interval")
            b_line[j] = abs(F[i, j])
    cross = {
        k: {kj: bool((np.abs(Hs[j] @ qs[i])[:n_lines] <= b_line + 1e-9).all()) for j, kj in enumerate(labels)}
        for i, k in enumerate(labels)
    }
    weights = [1.0 / len(labels)] * len(labels) if weights is None else list(weights)
    scenarios = []
    for i, (k, w) in enumerate(zip(labels, weights)):
        v = sum(np.sign(F[i, LINE_NAMES.index(e)]) * Hs[i][LINE_NAMES.index(e)] for e in intervals[k][2])
        scenarios.append(Scenario(k, states[i], dam_model(states[i], b_line=b_line), v, w))
    return b_line, scenarios, cross


def match_flows(state, q0, targets: dict) -> np.ndarray:
    """The balanced injection closest to ``q0`` whose flows under ``state`` hit
    ``targets`` (``{line: flow}``) exactly.  Lets the same lines bind in two hours
    at equal flow, which is what the ratings rule needs for a shared pattern.
    Least-norm correction: minimise ``|q - q0|`` s.t. ``H_rows q = f``, ``1^T q = 0``."""
    q0 = np.asarray(q0, dtype=float)
    Hs = _state_ptdf(state)
    rows = np.vstack([Hs[LINE_NAMES.index(e)] for e in targets] + [np.ones(n_nodes)])
    rhs = np.array(list(targets.values()) + [0.0])
    dq = np.linalg.lstsq(rows, rhs - rows @ q0, rcond=None)[0]
    return q0 + dq


def design_limits_milp(ftr: NetworkModel, scenarios: list[Scenario], tau: float, b_max: float = B_MAX,
                       symmetric_lines: bool = True, verbose: bool = False) -> LimitDesign:
    """Same problem as :func:`design_limits`, with the choice of dual vertex per
    hour made by binaries instead of enumeration.

    ``h_w`` stands for the FTR model's reach in hour ``w``.  It must equal ONE of
    the vertex values ``mu_k^T b`` -- a disjunction.  Binaries ``z_{w,k}`` with
    ``sum_k z = 1`` pick the vertex, and

        h_w >= mu_k^T b - M_w (1 - z_{w,k})     for every k

    is tight for the chosen ``k`` and slack by ``M_w`` for the others (big-M).
    ``M_w = max_k mu_k^T b_max`` is the largest reach any vertex can have under the
    fence, so it never cuts a solution off.  One MILP replaces the product of LPs.
    """
    K, m, n = ftr.K, ftr.n_rows, ftr.n_nodes
    b_max = np.broadcast_to(np.asarray(b_max, dtype=float), (m,))
    b = cp.Variable(m, nonneg=True)
    qs = [cp.Variable(n) for _ in scenarios]
    p, v = [sc.weight for sc in scenarios], [sc.direction for sc in scenarios]
    verts = [dual_vertices(ftr, vw) for vw in v]

    cons = [b <= b_max]
    if symmetric_lines:
        half = m // 2
        cons.append(b[:n_lines] == b[half : half + n_lines])
    hs, zs = [], []
    for sc, q, V in zip(scenarios, qs, verts):
        act = sc.dam.active
        cons += [K @ q <= b, cp.sum(q) == 0, sc.dam.K[act] @ q <= sc.dam.b[act]]
        h = cp.Variable()
        z = cp.Variable(len(V), boolean=True)
        M = float((V @ b_max).max())
        cons += [cp.sum(z) == 1, h >= V @ b - M * (1 - z)]
        hs.append(h)
        zs.append(z)
    value = sum(pw * (vw @ q) for pw, vw, q in zip(p, v, qs))
    cons.append(sum(pw * h for pw, h in zip(p, hs)) - value <= tau)
    prob = cp.Problem(cp.Maximize(value), cons)
    prob.solve(solver=cp.HIGHS, verbose=verbose)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise ValueError(f"MILP status {prob.status}")
    choice = tuple(int(np.argmax(z.value)) for z in zs)
    return LimitDesign(np.asarray(b.value), float(prob.value), choice, 1)
