"""The five-node network with a station switch: the ex-ante design case.

Node-breaker version of :mod:`texas5`.  Station ``H`` is split into two nodes
``H1`` (the ``NH`` side) and ``H2`` (the ``SH``/``DH`` side) joined by a switch.
The parallel ``WD1``/``WD2`` pair is merged into one line ``WD`` (reactance 0.5,
limit 200, electrically identical).

Every element is a row of one incidence matrix ``A = [A_L; A_S]``.  A switch
differs from a line only by its susceptance ``y``:

    open      y = 0          the element is absent
    relaxed   y in (0, inf)  an ordinary line with reactance 1/y
    closed    y = inf        its two nodes share one angle; its flow is free

A switch state *is* its susceptance (``CLOSED = inf``, ``OPEN = 0``), and the
day-ahead model at a state is :func:`model` with the switch unlimited.
:func:`switch_ptdf` is the PTDF at any ``y``, on the fixed six-node set, so every
downstream object (``NetworkModel``, ``intersection``, ``gap_summary``) is
unchanged.  One switch only.

Contents, in the order a session uses them:

    switch_ptdf, model         the PTDF and an FTR or DAM model at a switch susceptance
    Q, four_hours              the design case's hours: one rating vector, one
                               Scenario per hour
    scenarios_from_injections  the same from any {hour: (state, q, binding lines)}
    flows_table, match_flows   each injection under every state; shared patterns
    support, evaluate          h, U, V per hour
    dual_vertices              the congestion patterns a direction can price
    design_limits, design_limits_milp, rating_fence
                               S1: design the limits at a fixed susceptance
"""

from __future__ import annotations

import itertools
import warnings
from typing import NamedTuple

import cvxpy as cp
import numpy as np
import polars as pl

from ..network import ContingencyRows, NetworkModel, compute_ptdf, intersection
from ..solve import SupportProblem

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
SWITCHES = (("H", "H1", "H2"),)  # (name, from, to); flow positive H1 -> H2
SLACK = NODE_NAMES.index("D")

LINE_NAMES = tuple(l[0] for l in LINES)
SWITCH_NAMES = tuple(s[0] for s in SWITCHES)
ELEMENT_NAMES = LINE_NAMES + SWITCH_NAMES
n_nodes, n_lines, n_switches = len(NODE_NAMES), len(LINES), len(SWITCHES)
W, N, S, D, H1, H2 = range(n_nodes)  # node indices
WN, WS, WD, ND, NH, SD, SH, DH = range(n_lines)  # line indices

CLOSED, OPEN = np.inf, 0.0  # a switch state is its susceptance


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

# the closed switch makes H1 and H2 one bus: M collects nodes (columns) into buses (rows)
M_CLOSED = np.zeros((n_nodes - 1, n_nodes))
M_CLOSED[:4, :4] = np.eye(4)  # W, N, S, D are buses of their own
M_CLOSED[4, [H1, H2]] = 1.0  # bus H


# ---------------------------------------------------------------------------
# The PTDF at a switch susceptance
# ---------------------------------------------------------------------------
def switch_ptdf(y_switch: float, y_line=Y_L) -> np.ndarray:
    """PTDF (elements x nodes), the eight lines then the switch, with the switch at
    susceptance ``y_switch``: ``0`` open, finite relaxed, ``inf`` closed."""
    x_line = 1.0 / np.asarray(y_line, dtype=float)
    if y_switch == 0:  # open: the lines alone, and the switch carries nothing
        H_lines = compute_ptdf(A_L, x_line, SLACK)
        H_switch = np.zeros(n_nodes)
    elif np.isfinite(y_switch):  # relaxed: the switch is a line of reactance 1/y
        H = compute_ptdf(A, np.append(x_line, 1.0 / y_switch), SLACK)
        H_lines, H_switch = H[:n_lines], H[n_lines]
    else:  # closed: H1 and H2 are one bus, so the lines see the five-bus network ...
        H_lines = compute_ptdf(M_CLOSED @ A_L, x_line, SLACK) @ M_CLOSED
        # ... and the switch carries whatever H1 injects that H1's own lines do not
        H_switch = np.eye(n_nodes)[H1] - A_L[H1] @ H_lines
    return np.vstack([H_lines, H_switch])


def model(y_switch, b_line=B_L, b_switch=np.inf, y_line=Y_L, label: str | None = None) -> NetworkModel:
    """A base-case ``NetworkModel`` on the six nodes at switch susceptances
    ``y_switch`` with per-element limits.

    ``b_switch`` is the (symmetric) limit on each switch's flow.  ``inf`` with the
    switch closed is the physical day-ahead model, unbounded in node space along
    the H1/H2 shift; a finite value is a design choice (finding 8).
    """
    upper = np.append(np.asarray(b_line, dtype=float), float(b_switch))
    rows = ContingencyRows(
        key=None,
        label=label or f"y_switch={y_switch:g}",
        elements=ELEMENT_NAMES,
        H=switch_ptdf(y_switch, y_line),
        upper=upper,
        lower=upper.copy(),
    )
    return NetworkModel(nodes=NODE_NAMES, contingencies=(rows,))


# ---------------------------------------------------------------------------
# Hours: a physical switch state, an injection, the lines that bind, and from
# those the DAM model and the realized direction v_t
# ---------------------------------------------------------------------------
class Scenario(NamedTuple):
    label: str
    y_switch: float  # the hour's physical switch state, CLOSED or OPEN
    q: np.ndarray  # the hour's injection
    binding: list[str]  # the lines that bind in the hour
    dam: NetworkModel  # the hour's DAM model, node space, switch unlimited
    direction: np.ndarray  # v_t, node space, constant on each DAM bus
    weight: float = 1.0


def scenarios_from_injections(hours: dict, margin: float = 0.25, floor: float = 10.0, weights=None):
    """One rating vector and one :class:`Scenario` per hour.

    ``hours`` maps a label to ``(y_switch, q, [binding lines])``.  A binding line's
    rating is its flow in its own hour; every other line's rating is
    ``(1 + margin)`` times its largest flow over the hours (at least ``floor``).  A
    binding line must carry its largest flow in the hour it binds.  The direction
    is the pattern's: ``v_t = sum_e sign(f_e) h_e`` over the binding lines, on the
    hour's own PTDF.

    Returns ``(b_line, scenarios, cross)`` where ``cross[k][j]`` is True when hour
    ``k``'s injection is feasible under hour ``j``'s switch state at ``b_line``.
    """
    for _, q, _ in hours.values():
        assert abs(np.sum(q)) < 1e-9, "injection must balance"
    weights = [1.0 / len(hours)] * len(hours) if weights is None else list(weights)

    # each hour's line flows under its own switch state
    flows = {k: (switch_ptdf(y) @ q)[:n_lines] for k, (y, q, _) in hours.items()}

    # ratings: a binding line at its flow in its hour, every other line at its largest flow plus the margin
    largest = np.max(np.abs(list(flows.values())), axis=0)
    b_line = np.maximum(floor, (1 + margin) * largest)
    for k, (_, _, binding) in hours.items():
        for e in binding:
            j = LINE_NAMES.index(e)
            if abs(flows[k][j]) < largest[j] - 1e-9:
                raise ValueError(f"{e} cannot bind in {k}: it carries more flow in another hour")
            b_line[j] = abs(flows[k][j])

    # hour k's injection under hour j's switch state, within these ratings?
    cross = {
        k: {j: bool((np.abs(switch_ptdf(yj) @ qk)[:n_lines] <= b_line + 1e-9).all()) for j, (yj, _, _) in hours.items()}
        for k, (_, qk, _) in hours.items()
    }

    scenarios = []
    for (k, (y, q, binding)), w in zip(hours.items(), weights):
        H = switch_ptdf(y)
        v = sum(np.sign(flows[k][LINE_NAMES.index(e)]) * H[LINE_NAMES.index(e)] for e in binding)
        scenarios.append(Scenario(k, y, q, binding, model(y, b_line=b_line, label=f"dam {k}"), v, w))
    return b_line, scenarios, cross


def flows_table(scenarios: list[Scenario]) -> pl.DataFrame:
    """Each hour's injection under EVERY hour's switch state, columns
    ``"q_<k> | switch of <j>"``.  The diagonal is what each DAM sees; the
    off-diagonal is the cross-feasibility that decides whether ``V`` can be nonzero."""
    cols = {"element": list(ELEMENT_NAMES)}
    for k in scenarios:
        for j in scenarios:
            cols[f"q_{k.label} | switch of {j.label}"] = switch_ptdf(j.y_switch) @ k.q
    return pl.DataFrame(cols)


def match_flows(y_switch, q0, targets: dict) -> np.ndarray:
    """The balanced injection closest to ``q0`` whose flows at switch state
    ``y_switch`` hit ``targets`` (``{line: flow}``) exactly.  Lets the same lines
    bind in two hours at equal flow, which the ratings rule needs for a shared
    pattern.  Least-norm correction: ``min |q - q0|  s.t.  H_rows q = f, 1^T q = 0``."""
    q0 = np.asarray(q0, dtype=float)
    H = switch_ptdf(y_switch)
    rows = np.vstack([H[LINE_NAMES.index(e)] for e in targets] + [np.ones(n_nodes)])
    rhs = np.array(list(targets.values()) + [0.0])
    dq = np.linalg.lstsq(rows, rhs - rows @ q0, rcond=None)[0]
    return q0 + dq


# ---------------------------------------------------------------------------
# The hours of the design case
#
# The injections are chosen by hand: W and N inject renewables, S gas, D and the
# two halves of station H withdraw.  Everything else is computed from them by
# scenarios_from_injections: the ratings, the directions, the cross-feasibility.
# ---------------------------------------------------------------------------
Q = {  # hour:          W      N      S       D      H1      H2
    1: np.array([120.0,  40.0,  40.0,  -80.0,  -60.0,  -60.0]),  # evening: west wind, Houston load high
    2: np.array([ 90.0, 110.0,  10.0, -120.0,  -80.0,  -10.0]),  # midday: north wind and solar
    3: np.array([ 40.0,  30.0, 120.0,  -80.0,  -65.0,  -45.0]),  # evening: south gas
    4: np.array([100.0,  90.0,  20.0,  -60.0,  -30.0, -120.0]),  # midday, south
}


def four_hours(margin: float = 0.10):
    """The four-hour instance: ``(b_line, scenarios, cross)``, labels ``"1"``..``"4"``.
    Hour 4 is moved the least distance that makes DH and WD carry hour 1's flows
    under its own open switch (so hours 1 and 4 share a pattern) and ND stay just
    under hour 2's."""
    flows_1 = switch_ptdf(CLOSED) @ Q[1]
    q4 = match_flows(OPEN, Q[4], {"DH": flows_1[DH], "WD": flows_1[WD], "ND": 24.0})
    hours = {
        "1": (CLOSED, Q[1], ["DH", "WD"]),
        "2": (OPEN, Q[2], ["NH", "ND"]),
        "3": (CLOSED, Q[3], ["SD", "SH"]),
        "4": (OPEN, q4, ["DH", "WD"]),
    }
    return scenarios_from_injections(hours, margin)


# ---------------------------------------------------------------------------
# Evaluation: h, U, V of an FTR model over the hours
# ---------------------------------------------------------------------------
CENTER = {"solver": "CLARABEL"}


def support(model: NetworkModel, direction: np.ndarray) -> float:
    """``h_Q(v)``; ``+inf`` when the polytope is unbounded along ``v`` (the dual is
    then infeasible and cvxpy returns no value)."""
    try:
        return SupportProblem(model, direction).solve(solver=CENTER).value
    except (TypeError, ValueError, IndexError):  # cvxpy leaves .value None when the dual is infeasible
        return np.inf


def evaluate(ftr: NetworkModel, scenarios: list[Scenario]) -> pl.DataFrame:
    """One row per hour: ``h_ftr``, ``h_dam``, ``h_int``, ``U``, ``V``, and the
    weight.  ``E[U]`` is ``(weight * U).sum()``; likewise ``V``."""
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


# ---------------------------------------------------------------------------
# S1: design the limits b at a fixed susceptance (memo eq. design-b-vertex)
# ---------------------------------------------------------------------------
B_MAX = 1000.0  # the fence when none is given; keeps unpriced limits finite for reporting


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


def rating_fence(b_line, b_switch=B_MAX) -> np.ndarray:
    """The per-row fence for the design: no FTR line limit above its rating (the
    ISO's rule); the switch, which has no rating, up to ``b_switch``.  Co-indexed with
    the rows of :func:`model`: ``[upper lines, upper switches, lower lines, lower switches]``."""
    half = np.concatenate([np.asarray(b_line, dtype=float), np.full(n_switches, float(b_switch))])
    return np.concatenate([half, half])


class LimitDesign(NamedTuple):
    b: np.ndarray  # designed limits, co-indexed with the FTR model's rows
    objective: float  # sum_t p_t v_t^T q_t  (= sum_t p_t h_int at the design)
    vertices: tuple[int, ...]  # chosen dual vertex per hour
    n_lp: int  # LPs solved (enumeration) or 1 (MILP)
    status: str  # solver status; "user_limit" means an incumbent at the time limit
    gap: float  # relative MIP gap at return (0 for the enumeration)


SWITCH_MODES = ("fixed", "symmetric", "per_direction")


def _design_lp(ftr: NetworkModel, scenarios: list[Scenario], b_max, symmetric_lines: bool, switch: str):
    """The part of memo eq. design-b-vertex both solvers share: limits ``b`` under
    the fence, one injection ``q_t`` per hour feasible for the FTR rows at ``b``
    and for the hour's DAM rows, and the weighted intersection value.

    ``switch`` says what the design may do with the switch's limit: ``"fixed"``
    leaves it at the value ``ftr`` carries (the design chooses line limits only),
    ``"symmetric"`` designs one number for both directions, ``"per_direction"``
    designs the two directions separately.
    Returns ``(b, qs, cons, value)``."""
    if switch not in SWITCH_MODES:
        raise ValueError(f"switch must be one of {SWITCH_MODES}")
    K, m, n = ftr.K, ftr.n_rows, ftr.n_nodes
    half = m // 2
    up_switch, lo_switch = n_lines, half + n_lines  # the switch's two rows
    b = cp.Variable(m, nonneg=True)
    qs = [cp.Variable(n) for _ in scenarios]
    b_max = np.broadcast_to(np.asarray(b_max, dtype=float), (m,)).copy()  # scalar or per-row fence
    if switch == "fixed":  # an unlimited switch is pinned at B_MAX, which the fence then admits
        fixed = np.minimum(ftr.b[[up_switch, lo_switch]], B_MAX)
        b_max[[up_switch, lo_switch]] = np.maximum(b_max[[up_switch, lo_switch]], fixed)
    cons = [b <= b_max]
    if symmetric_lines:  # a line's rating is one number
        cons.append(b[:n_lines] == b[half : half + n_lines])
    if switch == "fixed":
        cons += [b[up_switch] == fixed[0], b[lo_switch] == fixed[1]]
    elif switch == "symmetric":
        cons.append(b[up_switch] == b[lo_switch])
    for sc, q in zip(scenarios, qs):
        act = sc.dam.active
        cons += [K @ q <= b, cp.sum(q) == 0, sc.dam.K[act] @ q <= sc.dam.b[act]]
    value = sum(sc.weight * (sc.direction @ q) for sc, q in zip(scenarios, qs))
    return b, qs, cons, value


def design_limits(ftr: NetworkModel, scenarios: list[Scenario], tau: float, b_max=B_MAX,
                  symmetric_lines: bool = True, switch: str = "fixed") -> LimitDesign:
    """Memo eq. design-b-vertex by enumeration over hour vertex tuples.

    Fixed vertices make the problem an LP in ``(b, q_t)``: maximise the weighted
    intersection value subject to exposure ``sum p_t (mu_t^T b - v_t^T q_t) <= tau``.
    The vertex ``mu_t`` and ``tau`` are Parameters, so the LP compiles once and the
    enumeration is a warm parameter sweep.
    """
    b, qs, cons, value = _design_lp(ftr, scenarios, b_max, symmetric_lines, switch)
    mus = [cp.Parameter(ftr.n_rows, nonneg=True) for _ in scenarios]
    tau_p = cp.Parameter(nonneg=True, value=tau)
    exposure = sum(sc.weight * (mu @ b) for sc, mu in zip(scenarios, mus))
    lp = cp.Problem(cp.Maximize(value), cons + [exposure - value <= tau_p])

    verts = [dual_vertices(ftr, sc.direction) for sc in scenarios]
    best, n_lp = None, 0
    for choice in itertools.product(*(range(len(V)) for V in verts)):
        for mu, V, k in zip(mus, verts, choice):
            mu.value = V[k]
        n_lp += 1
        try:
            lp.solve(solver=cp.HIGHS)
        except (cp.error.SolverError, ValueError):  # HiGHS reports some infeasible pairs as failures
            continue
        if lp.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and (best is None or lp.value > best.objective):
            best = LimitDesign(np.asarray(b.value), float(lp.value), choice, 0, lp.status, 0.0)
    if best is None:
        raise ValueError("no vertex choice is feasible at this tau")
    return best._replace(n_lp=n_lp)


def design_limits_milp(ftr: NetworkModel, scenarios: list[Scenario], tau: float, b_max=B_MAX,
                       symmetric_lines: bool = True, switch: str = "fixed", solver: str | None = None,
                       time_limit: float | None = None, verbose: bool = False, **solver_opts) -> LimitDesign:
    """Same problem as :func:`design_limits`, with the choice of dual vertex per
    hour made by binaries instead of enumeration.

    ``h_t`` stands for the FTR model's reach in hour ``t``.  It must equal ONE of
    the vertex values ``mu_k^T b`` -- a disjunction.  Binaries ``z_{t,k}`` with
    ``sum_k z = 1`` pick the vertex, and

        h_t >= mu_k^T b - M_t (1 - z_{t,k})     for every k

    is tight for the chosen ``k`` and slack by ``M_t`` for the others (big-M).
    ``M_t = max_k mu_k^T b_max`` is the largest reach any vertex can have under the
    fence, so it never cuts a solution off; a loose fence on the switch inflates it
    and slows the search.

    ``solver`` is ``cp.GUROBI`` when gurobipy is installed (its license lives
    outside the repo, in ``~/gurobi.lic``), else ``cp.HIGHS``; ``time_limit`` is
    seconds, spelled for whichever runs; other ``solver_opts`` go through as the
    solver spells them.  At the time limit the incumbent is returned with
    ``status="user_limit"`` and its ``gap``; only a run with no incumbent raises.
    """
    solver = solver or (cp.GUROBI if cp.GUROBI in cp.installed_solvers() else cp.HIGHS)
    if time_limit is not None:
        solver_opts["TimeLimit" if solver == cp.GUROBI else "time_limit"] = time_limit
    b_max = np.broadcast_to(np.asarray(b_max, dtype=float), (ftr.n_rows,))
    b, qs, cons, value = _design_lp(ftr, scenarios, b_max, symmetric_lines, switch)
    hs, zs = [], []
    for sc in scenarios:
        V = dual_vertices(ftr, sc.direction)
        h = cp.Variable()
        z = cp.Variable(len(V), boolean=True)
        M = float((V @ b_max).max())
        cons += [cp.sum(z) == 1, h >= V @ b - M * (1 - z)]
        hs.append(h)
        zs.append(z)
    exposure = sum(sc.weight * h for sc, h in zip(scenarios, hs))
    prob = cp.Problem(cp.Maximize(value), cons + [exposure - value <= tau])
    try:
        prob.solve(solver=solver, verbose=verbose, **solver_opts)
    except Exception as err:  # gurobipy raises its own GurobiError when it cannot open the license
        if solver != cp.GUROBI or "Gurobi" not in type(err).__name__ + str(err):
            raise
        warnings.warn(f"Gurobi could not start ({err}); solving with HiGHS instead", stacklevel=2)
        solver_opts.pop("TimeLimit", None)
        if time_limit is not None:
            solver_opts["time_limit"] = time_limit
        prob.solve(solver=cp.HIGHS, verbose=verbose, **solver_opts)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE, cp.USER_LIMIT) or b.value is None:
        raise ValueError(f"no design: MILP status {prob.status}")
    stats = prob.solver_stats.extra_stats  # HiGHS: HighsInfo.mip_gap; Gurobi: Model.MIPGap
    gap = float(getattr(stats, "MIPGap", getattr(stats, "mip_gap", np.nan)))
    choice = tuple(int(np.argmax(z.value)) for z in zs)
    return LimitDesign(np.asarray(b.value), float(prob.value), choice, 1, prob.status, gap)


__all__ = [
    "NODE_NAMES", "LINES", "SWITCHES", "LINE_NAMES", "ELEMENT_NAMES", "A", "A_L", "A_S", "X_L", "Y_L", "B_L", "M_CLOSED",
    "W", "N", "S", "D", "H1", "H2", "WN", "WS", "WD", "ND", "NH", "SD", "SH", "DH",
    "CLOSED", "OPEN", "switch_ptdf", "model",
    "Scenario", "Q", "four_hours",
    "flows_table", "scenarios_from_injections", "match_flows",
    "support", "evaluate",
    "dual_vertices", "rating_fence", "LimitDesign", "SWITCH_MODES", "design_limits", "design_limits_milp",
]
