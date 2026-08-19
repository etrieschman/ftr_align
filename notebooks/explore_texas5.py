# %%
import numpy as np
import scipy as sp
import polars as pl
import cvxpy as cp
from itertools import product
from tqdm import tqdm

from ftr_align import SupportProblem, NetworkModel, Contingency, meet, with_limits
from ftr_align.metrics import (
    block_table,
    constraint_table,
    row_labels,
    summary,
    gap_summary,
)
from ftr_align.cases import texas5
from ftr_align.cases.texas5 import (
    base_pattern_direction,
    base_pattern_rows,
    solve_limit_design,
)
from ftr_align.polytope import faces
from ftr_align.cases.texas5 import WN, WS, WD1, WD2, ND, NH, SD, SH, DH
from ftr_align.cases.texas5 import W, N, S, D, H

from ftr_align.solve import (
    CENTER,
)  # {"solver": "CLARABEL"}: the interior certificate J* needs

pl.Config.set_tbl_rows(50)
pl.Config.set_fmt_table_cell_list_len(-1)
pl.Config.set_fmt_str_lengths(100)  # Increase to your maximum expected string length
np.set_printoptions(precision=3, suppress=True)

# %%
# -------------------------------------
# INSPECT NETWORK
# -------------------------------------
net = texas5.NETWORK
H_base = net.ptdf()  # not `H`: that is the node index imported above
n = net.n_nodes
ell = net.n_elements

print("~~~~~~ Toy network")
print("nodes   :", net.node_names.tolist())
print("elements:", net.element_names.tolist())
print("slack   :", net.node_names[net.slack_idx])
print("x       :", net.x.tolist())

# Only this model's K is used by the designer -- its limits are what we solve for.
TEMPLATE = NetworkModel.build(net, [Contingency(None, texas5.BASE_LIMITS)])


def extra(b, q):
    """Case-specific ties: the parallel pair must stay indistinguishable, and W/N
    generate while H consumes."""
    return [
        b[WD1] == 100,
        b[WD2] == b[WD1],
        q[W] >= 0,
        q[N] >= 0,
        q[H] <= 0,
        q >= -1_000,
        q <= 1_000,
    ]


# %%
# -------------------------------------
# SEARCH DIRECTIONS
# -------------------------------------
results = []
tol = 1e-6

for directions in tqdm(product([-1, +1], repeat=8)):
    (
        d_parallel,
        d_wd_block,
        d_nd,
        d_sd,
        d_wn,
        d_ws,
        d_nh,
        d_sh,
    ) = directions

    patterns = {
        "parallel_wd": [
            (WD1, d_parallel),
            (WD2, d_parallel),
        ],
        "two_blocks": [
            (WN, +1),
            (ND, d_nd),
            (WD1, d_wd_block),
            (WD2, d_wd_block),
            (SD, d_sd),
            (DH, +1),
            (SH, +1),
        ],
        "outer_loop": [
            (WN, d_wn),
            (NH, d_nh),
            (SH, d_sh),
            (WS, d_ws),
        ],
        "no_loop": [
            (WN, +1),
            (SH, +1),
        ],
    }

    rows = {
        name: base_pattern_rows(TEMPLATE, pattern) for name, pattern in patterns.items()
    }
    problem, b, _ = solve_limit_design(TEMPLATE, rows, extra=extra)

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        continue
    if problem.value <= tol:
        continue

    results.append(
        {
            "patterns": patterns,
            "limits": b.value.copy(),
            "delta": float(problem.value),
            "n_pos": sum(d > 0 for d in directions),
            "parallel_WD": d_parallel,
            "block_WD": d_wd_block,
            "ND": d_nd,
            "SD": d_sd,
            "outer_WN": d_wn,
            "outer_WS": d_ws,
            "outer_NH": d_nh,
            "outer_SH": d_sh,
        }
    )

if not results:
    raise RuntimeError("No strictly feasible patterns found")

display(
    pl.DataFrame(
        [
            {k: v for k, v in r.items() if k not in {"patterns", "limits"}}
            for r in results
        ]
    ).sort(["delta", "n_pos"], descending=True)
)


# %%
# -------------------------------------
# SELECT BEST PATTERN
# -------------------------------------
best = max(results, key=lambda r: (round(r["delta"], 2), r["n_pos"]))
PATTERNS = best["patterns"]

ROWS = {n: base_pattern_rows(TEMPLATE, p) for n, p in PATTERNS.items()}
problem, b, q = solve_limit_design(TEMPLATE, ROWS, extra=extra)

print("status:", problem.status)
print("margin:", problem.value)

display(
    pl.DataFrame(
        {
            "element": net.element_names,
            # b is over rows; limits are symmetric, so the upper half is all of it
            "limit": b.value[:ell],
        }
    )
)


# %%
# -------------------------------------
# VERIFY PATTERNS AND BLOCKS
# -------------------------------------
# No bids: the # propositions hold at any y >= 0.
model = with_limits(TEMPLATE, b.value)

# One solve per pattern, reused for the check, the detail and the blocks.
solved = {}
for name, pattern in PATTERNS.items():
    problem_p = SupportProblem(model, base_pattern_direction(model, pattern))
    solved[name] = (problem_p, problem_p.solve(solver=CENTER))

# Did the construction work?  One row per pattern -- the designed limits should
# make the requested rows bind and nothing else.
display(
    pl.DataFrame(
        [
            {
                "pattern": name,
                "h": sol.value,
                "requested": len(ROWS[name]),
                "bound": int(sol.binding.sum()),
                "exact": ROWS[name] == set(np.flatnonzero(sol.binding).tolist()),
                "rows": row_labels(model, np.flatnonzero(sol.binding)),
            }
            for name, (_, sol) in solved.items()
        ]
    ).with_columns(pl.col("h").round(2))
)


# %%
# -------------------------------------
# PER-PATTERN DETAIL
# -------------------------------------
for name, pattern in PATTERNS.items():
    problem_p, sol = solved[name]
    print(f"\n~~~~~~ Pattern {name}:")
    display(pl.DataFrame(summary(model, problem_p.data.direction)))
    display(constraint_table(model, problem_p.data.direction))
    display(block_table(model, problem_p.data.direction))


# %%
# -------------------------------------
# DO CROSS-CONTINGENCY BLOCKS EXIST?
# -------------------------------------
# A block needs a trade z with  sum z_i kbar_i = 0  AND  sum z_i b_i = 0.  The
# first is b-free -- pure linear algebra on the stacked PTDF -- so it says which
# row sets *could ever* be a block, before any limit is designed.
#
# kbar rows are mean-removed, hence orthogonal to 1, hence live in a 4-dim
# subspace: any 5 rows are dependent, so minimal dependent sets (circuits) have
# size <= 5.  A circuit is rank(S) == |S| - 1 with a null vector of FULL support
# -- a zero entry means a smaller dependent set is hiding inside.
#
# Lower rows are negated upper rows, so search upper rows only; the sign of z
# picks the side later.  The outaged element's own row is identically zero under
# its own contingency, so it is dropped.
from itertools import combinations

MAX_CIRCUIT = 5  # rank is 4, so circuits run to 5; 4 catches the cheap majority

circuits = []
for c in tqdm(range(ell)):
    tmpl = NetworkModel.build(
        net, [Contingency(None, texas5.BASE_LIMITS), Contingency(c, texas5.BASE_LIMITS)]
    )
    base_rows = {int(i) for i in tmpl.rows_upper(None)}
    upper = [
        int(i)
        for i in list(tmpl.rows_upper(None)) + list(tmpl.rows_upper(c))
        if np.abs(tmpl.K[int(i)]).max() > 1e-9  # drop the dead row
    ]
    kbar = {i: tmpl.K[i] - tmpl.K[i].mean() for i in upper}

    for size in range(2, MAX_CIRCUIT + 1):
        for S in combinations(upper, size):
            M = np.array([kbar[i] for i in S])
            if np.linalg.matrix_rank(M, tol=1e-8) != size - 1:
                continue
            z = np.linalg.svd(M.T, full_matrices=True)[2][-1]
            if np.abs(z).min() < 1e-8:  # not minimal
                continue
            circuits.append(
                {
                    "outage": net.element_names[c],
                    "size": size,
                    "spans": bool(set(S) & base_rows) and bool(set(S) - base_rows),
                    "rows": row_labels(tmpl, list(S)),
                }
            )

circ = pl.DataFrame(circuits)
print(f"{circ.height} circuits, {circ['spans'].sum()} spanning base and contingency")
display(
    circ.pivot(
        index="size", on="spans", values="outage", aggregate_function="len"
    ).sort("size")
)
display(circ.filter(pl.col("spans")).sort(["size", "outage"]).head(20))


# %%
# -------------------------------------
# STEPS 1-3: CHOOSE OUTAGE, PATTERNS, AND DESIGN THE LIMITS
# -------------------------------------
# Ranked by small spanning circuits, but selected by *realizability*: a circuit
# can be perfectly good and still give margin 0, because `extra` ties b[WD2] to
# b[WD1] and the pair is parallel with equal reactance -- so binding one forces
# the other, and a non-pattern row sits at its limit.  Rather than special-case
# that, take the first (outage, pattern set) with a strictly positive margin.
rank = (
    circ.filter(pl.col("size") <= 3, pl.col("spans"))
    .group_by("outage").len().sort("len", descending=True)
)
display(rank)


def circuits_for(model, outage, base_up, sizes=(2, 3)):
    """Minimal dependent sets among usable upper rows, with their null vector.
    Mixed signs only: sum z_i b_i = 0 is otherwise unsatisfiable with b > 0."""
    rows = [
        int(i)
        for i in list(model.rows_upper(None)) + list(model.rows_upper(outage))
        if np.abs(model.K[int(i)]).max() > 1e-9
    ]
    kb = {i: model.K[i] - model.K[i].mean() for i in rows}
    out = []
    for size in sizes:
        for S in combinations(rows, size):
            M = np.array([kb[i] for i in S])
            if np.linalg.matrix_rank(M, tol=1e-8) != size - 1:
                continue
            z = np.linalg.svd(M.T, full_matrices=True)[2][-1]
            if np.abs(z).min() < 1e-8 or z.min() > -1e-8 or z.max() < 1e-8:
                continue
            out.append({"rows": list(S), "z": z.tolist(), "size": size,
                        "spans": bool(set(S) & base_up) and bool(set(S) - base_up)})
    return out


DESIGN = None
for outage_name in rank["outage"]:
    OUTAGE = int(np.flatnonzero(net.element_names == outage_name)[0])
    tmpl = NetworkModel.build(
        net,
        [Contingency(None, texas5.BASE_LIMITS), Contingency(OUTAGE, texas5.BASE_LIMITS)],
    )
    base_up = {int(i) for i in tmpl.rows_upper(None)}
    found = circuits_for(tmpl, OUTAGE, base_up)
    span = sorted([f for f in found if f["spans"]], key=lambda f: f["size"])
    base = sorted([f for f in found if not f["spans"]], key=lambda f: f["size"])

    for S_span in span[:8]:
        disjoint = [f for f in base if not set(f["rows"]) & set(S_span["rows"])]
        for S_base in (disjoint[:4] + [None]):  # richest set first
            patterns = {"span_block": S_span["rows"]}
            trades = [(S_span["rows"], S_span["z"])]
            if S_base is not None:
                patterns["base_block"] = S_base["rows"]
                trades.append((S_base["rows"], S_base["z"]))
            pr, b_d, q_d = solve_limit_design(tmpl, patterns, extra=extra, trades=trades)
            if pr.value is not None and pr.value > 1e-6:
                DESIGN = dict(outage=OUTAGE, tmpl=tmpl, patterns=patterns,
                              margin=float(pr.value), b=b_d.value.copy(),
                              span=S_span, base=S_base)
                break
        if DESIGN:
            break
    if DESIGN:
        break

if DESIGN is None:
    raise RuntimeError("no (outage, pattern set) realizable with a positive margin")

OUTAGE, tmpl = DESIGN["outage"], DESIGN["tmpl"]
print(f"outage  : {net.element_names[OUTAGE]}   margin: {DESIGN['margin']:.3f}")
for name, rows in DESIGN["patterns"].items():
    print(f"{name:11}: {row_labels(tmpl, rows)}")
display(
    pl.DataFrame(
        {
            "element": net.element_names,
            "base": DESIGN["b"][list(tmpl.rows_upper(None))].round(2),
            f"out_{net.element_names[OUTAGE]}": DESIGN["b"][
                list(tmpl.rows_upper(OUTAGE))
            ].round(2),
        }
    )
)


# %%
# -------------------------------------
# STEP 4: BUILD THE f-g PAIR
# -------------------------------------
# The design IS the meet, so f must carry the designed base limits and g must be
# looser there: meet = min(b_base, b_base/alpha) = b_base, and the contingency
# rows come from g alone.  FTR sees only the base case.
# The span circuit must be a block of *g* (V attributes on g's blocks), and its
# trade `sum z_i b_i = 0` was imposed on the designed limits.  So derate only the
# base rows OUTSIDE the circuit's support -- otherwise alpha breaks the trade on
# the very rows meant to form the block.
ALPHA = 0.9
b_base = DESIGN["b"][list(tmpl.rows_upper(None))]
b_cont = DESIGN["b"][list(tmpl.rows_upper(OUTAGE))]

span_elems = {int(r) % ell for r in DESIGN["span"]["rows"]}
b_g_base = np.array(
    [b if e in span_elems else b / ALPHA for e, b in enumerate(b_base)]
)

f_model = NetworkModel.build(net, [Contingency(None, b_base)])
g_model = NetworkModel.build(
    net, [Contingency(None, b_g_base), Contingency(OUTAGE, b_cont)]
)
m_model = meet(f_model, g_model)
print("meet reproduces the design:",
      bool(np.allclose(m_model.b[np.isfinite(m_model.b)], DESIGN["b"], atol=1e-9)))


# %%
# -------------------------------------
# STEP 5: INSPECT AT EACH PATTERN DIRECTION
# -------------------------------------
# U comes from the contingency rows f is blind to; V from the base rows g is
# looser on.  The headline is whether a block's members carry both a `base:` and
# an outage label -- that is a block spanning contingencies.
for name, rows in DESIGN["patterns"].items():
    y = np.zeros(tmpl.n_rows)
    y[list(rows)] = 1.0
    d = tmpl.K.T @ y

    print(f"\n~~~~~~ {name}: {row_labels(tmpl, rows)}")
    display(pl.DataFrame(gap_summary(f_model, g_model, d, solver=CENTER)))
    for mode, looser in (("U", f_model), ("V", g_model)):
        tbl = block_table(looser, d, m_model, labels={"mode": mode})
        spans = [
            any(lbl.startswith("base:") for lbl in mem)
            and any(not lbl.startswith("base:") for lbl in mem)
            for mem in tbl["members"]
        ]
        display(tbl.with_columns(spans=pl.Series(spans)).drop("rows"))
# %%
