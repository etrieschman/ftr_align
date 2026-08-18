# %%
import numpy as np
import scipy as sp
import polars as pl
import cvxpy as cp
from itertools import product
from tqdm import tqdm

from ftr_align import SupportProblem, NetworkModel, Contingency, with_limits
from ftr_align.solve import Lambda
from ftr_align.duality import (
    connected_blocks,
    robust_bounds,
    trade_matrix,
    trade_space,
)
from ftr_align.metrics import block_table, constraint_table, row_labels, summary
from ftr_align.cases import texas5
from ftr_align.cases.texas5 import (
    base_pattern_direction,
    base_pattern_rows,
    solve_limit_design,
)
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
# The designed limits make each pattern bind exactly, so each pattern gives a
# direction d = K^T y whose optimal face is the one we asked for.  No bids: the
# propositions hold at any y >= 0, and positing the pattern *is* positing y.
model = with_limits(TEMPLATE, b.value)


# One solve per pattern, reused for the check, the detail and the blocks.  J*
# comes off that same certificate rather than a second solve, which is why the
# solve is CENTER: `mu > 0` identifies J* only from the face's relative interior.
solved = {}
for name, pattern in PATTERNS.items():
    problem_p = SupportProblem(model, base_pattern_direction(model, pattern))
    solved[name] = (problem_p, problem_p.solve(solver=CENTER))

# Did the construction work?  One row per pattern rather than four tables to
# eyeball: `exact` is the whole check -- the designed limits should make the
# requested rows bind and nothing else.
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
