# %%
import numpy as np
import scipy as sp
import polars as pl
import cvxpy as cp
from itertools import product

from ftr_align import SupportProblem, NetworkModel, Contingency
from ftr_align.solve import Lambda
from ftr_align.duality import (
    J_star,
    attribution_blocks,
    block_totals,
    connected_blocks,
    robust_bounds,
    trade_matrix,
    trade_space,
)
from ftr_align.metrics import block_table, support_summary
from ftr_align.cases import texas5
from ftr_align.cases.texas5 import WN, WS, WD1, WD2, ND, NH, SD, SH, DH
from ftr_align.cases.texas5 import W, N, S, D, H

CLEAR = {"solver": "CLARABEL"}  # interior-point → analytic-center certificate (paper numbers)

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


# %%
# -------------------------------------
# LIMIT DESIGN
# -------------------------------------
def solve_limit_design(patterns):
    q = {name: cp.Variable(n, name=f"q_{name}") for name in patterns}
    b = cp.Variable(ell, name="b")
    delta = cp.Variable(nonneg=True, name="delta")

    const = [
        b >= 10,
        b <= 500,
        b[WD1] == 100,
        b[WD2] == b[WD1],
    ]

    for name, pattern in patterns.items():
        active = dict(pattern)
        f = H_base @ q[name]

        if len(active) != len(pattern):
            raise ValueError(f"{name} contains a repeated element")

        const += [
            cp.sum(q[name]) == 0,
            q[name][W] >= 0,
            q[name][N] >= 0,
            q[name][H] <= 0,
            q[name] >= -1_000,
            q[name] <= 1_000,
        ]

        for e in range(ell):
            if e in active:
                const += [f[e] == active[e] * b[e]]
            else:
                const += [
                    f[e] <= b[e] - delta,
                    f[e] >= -b[e] + delta,
                ]

    problem = cp.Problem(cp.Maximize(delta), const)
    problem.solve(solver=cp.HIGHS)
    return problem, b, q


# %%
# -------------------------------------
# SEARCH DIRECTIONS
# -------------------------------------
results = []
tol = 1e-6

for directions in product([-1, +1], repeat=8):
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

    problem, b, _ = solve_limit_design(patterns)

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        continue
    if problem.value <= tol:
        continue

    results.append({
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
    })

if not results:
    raise RuntimeError("No strictly feasible patterns found")

display(
    pl.DataFrame([
        {k: v for k, v in r.items() if k not in {"patterns", "limits"}}
        for r in results
    ]).sort(["delta", "n_pos"], descending=True)
)


# %%
# -------------------------------------
# SELECT BEST PATTERN
# -------------------------------------
best = max(results, key=lambda r: (r["delta"], r["n_pos"]))
PATTERNS = best["patterns"]

problem, b, q = solve_limit_design(PATTERNS)

print("status:", problem.status)
print("margin:", problem.value)

display(pl.DataFrame({
    "element": net.element_names,
    "limit": b.value,
}))


# %%
# -------------------------------------
# VERIFY PATTERNS AND BLOCKS
# -------------------------------------
# The designed limits make each pattern bind exactly, so each pattern gives a
# direction d = K^T y whose optimal face is the one we asked for.  No bids: the
# propositions hold at any y >= 0, and positing the pattern *is* positing y.
model = NetworkModel.build(
    net,
    [Contingency(None, upper=b.value)],
)


def pattern_direction(pattern):
    """d = K^T y for the certificate that puts unit weight on the pattern's
    signed rows -- upper row for +1, lower row for -1."""
    y = np.zeros(model.n_rows)
    for e, direction in pattern:
        y[e if direction == +1 else ell + e] = 1.0
    return model.K.T @ y


summaries = []
for name, pattern in PATTERNS.items():
    requested = dict(pattern)
    designed_flow = H_base @ q[name].value

    support_problem = SupportProblem(model, pattern_direction(pattern))
    solution = support_problem.solve(solver=CLEAR)

    rows = []

    for e, element in enumerate(net.element_names):
        for direction, row in [("upper", e), ("lower", ell + e)]:
            signed_flow = designed_flow[e] if direction == "upper" else -designed_flow[e]
            target_direction = requested.get(e)

            rows.append({
                "constraint": f"base : {element} : {direction}",
                "flow": signed_flow,
                "limit": b.value[e],
                "slack": b.value[e] - signed_flow,
                "target": (
                    target_direction == +1 and direction == "upper"
                ) or (
                    target_direction == -1 and direction == "lower"
                ),
                "binding": solution.binding[row].item(),
            })

    print(f"\n{name}")
    print(f"h(b; y) = {solution.value:,.2f}")

    display(
        pl.DataFrame(rows)
        .filter(pl.col("target") | pl.col("binding"))
        .sort("constraint")
    )

    blocks = attribution_blocks(support_problem)
    display(block_table(model, blocks,
                        block_totals(support_problem.data.b, solution.mu, blocks)))

    summaries.append(
        support_summary(support_problem, labels={"pattern": name}, solver=CLEAR)
    )


# %%
# -------------------------------------
# LIGHT SUMMARY ACROSS PATTERNS
# -------------------------------------
# The single-model counterpart of run_row.  dim_trade_space > 0 is the whole
# point of this network: it means some attributed value is unidentified at the
# constraint level and only the block total is reportable.
display(
    pl.DataFrame(summaries).with_columns(pl.col(pl.Float64).round(2))
)

# %%
