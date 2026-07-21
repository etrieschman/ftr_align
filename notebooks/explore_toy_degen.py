# %%
import numpy as np
import scipy as sp
import polars as pl
import cvxpy as cp

from ftr_align import SupportProblem, NetworkModel, Contingency
from ftr_align.solve import dual_feasible
from ftr_align.duality import (
    attribution_blocks,
    connected_blocks,
    classify,
    marginal_repair,
    robust_bounds,
    shapley_repair,
    support_index,
    trade_matrix,
    trade_space,
)
from ftr_align.metrics import EPS
from ftr_align.cases import toy_degen
from ftr_align.cases.toy_degen import WN, WS, WD1, WD2, ND, NH, SD, SH, DH
from ftr_align.cases.toy_degen import W, N, S, D, H

CLEAR = {"solver": "CLARABEL"}  # interior-point → analytic-center certificate (paper numbers)

pl.Config.set_tbl_rows(50)
pl.Config.set_fmt_table_cell_list_len(-1)
pl.Config.set_fmt_str_lengths(100)  # Increase to your maximum expected string length
np.set_printoptions(precision=3, suppress=True)

# %%
# -------------------------------------
# INSPECT NETWORK
# -------------------------------------
net = toy_degen.NETWORK
print("~~~~~~ Toy network:")
print("nodes   :", net.node_names.tolist())
print("elements:", net.element_names.tolist())
print("slack   :", net.node_names[net.slack_idx])
print("limits  :", toy_degen.BASE_LIMITS.tolist())
K = net.ptdf()
print(f"PTDF (line x node):\n{K}")
A = np.concatenate([K, -K], axis=0)
A_bar = A - np.mean(A, axis=1, keepdims=True)
active = [WS, WD1, SD, ND, NH, DH]
print(f"Connected blocks:{connected_blocks(A_bar[active].T)}")

# %%
# -------------------------------------
# ONE network + ONE b (from scenario iii); y selects the block per scenario
# -------------------------------------
q_iii    = np.array([300, 200, 100, -200, -400])       # fixes the limits
scenarios = {
    "i": [WD1, WD2], # reward WD corridor
    "ii":   [WN, WS, NH, SH], # reward perimeter
    "iii":[WN, ND, WD1, WD2, SD, SH, DH], # reward two triangles
}
b_plus = np.abs(np.round(K @ q_iii, 3))                # per-edge upper limit
b_plus = [
    b_plus[i] * 1.05 
    if i not in scenarios["iii"] else b_plus[i] 
    for i in range(len(b_plus))
    ]
b      = np.concatenate([b_plus, b_plus])            # [upper; lower] for A=[K;-K]

model = NetworkModel.build(net, contingencies=[Contingency(None, upper=b_plus)])

def make_y(edges, val=25.0):
    yv = np.zeros(len(net.element_names)*2)
    yv[edges] = val
    return yv

for name, rewarded in scenarios.items():
    y = make_y(rewarded)
    problem = SupportProblem(model, A.T @ y)
    h_b = problem.solve()
    print(f"\n{name}   h(b;y) = {h_b.value:,.0f}")
    print(f"binding =\n{h_b.binding}")
    display(attribution_blocks(problem))




# %%
# -------------------------------------
# MANUALLY CONSTRUCT DEGENERATE DISJOINT BLOCKS
# -------------------------------------
# Make disjoint connected blocks
# 0. choose active constraints that form disjoint blocks
active = [WN, WD1, WD2, ND, SD, SH, DH]
print(f"Connected blocks:{connected_blocks(A_bar[active].T)}")

# 1. choose injections
q = [300, 200, 100, -200, -400]
f = K @ q
print(f"balanced: {np.isclose(np.sum(q),0)}")
print(f"\t{net.node_names}\nq =\t{q}")
print(f"\t{net.element_names}\nf =\t{f}")

# 2. set the limits to make the chosen constraints active
b_plus = np.array([
    np.abs(np.round(fi, 3))
    if i in active else 400 
    for i,fi in enumerate(f)
    ], dtype=float) 
b = np.concatenate([b_plus, b_plus], axis=0)
y_val = np.ones(len(f))*25
y = np.array([
    y_val[i] 
    if i in active + [-i for i in active] else 0 
    for i,__ in enumerate(b)
])

# confirm via solve
qvar = cp.Variable(len(q))
prob = cp.Problem(
    cp.Maximize(y @ A @ qvar), 
    [A @ qvar <= b, cp.sum(qvar) == 0]
    )
prob.solve()
print(f"f@y =\t {f @ (y[:len(f)] - y[len(f):]):0.2f}  prob.value = {prob.value:0.2f}")
print(f"lmp =\t {-y @ A}")
print(f"MS_DAM = {y @ A @ q:,.2f}")

# %%
# -------------------------------------
# CONFIRM USING MACHINERY
# -------------------------------------
model = NetworkModel.build(net, contingencies=[Contingency(None, upper=b[:len(f)])])
problem = SupportProblem(model, A.T @ y)
h_b = problem.solve()
print(f"h(f) = MS_DAM = {h_b.value:,.0f}")
h_b.binding

blocks = attribution_blocks(problem)
# blocks.with_columns(
#     pl.col("members").list.join(separator=", "),
#     pl.col("rows").list.as_type(str).list.join(separator=", "),
# )
blocks

