# %%
import numpy as np
import polars as pl

import matplotlib.pyplot as plt

from ftr_align import SupportProblem, clear_dam, meet
from ftr_align.metrics import block_table, constraint_table, gap_summary, row_labels
from ftr_align.cases import toy
from ftr_align.solve import CENTER
from ftr_align.polytope import basis_from_columns, faces
from ftr_align.viz import (
    DAM_STYLE,
    FTR_STYLE,
    draw_optimum,
    draw_region,
    frame_axes,
)

# CENTER is the library's name for {"solver": "CLARABEL"} -- the analytic-centre
# certificate.  Required for J* and anything built on it (blocks, trade space),
# and the convention the paper's numbers follow.

pl.Config.set_tbl_rows(40)
np.set_printoptions(precision=3, suppress=True)

# %%
# -------------------------------------
# INSPECT NETWORK AND SINGLE SCENARIO
# -------------------------------------
net = toy.NETWORK
print("~~~~~~ Toy network:")
print("nodes   :", net.node_names.tolist())
print("elements:", net.element_names.tolist())
print("slack   :", net.node_names[net.slack_idx])
print("limits  :", toy.BASE_LIMITS.tolist())
print(f"PTDF (line x node):\n{net.ptdf()}")

# Test one scenario
# PATTERN, SCENARIO = "derate", "(a)"
# print(f"\n~~~~~~ Test scenario:")
# print(f"pattern={PATTERN}, scenario={SCENARIO}")
# f_model, g_model = toy.REDUNDANT_MODELS[PATTERN]
# scenario = toy.SCENARIOS[SCENARIO]
# dam = clear_dam(g_model, scenario, solver=CENTER)
# h_g = SupportProblem(g_model, dam.direction).solve(solver=CENTER)  # DAM
# h_f = SupportProblem(f_model, dam.direction).solve(solver=CENTER)  # FTR/SFT
# print("DAM_MS =", round(dam.merch_surp, 1))
# print("d = K.T@y =", dam.direction)
# print(f"h(g) = MS_DAM = {h_g.value:,.0f}")
# print(f"h(f) = {h_f.value:,.0f}")
# print(f"Δ  = h(f)-h(g) = {h_f.value - h_g.value:,.0f}")

# %%
# -------------------------------------
# SWEEP: one row per (case, scenario)
# -------------------------------------
runs = []
for case, (f_model, g_model) in toy.MODELS.items():
    for scenario_name, scenario in toy.SCENARIOS.items():
        dam = clear_dam(g_model, scenario, solver=CENTER)
        runs.append(
            gap_summary(
                f_model,
                g_model,
                dam.direction,
                labels={"case": case, "scenario": scenario_name},
                solver=CENTER,
            )
        )
runs_df = pl.DataFrame(runs).with_columns(pl.col(pl.Float64).round(2))
display(runs_df)


# %%
# -------------------------------------
# BLOCKS: what each is worth, and what it costs
# -------------------------------------
# block_table carries two quantities over one partition:
#
#   value  sums to h(model)               constant on the dual face, no range
#   loss   sums to h(model) - h(target)   read at one q in Q(target); the range
#                                         is over that choice
#
# The failure mode is which model you pass first: (f, d, f^g) is U, (g, d, f^g)
# is V -- always the model that loses the value.  Drop the target for the value
# half alone.
#
# The redundant variant is the interesting one: SLa/SLb are parallel with
# identical PTDF rows, so they land in one block and cannot be split.
PATTERN, SCENARIO = "mixed", "(a)"
f_model, g_model = toy.REDUNDANT_MODELS[PATTERN]
m_model = meet(f_model, g_model)
d = clear_dam(g_model, toy.SCENARIOS[SCENARIO], solver=CENTER).direction

print(f"PATTERN={PATTERN}, SCENARIO={SCENARIO}")
display(pl.DataFrame(gap_summary(f_model, g_model, d, solver=CENTER)))

# One table per failure mode.  Reading a row: this block is worth value
# and accounts for loss, with a range depending on which q_meet is used
for mode, looser in (("U", f_model), ("V", g_model)):
    display(block_table(looser, d, m_model, labels={"mode": mode}))

# Per-constraint detail underneath the blocks: limits, the difference kind, the
# certificate, the row's share, and the block it landed in.  Only rows that
# disagree or carry value.
display(constraint_table(g_model, d, m_model, solver=CENTER))

# %%
# -------------------------------------
# GEOMETRY: the injection polytopes
# -------------------------------------
# T is used to set which variables are drawn on the x/y axes.
# The default is to drop the slack. In our case we want x = -L and y = S,
# so we construct T: <T,(x,y)> = (L, S, C)
T = basis_from_columns([(0.0, 1.0, -1.0), (1.0, -1.0, 0.0)])
XLABEL, YLABEL = r"$L$  (load served, MW)", r"$q_S$  (solar dispatch, MW)"

for scen in toy.SCENARIOS:
    fig, axes = plt.subplots(1, len(toy.MODELS), figsize=(4.4 * len(toy.MODELS), 4.2))
    for ax, (case, (f_model, g_model)) in zip(axes, toy.MODELS.items()):
        d = clear_dam(g_model, toy.SCENARIOS[scen], solver=CENTER).direction
        # Framing first: the drawing layers clip to the *current* axis limits.
        frame_axes(
            ax,
            g_model,
            title=f"{case}  {scen}",
            xlim=(-100, 200),
            ylim=(-100, 200),
            xlabel=XLABEL,
            ylabel=YLABEL,
        )
        draw_region(ax, g_model, T, label=r"$\mathcal{Q}(g)$  DAM", **DAM_STYLE)
        draw_region(ax, f_model, T, label=r"$\mathcal{Q}(f)$  FTR", **FTR_STYLE)
        draw_optimum(ax, g_model, d, T, solver=CENTER, **DAM_STYLE)
        draw_optimum(ax, f_model, d, T, solver=CENTER, **FTR_STYLE)
    axes[0].legend(fontsize=8, loc="upper left", frameon=False)
    fig.tight_layout()

# %%
# -------------------------------------
# REGIMES: every realizable vertex (doesn't include facets)
# -------------------------------------
# faces() enumerates the vertices of Q(b) with the rows tight at each and a
# direction exposing it.
for case, (f_model, g_model) in toy.MODELS.items():
    print(f"\n~~~~~~~~ {case}")
    for name, model in (("f (FTR)", f_model), ("g (DAM)", g_model)):
        print(f"  {name}: {len(faces(model))} vertices")
        for face in faces(model):
            rows = ", ".join(row_labels(model, face.rows))
            print(f"    q={np.round(face.q, 1)}  tight={{{rows}}}")

# %%
