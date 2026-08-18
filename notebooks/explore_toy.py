# %%
import numpy as np
import polars as pl

from ftr_align import SupportProblem, clear_dam
from ftr_align.attribution import block_shares, failure_modes
from ftr_align.duality import (
    J_star,
    attribution_blocks,
    block_totals,
    robust_bounds,
    trade_matrix,
    trade_space,
)
from ftr_align.metrics import block_table, row_labels, row_table, run_row
from ftr_align.cases import toy
from ftr_align.solve import CENTER

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
PATTERN, SCENARIO = "derate", "(a)"
print(f"\n~~~~~~ Test scenario:")
print(f"pattern={PATTERN}, scenario={SCENARIO}")
f_model, g_model = toy.REDUNDANT_MODELS[PATTERN]
scenario = toy.SCENARIOS[SCENARIO]
dam = clear_dam(g_model, scenario, solver=CENTER)
h_g = SupportProblem(g_model, dam.direction).solve(solver=CENTER)  # DAM
h_f = SupportProblem(f_model, dam.direction).solve(solver=CENTER)  # FTR/SFT
print("DAM_MS =", round(dam.merch_surp, 1))
print("d = K.T@y =", dam.direction)
print(f"h(g) = MS_DAM = {h_g.value:,.0f}")
print(f"h(f) = {h_f.value:,.0f}")
print(f"Δ  = h(f)-h(g) = {h_f.value - h_g.value:,.0f}")

# %%
# -------------------------------------
# SWEEP: one row per (case, scenario)
# -------------------------------------
# run_row(f, g, d) bundles what one cell needs.  Written out rather than
# comprehended so it is obvious what varies and what each call costs:
#
#   clear_dam(g, scenario) -> y*, and the direction d = K^T y* it induces
#   run_row(f, g, d)       -> failure_modes (3 support solves) + per-mode floor,
#                             block count and trade-space dimension
runs = []
for case, (f_model, g_model) in toy.MODELS.items():
    for scenario_name, scenario in toy.SCENARIOS.items():
        dam = clear_dam(g_model, scenario, solver=CENTER)
        runs.append(
            run_row(
                f_model,
                g_model,
                dam.direction,
                labels={"case": case, "scenario": scenario_name},
                solver=CENTER,
            )
        )
runs = pl.DataFrame(runs).with_columns(pl.col(pl.Float64).round(1))

# (1) The gap and its two one-signed halves.  Delta = U - V identically.
display(runs.select("case", "scenario", "h_g", "h_f", "Delta", "U", "V"))

# %%
# (2) How much of each failure mode the floor explains.  Only *level*
# differences can carry a floor at all, so a coverage-difference case shows a
# structural zero rather than a small number.
display(
    runs.select(
        "case", "scenario",
        "U", "floor_U", "floor_ratio_U",
        "V", "floor_V", "floor_ratio_V",
    )
)

# %%
# (3) Attribution shape.  Blocks PARTITION the priced rows, so n_blocks counts
# groups rather than ambiguity: two rows that cannot trade give TWO singleton
# blocks, which is the fully-identified case.  The ambiguity is dim_trade_space
# (0 = every row individually attributable) or equivalently max_block (1 = same).
# The plain 3-node has no parallel elements, so it is all singletons.
display(
    runs.select(
        "case", "scenario",
        "n_priced_V", "n_blocks_V", "max_block_V", "dim_trade_space_V",
    )
)

# %%
PATTERN = "mixed"
SCENARIO = "(a)"
print(f"PATTERN={PATTERN}, SCENARIO={SCENARIO}\n")
# Robust duals & attribution blocks (redundant variant)
f_model, g_model = toy.REDUNDANT_MODELS[PATTERN]
dam = clear_dam(g_model, toy.SCENARIOS[SCENARIO], solver=CENTER)
dam_prob = SupportProblem(g_model, dam.direction)
ftr_prob = SupportProblem(f_model, dam.direction)

# One CENTER solve per model carries everything: the value, mu, J* by strict
# complementarity, the blocks built on J*, and the block totals.  robust_bounds
# is the exception -- its face LPs run on HiGHS and its base solve is pinned to
# that engine, so it cannot share this certificate and does its own.
for name, model, prob in (("DAM", g_model, dam_prob), ("FTR", f_model, ftr_prob)):
    sol = prob.solve(solver=CENTER)
    index = J_star(prob, sol)
    blocks = attribution_blocks(prob, index)
    D = trade_space(trade_matrix(prob, index))
    lo, hi = robust_bounds(prob, solver=CENTER)

    print(f"\n~~~~~~~~ {name} model")
    print(f"{name} support value:", round(sol.value, 1))
    print(f"{name} support rows :", index.tolist())
    print(f"{name} mu ranges    :", [(round(lo[i], 1), round(hi[i], 1)) for i in index])
    print(f"{name} trade space dim:", D.shape[1])
    display(block_table(model, blocks, block_totals(prob.data.b, sol.mu, blocks)))


# Failure modes and their block-level attribution.  U decomposes over the blocks
# of the FTR support problem, V over those of the DAM one (prop:block_underfunding).
d = clear_dam(g_model, toy.SCENARIOS[SCENARIO], solver=CENTER).direction
print("\n~~~~~~~~ Failure modes")
print(failure_modes(f_model, g_model, d, solver=CENTER))
for mode, model in (("U", f_model), ("V", g_model)):
    blocks, shares = block_shares(f_model, g_model, d, mode=mode, solver=CENTER)
    print(f"\n{mode} by block")
    display(block_table(model, blocks, shares, value_name=mode))

# Per-constraint detail: limits, the difference kind, the certificate, the row's
# share, and the block it landed in.  Only rows that disagree or carry value.
print("\n~~~~~~~~ Per-constraint detail (V)")
display(row_table(f_model, g_model, d, mode="V", solver=CENTER))

# %%

# %%
# -------------------------------------
# GEOMETRY: the injection polytopes
# -------------------------------------
# At three nodes power balance leaves a 2-D injection space, so this picture is
# exact -- not a projection.  Q(f^g) traces whichever model is tighter, which is
# cor:canonical made visible.
import matplotlib.pyplot as plt

from ftr_align import meet
from ftr_align.polytope import faces, polygon
from ftr_align.viz import (
    DAM_STYLE, FTR_STYLE, MEET_STYLE,
    draw_optimum, draw_region, label_axes,
)

SCEN = "(a)"
fig, axes = plt.subplots(1, len(toy.MODELS), figsize=(4.4 * len(toy.MODELS), 4.2))
for ax, (case, (f_model, g_model)) in zip(axes, toy.MODELS.items()):
    d = clear_dam(g_model, toy.SCENARIOS[SCEN], solver=CENTER).direction
    ax.set_xlim(-170, 170)
    ax.set_ylim(-210, 210)
    draw_region(ax, g_model, label=r"$\mathcal{Q}(g)$  DAM", **DAM_STYLE)
    draw_region(ax, f_model, label=r"$\mathcal{Q}(f)$  FTR", **FTR_STYLE)
    draw_region(ax, meet(f_model, g_model), label=r"$\mathcal{Q}(f \wedge g)$", **MEET_STYLE)
    draw_optimum(ax, g_model, d, solver=CENTER, **DAM_STYLE)
    draw_optimum(ax, f_model, d, solver=CENTER, **FTR_STYLE)
    label_axes(ax, g_model)
    ax.axhline(0, lw=0.4, c="k")
    ax.axvline(0, lw=0.4, c="k")
    ax.set_title(f"{case}  ({SCEN})")
axes[0].legend(fontsize=8, loc="upper left", frameon=False)
fig.tight_layout()

# %%
# -------------------------------------
# REGIMES: every realizable active set
# -------------------------------------
# faces() enumerates the vertices of Q(b) with the rows tight at each and a
# direction exposing it.  Vertices and normal-fan cones are dual, so this *is*
# the list of active sets attainable at an optimum -- the d-sweep as a lookup
# rather than a scan.  In 2-D they come back in cyclic order, i.e. sweep order.
for case, (f_model, g_model) in toy.MODELS.items():
    print(f"\n~~~~~~~~ {case}")
    for name, model in (("f (FTR)", f_model), ("g (DAM)", g_model)):
        print(f"  {name}: {len(faces(model))} vertices")
        for face in faces(model):
            rows = ", ".join(row_labels(model, face.rows))
            print(f"    q={np.round(face.q, 1)}  tight={{{rows}}}")
