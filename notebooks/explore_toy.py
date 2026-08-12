# %%
import numpy as np
import polars as pl

from ftr_align import SupportProblem, clear_dam, dual_summary
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

CLEAR = {"solver": "CLARABEL"}  # interior-point → analytic-center certificate (paper numbers)

def show_blocks(problem, model, mu=None, index=None, solver=CLEAR):
    """Blocks with their attributed value W_{J_r}: compute in duality, label in
    metrics."""
    mu = problem.solve(solver=solver).mu if mu is None else mu
    blocks = attribution_blocks(problem, index=index)
    return block_table(model, blocks, block_totals(problem.data.b, mu, blocks))


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
dam = clear_dam(g_model, scenario, solver=CLEAR)
h_g = SupportProblem(g_model, dam.direction).solve(solver=CLEAR)  # DAM
h_f = SupportProblem(f_model, dam.direction).solve(solver=CLEAR)  # FTR/SFT
print("DAM_MS =", round(dam.merch_surp, 1))
print("d = K.T@y =", dam.direction)
print(f"h(g) = MS_DAM = {h_g.value:,.0f}")
print(f"h(f) = {h_f.value:,.0f}")
print(f"Δ  = h(f)-h(g) = {h_f.value - h_g.value:,.0f}")

# %%
# Table I -- one run_row per (case, scenario) cell.  run_row returns a dict, so
# the sweep is a list comprehension and a new column is a new key.
runs = pl.DataFrame([
    run_row(f_model, g_model,
            clear_dam(g_model, scenario, solver=CLEAR).direction,
            labels={"variation": vname, "scenario": sname}, solver=CLEAR)
    for vname, (f_model, g_model) in toy.MODELS.items()
    for sname, scenario in toy.SCENARIOS.items()
])
display(runs.select(["variation", "scenario", "h_f", "h_g", "Delta", "U", "V"]))

# The floor's share of each failure mode -- the T1 number.  Reported per mode
# because only *level* differences carry a floor at all (cor:diagnosable).
display(runs.select([
    "variation", "scenario", "U", "V",
    "floor_U", "floor_ratio_U", "floor_V", "floor_ratio_V",
]))

# Attribution shape: block count and trade-space dimension per cell (the N1
# question, in miniature).
display(runs.select([
    "variation", "scenario",
    "n_blocks_U", "max_block_U", "n_blocks_V", "max_block_V",
    "dim_trade_space_U", "dim_trade_space_V",
]))

# %%
# Table II: dual attribution
for model in [toy.MODELS, toy.REDUNDANT_MODELS]:
    blocks = []
    for vname, (f_model, g_model) in model.items():
        for sname, scenario in toy.SCENARIOS.items():
            dam = clear_dam(g_model, scenario, solver=CLEAR)
            sol_f = SupportProblem(f_model, dam.direction).solve(solver=CLEAR)
            sol_g = SupportProblem(g_model, dam.direction).solve(solver=CLEAR)
            blocks.append(
                dual_summary(
                    f_model,
                    sol_f,
                    g_model,
                    sol_g,
                    labels={"variation": vname, "scenario": sname},
                )
            )
    blocks_df = (
        pl.concat(blocks)
        .melt(
            id_vars=["variation", "scenario", "contingency", "element"], value_name="mu"
        )
        .pivot(
            index=["variation", "scenario", "variable"],
            on=["contingency", "element"],
            values="mu",
        )
        .sort(by=["variation", "scenario", "variable"])
    )
    display(blocks_df)


# %%
PATTERN = "mixed"
SCENARIO = "(a)"
print(f"PATTERN={PATTERN}, SCENARIO={SCENARIO}\n")
# Robust duals & attribution blocks (redundant variant)
f_model, g_model = toy.REDUNDANT_MODELS[PATTERN]
dam = clear_dam(g_model, toy.SCENARIOS[SCENARIO], solver=CLEAR)
dam_prob = SupportProblem(g_model, dam.direction)
ftr_prob = SupportProblem(f_model, dam.direction)

# DAM model
lo, hi = robust_bounds(dam_prob, solver=CLEAR)
index = J_star(dam_prob)
C = trade_matrix(dam_prob, index)
D = trade_space(C)
print("~~~~~~~~ DAM model")
print("DAM support value:", round(dam_prob.solve(solver=CLEAR).value, 1))
print("DAM support rows :", index.tolist())
print("DAM mu ranges    :", [(round(lo[i], 1), round(hi[i], 1)) for i in index])
print("DAM trade space dim:", D.shape[1])
display(show_blocks(dam_prob, g_model, index=index))


# FTR model
lo, hi = robust_bounds(ftr_prob, solver=CLEAR)
index = J_star(ftr_prob)
C = trade_matrix(ftr_prob, index)
D = trade_space(C)
print("\n~~~~~~~~ FTR model")
print("FTR support value:", round(ftr_prob.solve(solver=CLEAR).value, 1))
print("FTR support rows :", index.tolist())
print("FTR mu ranges    :", [(round(lo[i], 1), round(hi[i], 1)) for i in index])
print("FTR trade space dim:", D.shape[1])
display(show_blocks(ftr_prob, f_model, index=index))


# Failure modes and their block-level attribution.  U decomposes over the blocks
# of the FTR support problem, V over those of the DAM one (prop:block_underfunding).
d = clear_dam(g_model, toy.SCENARIOS[SCENARIO], solver=CLEAR).direction
print("\n~~~~~~~~ Failure modes")
print(failure_modes(f_model, g_model, d, solver=CLEAR))
for mode, model in (("U", f_model), ("V", g_model)):
    blocks, shares = block_shares(f_model, g_model, d, mode=mode, solver=CLEAR)
    print(f"\n{mode} by block")
    display(block_table(model, blocks, shares, value_name=mode))

# Per-constraint detail: limits, the difference kind, the certificate, the row's
# share, and the block it landed in.  Only rows that disagree or carry value.
print("\n~~~~~~~~ Per-constraint detail (V)")
display(row_table(f_model, g_model, d, mode="V", solver=CLEAR))

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
    d = clear_dam(g_model, toy.SCENARIOS[SCEN], solver=CLEAR).direction
    ax.set_xlim(-170, 170)
    ax.set_ylim(-210, 210)
    draw_region(ax, g_model, label=r"$\mathcal{Q}(g)$  DAM", **DAM_STYLE)
    draw_region(ax, f_model, label=r"$\mathcal{Q}(f)$  FTR", **FTR_STYLE)
    draw_region(ax, meet(f_model, g_model), label=r"$\mathcal{Q}(f \wedge g)$", **MEET_STYLE)
    draw_optimum(ax, g_model, d, solver=CLEAR, **DAM_STYLE)
    draw_optimum(ax, f_model, d, solver=CLEAR, **FTR_STYLE)
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
