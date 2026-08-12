# %%
import numpy as np
import polars as pl

from ftr_align import SupportProblem, clear_dam, dual_summary
from ftr_align.duality import (
    attribution_blocks,
    classify,
    marginal_repair,
    robust_bounds,
    J_star_from_bounds,
    trade_matrix,
    trade_space,
)
from ftr_align.metrics import EPS
from ftr_align.cases import toy

CLEAR = {"solver": "CLARABEL"}  # interior-point → analytic-center certificate (paper numbers)

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
# Table I
rows = []
for vname, (f_model, g_model) in toy.MODELS.items():
    for sname, scenario in toy.SCENARIOS.items():
        dam = clear_dam(g_model, scenario, solver=CLEAR)
        sol_f = SupportProblem(f_model, dam.direction).solve(solver=CLEAR)
        sol_g = SupportProblem(g_model, dam.direction).solve(solver=CLEAR)
        rows.append(
            {
                "variation": vname,
                "scenario": sname,
                "MS_DAM": sol_g.value,
                "Delta": sol_f.value - sol_g.value,
                "eta": None if abs(sol_g.value) < EPS else sol_f.value / sol_g.value,
            }
        )
pl.DataFrame(rows)

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
index = J_star_from_bounds(hi)
klass = classify(lo, hi)
C = trade_matrix(dam_prob, index)
D = trade_space(C)
print("~~~~~~~~ DAM model")
print("DAM support value:", round(dam_prob.solve(solver=CLEAR).value, 1))
print("DAM support rows :", index.tolist())
print("DAM classes      :", [klass[i] for i in index])
print("DAM trade space dim:", D.shape[1])
display(attribution_blocks(dam_prob, solver=CLEAR))


# FTR model
lo, hi = robust_bounds(ftr_prob, solver=CLEAR)
index = J_star_from_bounds(hi)
klass = classify(lo, hi)
C = trade_matrix(ftr_prob, index)
D = trade_space(C)
print("\n~~~~~~~~ FTR model")
print("FTR support value:", round(ftr_prob.solve(solver=CLEAR).value, 1))
print("FTR support rows :", index.tolist())
print("FTR classes      :", [klass[i] for i in index])
print("FTR trade space dim:", D.shape[1])
display(attribution_blocks(ftr_prob, solver=CLEAR))


# Repair: per-block attribution of the gap Δ(f,g;y) = h(f;y) - h(g;y).  Each
# block's standalone effect -- not additive when the two failure modes mask each
# other (prop:repair_nonadditive).
d = clear_dam(g_model, toy.SCENARIOS[SCENARIO], solver=CLEAR).direction
print("\n~~~~~~~~ Repair of gap")
marginal_repair(f_model, g_model, d, solver=CLEAR)

# %%
