# %%
import numpy as np
import polars as pl

from ftr_align import SupportProblem, clear_dam, dual_summary, gap, ratio
from ftr_align.duality import (
    attribution_blocks,
    classify,
    robust_bounds,
    support_index,
    trade_matrix,
    trade_space,
)
from ftr_align.cases import toy

CLEAR = "CLARABEL"  # interior-point → analytic-center certificate (paper numbers)

pl.Config.set_tbl_rows(40)
np.set_printoptions(precision=3, suppress=True)

# %%
net = toy.NETWORK
print("nodes   :", net.node_names.tolist())
print("elements:", net.element_names.tolist())
print("slack   :", net.node_names[net.slack_idx])
print("limits  :", toy.BASE_LIMITS.tolist())
print("\nPTDF (line x node):")
print(net.ptdf())

# %%
# Test one scenario
dam_model, ftr_model = toy.REDUNDANT_MODELS["derate"]
scenario = toy.SCENARIOS["(a)"]

dam = clear_dam(dam_model, scenario, solver=CLEAR)
print("DAM merchandising surplus:", round(dam.merch_surp, 1))
print("node-space direction d   :", dam.direction)

h_f = SupportProblem(dam_model, dam.direction).solve(solver=CLEAR)  # DAM
h_g = SupportProblem(ftr_model, dam.direction).solve(solver=CLEAR)  # FTR

print(f"\nMS_DAM = h(f) = {h_f.value:,.0f}")
print(f"h(g)          = {h_g.value:,.0f}")
print(f"Δ  = h(g)-h(f) = {gap(h_g, h_f):,.0f}   (>0 underfunding, <0 hedging ineff.)")
print(f"η  = h(g)/h(f) = {ratio(h_g, h_f):.3f}")

# %%
# Table I
rows = []
for vname, (dam_model, ftr_model) in toy.MODELS.items():
    for sname, scenario in toy.SCENARIOS.items():
        dam = clear_dam(dam_model, scenario, solver=CLEAR)
        sol_f = SupportProblem(dam_model, dam.direction).solve(solver=CLEAR)
        sol_g = SupportProblem(ftr_model, dam.direction).solve(solver=CLEAR)
        rows.append(
            {
                "variation": vname,
                "scenario": sname,
                "MS_DAM": sol_f.value,
                "Delta": gap(sol_g, sol_f),
                "eta": ratio(sol_g, sol_f),
            }
        )
pl.DataFrame(rows)

# %%
# Table II: dual attribution
blocks = []
for vname, (dam_model, ftr_model) in toy.MODELS.items():
    for sname, scenario in toy.SCENARIOS.items():
        dam = clear_dam(dam_model, scenario, solver=CLEAR)
        sol_f = SupportProblem(dam_model, dam.direction).solve(solver=CLEAR)
        sol_g = SupportProblem(ftr_model, dam.direction).solve(solver=CLEAR)
        blocks.append(
            dual_summary(
                dam_model,
                sol_f,
                ftr_model,
                sol_g,
                labels={"variation": vname, "scenario": sname},
            )
        )
(
    pl.concat(blocks)
    .melt(id_vars=["variation", "scenario", "contingency", "element"], value_name="mu")
    .pivot(
        index=["variation", "scenario", "variable"],
        columns=["contingency", "element"],
        values="mu",
    )
    .sort(by=["variation", "scenario", "variable"])
)


# %%
# Robust duals & attribution blocks (redundant variant)
dam_model = toy.REDUNDANT_MODELS["dam_outage"][0]
dam = clear_dam(dam_model, toy.SCENARIOS["(a)"], solver=CLEAR)
prob = SupportProblem(dam_model, dam.direction)
print("support value:", round(prob.solve(solver=CLEAR).value, 1))

lo, hi = robust_bounds(prob, solver=CLEAR)
index = support_index(hi)
klass = classify(lo, hi)
print("support rows :", index.tolist())
print("classes      :", [klass[i] for i in index])

C = trade_matrix(prob, index)
D = trade_space(C)
print("trade space dim:", D.shape[1])

attribution_blocks(prob, solver=CLEAR)
