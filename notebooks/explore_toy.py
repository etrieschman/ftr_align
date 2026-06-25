# %%
import numpy as np
import polars as pl

from ftr_align import SupportProblem, clear_dam, dual_summary
from ftr_align.duality import (
    attribution_blocks,
    classify,
    marginal_repair,
    robust_bounds,
    shapley_repair,
    support_index,
    trade_matrix,
    trade_space,
)
from ftr_align.metrics import EPS
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
print(
    f"Δ  = h(g)-h(f) = {h_g.value - h_f.value:,.0f}   (>0 underfunding, <0 hedging ineff.)"
)
print(f"η  = h(g)/h(f) = {None if abs(h_f.value) < EPS else h_g.value / h_f.value:.3f}")

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
                "Delta": sol_g.value - sol_f.value,
                "eta": None if abs(sol_f.value) < EPS else sol_g.value / sol_f.value,
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
# Table IIb: dual attribution redundant models
blocks = []
for vname, (dam_model, ftr_model) in toy.REDUNDANT_MODELS.items():
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
dam_model, ftr_model = toy.REDUNDANT_MODELS["mixed"]
dam = clear_dam(dam_model, toy.SCENARIOS["(a)"], solver=CLEAR)
dam_prob = SupportProblem(dam_model, dam.direction)
ftr_prob = SupportProblem(ftr_model, dam.direction)

# DAM model
lo, hi = robust_bounds(dam_prob, solver=CLEAR)
index = support_index(hi)
klass = classify(lo, hi)
C = trade_matrix(dam_prob, index)
D = trade_space(C)
print("DAM support value:", round(dam_prob.solve(solver=CLEAR).value, 1))
print("DAM support rows :", index.tolist())
print("DAM classes      :", [klass[i] for i in index])
print("DAM trade space dim:", D.shape[1])
display(attribution_blocks(dam_prob, solver=CLEAR))


# FTR model
lo, hi = robust_bounds(ftr_prob, solver=CLEAR)
index = support_index(hi)
klass = classify(lo, hi)
C = trade_matrix(ftr_prob, index)
D = trade_space(C)
print("FTR support value:", round(ftr_prob.solve(solver=CLEAR).value, 1))
print("FTR support rows :", index.tolist())
print("FTR classes      :", [klass[i] for i in index])
print("FTR trade space dim:", D.shape[1])
display(attribution_blocks(ftr_prob, solver=CLEAR))


# %%
# Repair: per-block attribution of the gap Δ = h(g) - h(f).  marginal_repair =
# each block's standalone effect (not additive when drivers mask each other);
# shapley_repair = order-averaged, additive (sums to Δ).
dam_model, ftr_model = toy.REDUNDANT_MODELS[
    "mixed"
]  # both underfunding + hedging drivers
d = clear_dam(dam_model, toy.SCENARIOS["(a)"], solver=CLEAR).direction
(
    marginal_repair(dam_model, ftr_model, d, solver=CLEAR).join(
        shapley_repair(dam_model, ftr_model, d, solver=CLEAR),
        on=["driver", "members", "rows", "repair_rows"],
        how="full",
        coalesce=True,
        suffix="_shapley",
    )
)

# %%
