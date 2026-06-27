# %%
import numpy as np
import polars as pl
from tqdm import tqdm

from ftr_align import SupportProblem, clear_dam, dual_summary
from ftr_align import network
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
from ftr_align.cases import rts_gmlc, toy

CLEAR = "CLARABEL"  # interior-point → analytic-center certificate (paper numbers)

pl.Config.set_tbl_rows(40)
np.set_printoptions(precision=3, suppress=True)


# %%
# -------------------------------------
# INSPECT NETWORK AND SINGLE SCENARIO
# -------------------------------------
net = rts_gmlc.load_network()
print("~~~~~~ RTS GMLC network:")
print("n nodes   :", len(net.node_names.tolist()))
print("n elements:", len(net.element_names.tolist()))
print("slack   :", net.node_names[net.slack_idx])
print(f"PTDF shape (line x node): {net.ptdf().shape}")
cont = rts_gmlc.n1_contingencies(net, verbose=True)
model = network.NetworkModel.build(network=net, contingencies=cont)
model.labels().head()

# %%
# -------------------------------------
# Random networks
# -------------------------------------
dam_cont = [cont[i] for i in np.random.choice(len(cont), 24, replace=False)]
ftr_cont = [cont[i] for i in np.random.choice(len(cont), 24, replace=False)]
dam_model = network.NetworkModel.build(network=net, contingencies=dam_cont)
ftr_model = network.NetworkModel.build(network=net, contingencies=ftr_cont)

interval_rows = []
dual_rows = []
interval_start = rts_gmlc.interval_index(8, 15, 1)
interval_end = rts_gmlc.interval_index(8, 16, 1)
intervals = np.arange(interval_start, interval_end, 1, dtype=int)
for interval in tqdm(intervals):
    dam_sol = clear_dam(
        dam_model, rts_gmlc.dam_instance(interval=interval, network=net), solver=CLEAR
    )
    sol_f = SupportProblem(dam_model, dam_sol.direction).solve(solver=CLEAR)
    sol_g = SupportProblem(ftr_model, dam_sol.direction).solve(solver=CLEAR)
    interval_rows.append(
        {
            "interval": interval,
            "MS_DAM": sol_f.value,
            "Delta": sol_g.value - sol_f.value,
            "eta": None if abs(sol_f.value) < EPS else sol_g.value / sol_f.value,
        }
    )
    dual_rows.append(
        dual_summary(
            dam_model,
            sol_f,
            ftr_model,
            sol_g,
            labels={"interval": interval},
        )
    )
interval_df = pl.DataFrame(interval_rows)
dual_df = pl.concat(dual_rows, how="vertical")

# %%
# ---------------------------------
# Inspect the worst-case interval
# ---------------------------------
# get worst-case interval and recompute problems
max_int = intervals[interval_df["Delta"].arg_max()]
dam_sol = clear_dam(
    dam_model, rts_gmlc.dam_instance(interval=max_int, network=net), solver=CLEAR
)
dam_prob = SupportProblem(dam_model, dam_sol.direction)
ftr_prob = SupportProblem(ftr_model, dam_sol.direction)

# inspect DAM model
lo, hi = robust_bounds(dam_prob, solver=CLEAR)
index = support_index(hi)
klass = classify(lo, hi)
C = trade_matrix(dam_prob, index)
D = trade_space(C)
print("~~~~~~~~ DAM model")
print("DAM support value:", round(dam_prob.solve(solver=CLEAR).value, 1))
print("DAM support rows :", index.tolist())
print("DAM classes      :", [klass[i] for i in index])
print("DAM trade space dim:", D.shape[1])
display(attribution_blocks(dam_prob, solver=CLEAR))


# inspect FTR model
lo, hi = robust_bounds(ftr_prob, solver=CLEAR)
index = support_index(hi)
klass = classify(lo, hi)
C = trade_matrix(ftr_prob, index)
D = trade_space(C)
print("\n~~~~~~~~ FTR model")
print("FTR support value:", round(ftr_prob.solve(solver=CLEAR).value, 1))
print("FTR support rows :", index.tolist())
print("FTR classes      :", [klass[i] for i in index])
print("FTR trade space dim:", D.shape[1])
display(attribution_blocks(ftr_prob, solver=CLEAR))


# Repair: per-block attribution of the gap Δ = h(g) - h(f).  marginal_repair =
# each block's standalone effect (not additive when drivers mask each other);
# shapley_repair = order-averaged, additive (sums to Δ).
print("\n~~~~~~~~ Repair of gap")
(
    marginal_repair(dam_model, ftr_model, dam_sol.direction, solver=CLEAR).join(
        shapley_repair(dam_model, ftr_model, dam_sol.direction, solver=CLEAR),
        on=["driver", "members", "idxs", "repair_idxs"],
        how="full",
        coalesce=True,
        suffix="_shapley",
    )
)
# %%
