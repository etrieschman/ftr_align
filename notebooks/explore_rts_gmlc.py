# %%
import numpy as np
import polars as pl
from tqdm import tqdm
import plotly.express as px

from ftr_align import SupportProblem, clear_dam, dual_summary
from ftr_align import network
from ftr_align.duality import (
    attribution_blocks,
    classify,
    marginal_repair,
    robust_bounds,
    shapley_repair,
    support_index,
    support_set,
    trade_matrix,
    trade_space,
)
from ftr_align.metrics import EPS
from ftr_align.cases import rts_gmlc
from tests.test_toy_blocks import CLEAR

SOLVER = {"solver": "HiGHS"}

pl.Config.set_tbl_rows(40)
np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(12345)


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
base = [cont[0]]  # always include base case
dam_cont = base + [cont[i + 1] for i in rng.choice(len(cont) - 1, 25, replace=False)]
ftr_cont = base + [cont[i + 1] for i in rng.choice(len(cont) - 1, 25, replace=False)]
dam_model = network.NetworkModel.build(network=net, contingencies=dam_cont)
ftr_model = network.NetworkModel.build(network=net, contingencies=ftr_cont)

interval_rows = []
dual_rows = []
interval_start = rts_gmlc.interval_index(8, 5, 1)
interval_end = rts_gmlc.interval_index(8, 15, 1)
intervals = np.arange(interval_start, interval_end, 1, dtype=int)
for interval in tqdm(intervals):
    dam_sol = clear_dam(
        dam_model, rts_gmlc.dam_instance(interval=interval, network=net), solver=SOLVER
    )
    sol_f = SupportProblem(dam_model, dam_sol.direction).solve(solver=SOLVER)
    sol_g = SupportProblem(ftr_model, dam_sol.direction).solve(solver=SOLVER)
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
worst_int = interval_df.sort("Delta", descending=True)["interval"][0]
print("Worst interval:", interval_df.filter(pl.col("interval") == worst_int))
worst_int_idx = intervals[intervals == worst_int][0]
dam_sol = clear_dam(
    dam_model, rts_gmlc.dam_instance(interval=worst_int_idx, network=net), solver=SOLVER
)
dam_prob = SupportProblem(dam_model, dam_sol.direction)
ftr_prob = SupportProblem(ftr_model, dam_sol.direction)

# inspect DAM model.  support_set gets I(b;d) from one CLARABEL solve (strict
# complementarity) -- ~50-130x cheaper than the robust_bounds face-LP loop, which
# we'd need only for lo/hi ranges (classify).  Pass index to skip it in attribution.
dam_dual = dam_prob.solve(solver={"solver": "CLARABEL"})
index = support_set(dam_prob)
C = trade_matrix(dam_prob, index)
D = trade_space(C)
print("~~~~~~~~ DAM model")
print("DAM support value:", round(dam_dual.value, 1))
print("DAM support rows :", index.tolist())
print("DAM trade space dim:", D.shape[1])
dam_attr = attribution_blocks(dam_prob, mu=dam_dual.mu, index=index)
display(dam_attr)


# inspect FTR model
ftr_dual = ftr_prob.solve(solver={"solver": "CLARABEL"})
index = support_set(ftr_prob)
C = trade_matrix(ftr_prob, index)
D = trade_space(C)
print("\n~~~~~~~~ FTR model")
print("FTR support value:", round(ftr_dual.value, 1))
print("FTR support rows :", index.tolist())
print("FTR trade space dim:", D.shape[1])
ftr_attr = attribution_blocks(ftr_prob, mu=ftr_dual.mu, index=index)
display(ftr_attr)


# Repair: per-block attribution of the gap Δ = h(g) - h(f).  marginal_repair =
# each block's standalone effect (not additive when drivers mask each other);
# shapley_repair = order-averaged, additive (sums to Δ).
print("\n~~~~~~~~ Repair of gap")
(
    marginal_repair(dam_model, ftr_model, dam_sol.direction, solver=SOLVER).join(
        shapley_repair(dam_model, ftr_model, dam_sol.direction, solver=CLEAR),
        on=["driver", "members", "idxs", "repair_idxs"],
        how="full",
        coalesce=True,
        suffix="_shapley",
    )
)
# %%
