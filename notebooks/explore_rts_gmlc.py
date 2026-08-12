# %%
import numpy as np
import polars as pl
from tqdm import tqdm
import plotly.express as px

from ftr_align import SupportProblem, clear_dam, dual_summary
from ftr_align import network
from ftr_align.duality import (
    attribution_blocks,
    block_totals,
    robust_bounds,
    J_star,
    trade_matrix,
    trade_space,
)
from ftr_align.attribution import block_shares, failure_modes
from ftr_align.metrics import EPS, block_table
from ftr_align.cases import rts_gmlc

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
    sol_f = SupportProblem(ftr_model, dam_sol.direction).solve(solver=SOLVER)
    sol_g = SupportProblem(dam_model, dam_sol.direction).solve(solver=SOLVER)
    interval_rows.append(
        {
            "interval": interval,
            "MS_DAM": sol_g.value,
            "Delta": sol_f.value - sol_g.value,
            "eta": None if abs(sol_g.value) < EPS else sol_f.value / sol_g.value,
        }
    )
    dual_rows.append(
        dual_summary(
            ftr_model,
            sol_f,
            dam_model,
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

# inspect DAM model.  J_star gets J*(b;y) from one CLARABEL solve (strict
# complementarity) -- ~50-130x cheaper than the robust_bounds face-LP loop, which
# we'd need only for the lo/hi ranges.  Pass index to skip it in attribution.
dam_dual = dam_prob.solve(solver={"solver": "CLARABEL"})
index = J_star(dam_prob)
C = trade_matrix(dam_prob, index)
D = trade_space(C)
print("~~~~~~~~ DAM model")
print("DAM support value:", round(dam_dual.value, 1))
print("DAM support rows :", index.tolist())
print("DAM trade space dim:", D.shape[1])
dam_blocks = attribution_blocks(dam_prob, index=index)
dam_attr = block_table(dam_model, dam_blocks,
                         block_totals(dam_prob.data.b, dam_dual.mu, dam_blocks))
display(dam_attr)


# inspect FTR model
ftr_dual = ftr_prob.solve(solver={"solver": "CLARABEL"})
index = J_star(ftr_prob)
C = trade_matrix(ftr_prob, index)
D = trade_space(C)
print("\n~~~~~~~~ FTR model")
print("FTR support value:", round(ftr_dual.value, 1))
print("FTR support rows :", index.tolist())
print("FTR trade space dim:", D.shape[1])
ftr_blocks = attribution_blocks(ftr_prob, index=index)
ftr_attr = block_table(ftr_model, ftr_blocks,
                         block_totals(ftr_prob.data.b, ftr_dual.mu, ftr_blocks))
display(ftr_attr)


# Failure modes and their block-level attribution (prop:block_underfunding).
print("\n~~~~~~~~ Failure modes")
print(failure_modes(ftr_model, dam_model, dam_sol.direction, solver=SOLVER))
for mode, model in (("U", ftr_model), ("V", dam_model)):
    blocks, shares = block_shares(
        ftr_model, dam_model, dam_sol.direction, mode=mode, solver=SOLVER
    )
    print(f"\n{mode} by block")
    display(block_table(model, blocks, shares, value_name=mode))
# %%
