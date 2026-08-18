# %%
import numpy as np
import polars as pl
from tqdm import tqdm
import plotly.express as px

from ftr_align import SupportProblem, clear_dam, meet
from ftr_align import network
from ftr_align.duality import (
    attribution_blocks,
    block_totals,
    robust_bounds,
    J_star,
    trade_matrix,
    trade_space,
)
from ftr_align.metrics import gap_summary
from ftr_align.metrics import EPS, block_table
from ftr_align.cases import rts_gmlc
from ftr_align.solve import CENTER

# One engine throughout.  Anything touching J* -- the attribution blocks, the
# trade space -- needs the analytic-centre certificate, and simplex measured only
# ~25% faster on the value-only sweep (191ms vs 240ms per solve), which is not
# worth running two kinds of certificate through one notebook.
SOLVER = CENTER

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

# NOTE (efficiency): this loop is the scaling bottleneck and is due a rework.
# Per interval it clears the DAM and then solves both support problems on a
# ~28,800-row K, and it does that for every hour in the window.  Nothing here is
# reused across intervals even though the models -- and so K -- never change.
# Worth revisiting: warm-starting from the previous interval's basis, restricting
# to an active set rather than the full stacked K, and batching the clearings.
interval_rows = []
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
            # Delta as a fraction of DAM merchandising surplus.
            "relative_gap": (
                None
                if abs(sol_g.value) < EPS
                else (sol_f.value - sol_g.value) / sol_g.value
            ),
        }
    )
# relative_gap is None wherever MS_DAM is ~0, and a window where that holds
# throughout would otherwise infer a Null-dtype column that arithmetic rejects.
interval_df = pl.DataFrame(interval_rows, schema_overrides={"relative_gap": pl.Float64})

# %%
# ---------------------------------
# Summary across intervals
# ---------------------------------
# Delta is signed: positive is underfunding exposure, negative is lost hedge
# value, and a window can contain both -- so the extremes matter more than the
# mean.
display(
    interval_df.select(
        pl.len().alias("intervals"),
        pl.col("MS_DAM").mean().alias("MS_DAM_mean"),
        pl.col("Delta").mean().alias("Delta_mean"),
        pl.col("Delta").min().alias("Delta_min"),
        pl.col("Delta").max().alias("Delta_max"),
        (pl.col("Delta") > 0).sum().alias("intervals_underfunded"),
        pl.col("relative_gap").abs().max().alias("relative_gap_absmax"),
    )
)
display(interval_df.sort("Delta", descending=True).head(10))

px.line(
    interval_df, x="interval", y="Delta", title="Alignment gap by interval"
).show()

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

# One CENTER solve per model carries the value, mu, J* (strict complementarity),
# the blocks and their totals.  At ~28,800 rows the saving is not cosmetic: this
# used to solve each problem twice, once here and again inside J_star.
for name, model, prob in (("DAM", dam_model, dam_prob), ("FTR", ftr_model, ftr_prob)):
    sol = prob.solve(solver=CENTER)
    index = J_star(prob, sol)
    blocks = attribution_blocks(prob, index)
    D = trade_space(trade_matrix(prob, index))

    print(f"\n~~~~~~~~ {name} model")
    print(f"{name} support value:", round(sol.value, 1))
    print(f"{name} support rows :", index.tolist())
    print(f"{name} trade space dim:", D.shape[1])
    display(block_table(model, prob.data.direction))


# Failure modes and their block-level attribution (prop:block_underfunding).
print("\n~~~~~~~~ Failure modes")
print(gap_summary(ftr_model, dam_model, dam_sol.direction, solver=SOLVER))
# The mode is which model you pass first -- always the one that *loses* the
# value, measured against the intersection (prop:block_underfunding).
meet_model = meet(ftr_model, dam_model)
for mode, model in (("U", ftr_model), ("V", dam_model)):
    print(f"\n{mode} by block")
    display(block_table(model, dam_sol.direction, meet_model, labels={"mode": mode}))
# %%
