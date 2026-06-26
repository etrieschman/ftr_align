# %%
import numpy as np
import polars as pl

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
from ftr_align.cases import rts_gmlc

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
