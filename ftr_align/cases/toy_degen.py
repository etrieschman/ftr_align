"""
"""
from __future__ import annotations

import numpy as np

from ..network import Contingency, NetworkModel, PhysicalNetwork
from ..solve import DamInstance

NODE_NAMES = np.array(["W", "N", "S", "D", "H"])
W, N, S, D, H = 0, 1, 2, 3, 4
ELEMENT_NAMES = np.array(["WN", "WS", "WD1", "WD2", "ND", "NH", "SD", "SH", "DH"])
WN, WS, WD1, WD2, ND, NH, SD, SH, DH = 0, 1, 2, 3, 4, 5, 6, 7, 8

# incidence (node x line), reactances, slack at L
INC = np.array([
    [1,  1,  1,  1,  0,  0,  0,  0,  0], 
    [-1, 0,  0,  0,  1,  1,  0,  0,  0], 
    [0, -1,  0,  0,  0,  0,  1,  1,  0], 
    [0,  0, -1, -1, -1,  0, -1,  0,  1], 
    [0,  0,  0,  0,  0,  -1,  0, -1, -1]], dtype=float)
X = np.array([2., 1., 1., 1., 2., 2., 1., 2., 1.])  # reactances
BASE_LIMITS = np.array([100.]*len(ELEMENT_NAMES))  # line limits
NETWORK = PhysicalNetwork(
    inc=INC, x=X, slack_idx=D, node_names=NODE_NAMES, element_names=ELEMENT_NAMES
)

# DAM bid structure shared by every clearing scenario
M_GEN = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)  # gS at S, gC at C
M_DEM = np.array([[0], [0], [1]], dtype=float)  # dL at L
MIN_GEN = np.zeros(2)
P_GEN = np.array([5.0, 150.0])

def dam_instance(q_dem: list[float], max_gen: list[float]) -> DamInstance:
    """A DAM clearing scenario.  Only demand and the generation caps vary across
    the toy patterns; the rest of the bid structure is fixed."""
    return DamInstance(
        M_gen=M_GEN,
        M_dem=M_DEM,
        min_gen=MIN_GEN,
        max_gen=np.asarray(max_gen, dtype=float),
        p_gen=P_GEN,
        q_dem=np.asarray(q_dem, dtype=float),
    )