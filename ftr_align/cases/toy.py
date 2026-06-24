"""The 3-node toy model (PowerUp paper, Appendix B) -- the reference oracle.

Nodes S (solar), C (coal), L (load); lines SL, CL, SC.  Lines SL and SC have
finite limits (75, 25 MW); CL is effectively unconstrained (300 MW).  Smallest
instance where every object is hand-checkable; Tables II-III give exact targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..network import Contingency, ContingencyKey, NetworkModel, PhysicalNetwork
from ..solve import DamInstance

NODE_NAMES = np.array(["S", "C", "L"])
ELEMENT_NAMES = np.array(["SL", "CL", "SC"])
SL, CL, SC = 0, 1, 2

# incidence (node x line), reactances, slack at L
INC = np.array([[1, 0, 1], [0, 1, -1], [-1, -1, 0]], dtype=float)
X = np.array([1.0, 1.0, 1.0])
BASE_LIMITS = np.array([75.0, 300.0, 25.0])

# DAM bid structure (common to all patterns)
M_GEN = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)  # gS at S, gC at C
M_DEM = np.array([[0], [0], [1]], dtype=float)  # dL at L
MIN_GEN = np.zeros(2)
P_GEN = np.array([5.0, 150.0])

# congestion patterns: (q_dem, max_gen) -- reused across all model differences
PATTERNS: dict[str, dict] = {
    "(a)": {"q_dem": [150.0], "max_gen": [150.0, 300.0]},
    "(b)": {"q_dem": [100.0], "max_gen": [150.0, 300.0]},
    "(c)": {"q_dem": [100.0], "max_gen": [0.5 * (100.0 - 75.0), 300.0]},
}

# model differences: DAM contingencies, FTR contingencies, FTR derate.
DIFFERENCES: dict[str, dict] = {
    "derate": {"dam": [None], "ftr": [None], "alpha": 0.75},
    "extra_ftr": {"dam": [None], "ftr": [None, SL], "alpha": 1.0},
    "dam_outage": {"dam": [None, SC], "ftr": [None], "alpha": 1.0},
}


def toy_network() -> PhysicalNetwork:
    return PhysicalNetwork(
        inc=INC, x=X, slack_idx=2,
        node_names=NODE_NAMES, element_names=ELEMENT_NAMES,
    )


def _symmetric_model(
    net: PhysicalNetwork, keys: list[ContingencyKey], limits: np.ndarray
) -> NetworkModel:
    limits = np.asarray(limits, dtype=float)
    conts = [Contingency(k, limits.copy(), limits.copy()) for k in keys]
    return NetworkModel.build(net, conts)


def instance(pattern: str) -> DamInstance:
    p = PATTERNS[pattern]
    return DamInstance(
        M_gen=M_GEN, M_dem=M_DEM, min_gen=MIN_GEN,
        max_gen=np.asarray(p["max_gen"], dtype=float),
        p_gen=P_GEN, q_dem=np.asarray(p["q_dem"], dtype=float),
    )


@dataclass
class ToyCase:
    name: str
    dam_model: NetworkModel
    ftr_model: NetworkModel
    instances: dict[str, DamInstance]


def build_case(difference: str) -> ToyCase:
    """DAM and FTR models defined independently (no alignment needed: the gap
    uses the node-space direction)."""
    spec = DIFFERENCES[difference]
    net = toy_network()
    dam = _symmetric_model(net, spec["dam"], BASE_LIMITS)
    ftr = _symmetric_model(net, spec["ftr"], BASE_LIMITS * spec["alpha"])
    return ToyCase(difference, dam, ftr, {k: instance(k) for k in PATTERNS})


# --- redundant (double-circuit) variant ----------------------------------
# Split line SL into two parallel circuits SLa, SLb, each reactance 2 (combined
# = 1) and limit 37.5 (combined = 75).  Electrically identical to the base toy,
# but SLa and SLb have identical PTDF rows -> mu trades between them, Lambda* is
# non-singleton, and {SLa, SLb} is a genuine size-2 attribution block.
REDUNDANT_ELEMENT_NAMES = np.array(["SLa", "SLb", "CL", "SC"])
REDUNDANT_INC = np.array([[1, 1, 0, 1], [0, 0, 1, -1], [-1, -1, -1, 0]], dtype=float)
REDUNDANT_X = np.array([2.0, 2.0, 1.0, 1.0])
REDUNDANT_LIMITS = np.array([37.5, 37.5, 300.0, 25.0])


def redundant_network() -> PhysicalNetwork:
    return PhysicalNetwork(
        inc=REDUNDANT_INC, x=REDUNDANT_X, slack_idx=2,
        node_names=NODE_NAMES, element_names=REDUNDANT_ELEMENT_NAMES,
    )


def build_redundant_case() -> ToyCase:
    net = redundant_network()
    dam = _symmetric_model(net, [None], REDUNDANT_LIMITS)
    ftr = _symmetric_model(net, [None], REDUNDANT_LIMITS)
    return ToyCase("redundant", dam, ftr, {k: instance(k) for k in PATTERNS})
