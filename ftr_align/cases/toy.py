"""The 3-node toy model (PowerUp paper, Appendix B) -- the reference oracle.

Nodes S (solar), C (coal), L (load); lines SL, CL, SC.  Lines SL and SC have
finite limits (75, 25 MW); CL is effectively unconstrained (300 MW).  Smallest
instance where every object is hand-checkable; Tables II-III give exact targets.

The fixed data (network, limits, bid structure) plus the paper's cases as
constants:

  * ``SCENARIOS`` -- the three DAM clearing scenarios (a)/(b)/(c) as
    ``DamInstance``s (built with :func:`dam_instance`).
  * ``MODELS`` -- the three DAM/FTR model differences as ``(dam, ftr)`` pairs.
    Both share ``NETWORK`` and differ only in the contingencies each enforces
    (and a 0.75 FTR limit derate in ``"derate"``).
  * ``REDUNDANT_MODELS`` -- the same three differences on the double-circuit
    network (parallel SLa/SLb), for exercising the non-singleton dual face.
"""

from __future__ import annotations

import numpy as np

from ..network import Contingency, NetworkModel, PhysicalNetwork
from ..solve import DamInstance

NODE_NAMES = np.array(["S", "C", "L"])
ELEMENT_NAMES = np.array(["SL", "CL", "SC"])
SL, CL, SC = 0, 1, 2

# incidence (node x line), reactances, slack at L
INC = np.array([[1, 0, 1], [0, 1, -1], [-1, -1, 0]], dtype=float)
X = np.array([1.0, 1.0, 1.0])
BASE_LIMITS = np.array([75.0, 300.0, 25.0])
NETWORK = PhysicalNetwork(
    inc=INC, x=X, slack_idx=-1, node_names=NODE_NAMES, element_names=ELEMENT_NAMES
)

# DAM bid structure shared by every clearing scenario
M_GEN = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)  # gS at S, gC at C
M_DEM = np.array([[0], [0], [1]], dtype=float)  # dL at L
MIN_GEN = np.zeros(2)
P_GEN = np.array([5.0, 150.0])

# redundant (double-circuit) variant: SL split into parallel SLa, SLb (each
# reactance 2 -> combined 1, limit 37.5 -> combined 75).  Electrically identical
# to the base toy, but SLa/SLb share a PTDF row, so mu trades between them and
# {SLa, SLb} is a genuine size-2 attribution block.
REDUNDANT_ELEMENT_NAMES = np.array(["SLa", "SLb", "CL", "SC"])
REDUNDANT_INC = np.array([[1, 1, 0, 1], [0, 0, 1, -1], [-1, -1, -1, 0]], dtype=float)
REDUNDANT_X = np.array([2.0, 2.0, 1.0, 1.0])
REDUNDANT_LIMITS = np.array([37.5, 37.5, 300.0, 25.0])
REDUNDANT_NETWORK = PhysicalNetwork(
    inc=REDUNDANT_INC,
    x=REDUNDANT_X,
    slack_idx=-1,
    node_names=NODE_NAMES,
    element_names=REDUNDANT_ELEMENT_NAMES,
)


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


# --- the paper's cases (PowerUp Tables II-III) ----------------------------
SCENARIOS: dict[str, DamInstance] = {
    "(a)": dam_instance(q_dem=[150.0], max_gen=[150.0, 300.0]),
    "(b)": dam_instance(q_dem=[100.0], max_gen=[150.0, 300.0]),
    "(c)": dam_instance(q_dem=[100.0], max_gen=[0.5 * (100.0 - 75.0), 300.0]),
}

_BASE = Contingency(None, BASE_LIMITS)  # base case at full limits
MODELS: dict[str, tuple[NetworkModel, NetworkModel]] = {
    "derate": (
        NetworkModel.build(NETWORK, [_BASE]),
        NetworkModel.build(NETWORK, [Contingency(None, 0.75 * BASE_LIMITS)]),
    ),
    "extra_ftr": (
        NetworkModel.build(NETWORK, [_BASE]),
        NetworkModel.build(NETWORK, [_BASE, Contingency(SL, BASE_LIMITS)]),
    ),
    "dam_outage": (
        NetworkModel.build(NETWORK, [_BASE, Contingency(SC, BASE_LIMITS)]),
        NetworkModel.build(NETWORK, [_BASE]),
    ),
    "mixed": (
        NetworkModel.build(NETWORK, [_BASE, Contingency(SC, BASE_LIMITS)]),
        NetworkModel.build(NETWORK, [Contingency(None, 0.75 * BASE_LIMITS)]),
    ),
}

_REDUNDANT_BASE = Contingency(None, REDUNDANT_LIMITS)
REDUNDANT_MODELS: dict[str, tuple[NetworkModel, NetworkModel]] = {
    "derate": (
        NetworkModel.build(REDUNDANT_NETWORK, [_REDUNDANT_BASE]),
        NetworkModel.build(
            REDUNDANT_NETWORK, [Contingency(None, 0.75 * REDUNDANT_LIMITS)]
        ),
    ),
    "extra_ftr": (
        NetworkModel.build(REDUNDANT_NETWORK, [_REDUNDANT_BASE]),
        NetworkModel.build(
            REDUNDANT_NETWORK,
            [_REDUNDANT_BASE, Contingency((0, 1), REDUNDANT_LIMITS)],
        ),
    ),
    "dam_outage": (
        NetworkModel.build(
            REDUNDANT_NETWORK, [_REDUNDANT_BASE, Contingency(3, REDUNDANT_LIMITS)]
        ),
        NetworkModel.build(REDUNDANT_NETWORK, [_REDUNDANT_BASE]),
    ),
    "mixed": (
        NetworkModel.build(
            REDUNDANT_NETWORK, [_REDUNDANT_BASE, Contingency(3, REDUNDANT_LIMITS)]
        ),
        NetworkModel.build(
            REDUNDANT_NETWORK,
            [Contingency(None, 0.75 * REDUNDANT_LIMITS)],
        ),
    ),
}
