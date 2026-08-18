"""ftr_align -- FTR/DAM structural misalignment via support-function geometry.

Layered so that each level only depends on the ones below it:

* ``network``     -- geometry: incidence ``A``, PTDF ``H``, stacked ``K``, models
* ``solve``       -- the support LP and DAM clearing
* ``duality``     -- dual face, primal face, trade space, attribution blocks
* ``attribution`` -- failure modes, repairs, floor/ceiling, block shares
* ``polytope``    -- the V-representation: vertices, active sets, directions
* ``metrics``     -- the only layer that labels rows and emits tables
* ``viz``         -- 3-node figures, one layer per call

What is re-exported here is the **working set**: what you reach for to set a
problem up, solve it, and read the answer.  Everything else stays one import
deeper (``from ftr_align.duality import robust_bounds``) -- not because it is
private, but because a flat namespace of seventy names is not an API you can
hold in your head.

A typical session::

    f, g = toy.MODELS["mixed"]                     # an (FTR, DAM) pair
    d = clear_dam(g, scenario).direction           # y*, and d = K^T y*
    gap_summary(f, g, d)                           # Delta, U, V, floors  (a dict)
    block_table(g, d, meet(f, g))                  # per block: W and U_B
    constraint_table(f, g, d, mode="V")            # per constraint, underneath
"""

# -- set a problem up ---------------------------------------------------------
from .network import (
    Contingency,
    NetworkModel,
    PhysicalNetwork,
    align,
    is_connected,
    meet,
    with_limits,
)

# -- solve it -----------------------------------------------------------------
from .solve import CENTER, VERTEX, DamInstance, DamResult, SupportProblem, clear_dam

# -- the quantities -----------------------------------------------------------
from .attribution import ceiling, differences, floor, repair_value
from .duality import J_star, attribution_blocks, robust_bounds
from .polytope import faces, polygon

# -- read the answer ----------------------------------------------------------
from .metrics import block_table, constraint_table, gap_summary, summary

__all__ = [
    # set up
    "Contingency",
    "NetworkModel",
    "PhysicalNetwork",
    "align",
    "is_connected",
    "meet",
    "with_limits",
    # solve
    "CENTER",
    "VERTEX",
    "DamInstance",
    "DamResult",
    "SupportProblem",
    "clear_dam",
    # quantities
    "J_star",
    "attribution_blocks",
    "differences",
    "ceiling",
    "faces",
    "floor",
    "polygon",
    "repair_value",
    "robust_bounds",
    # tables
    "block_table",
    "constraint_table",
    "gap_summary",
    "summary",
]
