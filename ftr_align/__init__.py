"""ftr_align -- FTR/DAM structural misalignment via support-function geometry.

Layered so that each level only depends on the ones below it:

* ``network``     -- geometry: incidence ``A``, PTDF ``H``, stacked ``K``, models
* ``solve``       -- the support LP and DAM clearing
* ``duality``     -- dual face, primal face, shift space, attribution blocks
* ``attribution`` -- row and block shares of a failure mode
* ``polytope``    -- the V-representation: vertices, active sets, directions
* ``metrics``     -- the only layer that labels rows and emits tables
* ``viz``         -- 3-node figures, one layer per call

What is re-exported here is the **working set**: what you reach for to set a
problem up, solve it, and read the answer.  Everything else stays one import
deeper (``from ftr_align.duality import robust_bounds``) -- not because it is
private, but because a flat namespace of seventy names is not an API you can
hold in your head.

A typical session::

    ftr, dam = toy.MODELS["mixed"]                 # an (FTR, DAM) pair
    v = clear_dam(dam, scenario).direction         # y*, and v = K^T y*
    gap_summary(ftr, dam, v)                       # Delta, U, V, block shape  (a dict)
    block_table(dam, v, intersection(ftr, dam))    # per block: W_B and V_B
    constraint_table(dam, v, intersection(ftr, dam))  # per constraint, underneath
"""

# -- set a problem up ---------------------------------------------------------
from .network import (
    Contingency,
    ContingencyRows,
    NetworkModel,
    PhysicalNetwork,
    intersection,
    is_connected,
    with_limits,
)

# -- solve it -----------------------------------------------------------------
from .solve import CENTER, VERTEX, DamInstance, DamResult, SupportProblem, clear_dam

# -- the quantities -----------------------------------------------------------
from .duality import J_star, attribution_blocks, robust_bounds
from .polytope import faces, polygon

# -- read the answer ----------------------------------------------------------
from .metrics import block_table, constraint_table, gap_summary, summary

__all__ = [
    # set up
    "Contingency",
    "ContingencyRows",
    "NetworkModel",
    "PhysicalNetwork",
    "intersection",
    "is_connected",
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
    "faces",
    "polygon",
    "robust_bounds",
    # tables
    "block_table",
    "constraint_table",
    "gap_summary",
    "summary",
]
