"""ftr_align -- FTR/DAM structural misalignment via support-function geometry.

Layered so that each level only depends on the ones below it:

* ``network``     -- geometry: incidence ``A``, PTDF ``H``, stacked ``K``, models
* ``solve``       -- the support LP and DAM clearing
* ``duality``     -- dual face, primal face, trade space, attribution blocks
* ``attribution`` -- failure modes, repairs, floor/ceiling, block shares
* ``metrics``     -- the only layer that labels rows and emits tables
"""

from .network import (
    Contingency,
    NetworkModel,
    PhysicalNetwork,
    align,
    compute_ptdf,
    is_connected,
    meet,
    with_limits,
)
from .solve import (
    DamInstance,
    DamResult,
    SupportData,
    SupportProblem,
    SupportSolution,
    clear_dam,
    solve_support_cvxpy,
)
from .attribution import differences, failure_modes
from .metrics import alignment_summary, dual_summary, net_dual, row_table, run_row

__all__ = [
    "Contingency",
    "NetworkModel",
    "PhysicalNetwork",
    "align",
    "compute_ptdf",
    "is_connected",
    "meet",
    "with_limits",
    "DamInstance",
    "DamResult",
    "SupportData",
    "SupportProblem",
    "SupportSolution",
    "clear_dam",
    "solve_support_cvxpy",
    "differences",
    "failure_modes",
    "alignment_summary",
    "dual_summary",
    "net_dual",
    "row_table",
    "run_row",
]
