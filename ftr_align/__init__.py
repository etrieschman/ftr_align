"""ftr_align -- FTR/DAM structural misalignment via support-function geometry."""

from .network import (
    Contingency,
    NetworkModel,
    PhysicalNetwork,
    align,
    compute_ptdf,
    embed,
    is_connected,
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
from .metrics import alignment_summary, dual_summary

__all__ = [
    "Contingency",
    "NetworkModel",
    "PhysicalNetwork",
    "align",
    "compute_ptdf",
    "embed",
    "is_connected",
    "DamInstance",
    "DamResult",
    "SupportData",
    "SupportProblem",
    "SupportSolution",
    "clear_dam",
    "solve_support_cvxpy",
    "alignment_summary",
    "dual_summary",
]
