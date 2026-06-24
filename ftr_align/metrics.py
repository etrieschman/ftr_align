"""Cross-solve quantities. Each operates on ``SupportSolution`` values so it
composes over collections of certificates (multi-interval, ex-ante) later.
"""

from __future__ import annotations

from .solve import SupportSolution


def gap(sol_g: SupportSolution, sol_f: SupportSolution) -> float:
    """Alignment gap ``Delta = h(g; y) - h(f; y)``.

    Positive => FTR model extends farther than DAM (underfunding exposure);
    negative => FTR model is more restrictive (hedging inefficiency).
    Both solutions must be evaluated on the same geometry and certificate.
    """
    return sol_g.value - sol_f.value


def ratio(sol_g: SupportSolution, sol_f: SupportSolution) -> float | None:
    """Alignment ratio ``eta = h(g; y) / h(f; y)``; ``None`` if ``h(f) == 0``."""
    if abs(sol_f.value) < 1e-9:
        return None
    return sol_g.value / sol_f.value
