"""Derive a case's properties from its data instead of asserting them.

The toy patterns and scenarios are working data -- they get retuned as the
analysis asks different questions.  A test that hardcodes "`mixed` crosses" or
"`derate` is a 0.75 scaling" tests the data, and breaks on a retune that was
perfectly deliberate.

So: ask the models.  ``nesting`` and ``uniform_scale`` read the limit vectors,
and ``find_case`` searches for a case with the property a test needs and skips
if the data no longer provides one.  The theorems stay pinned; the fixtures move
freely.
"""

import numpy as np
import pytest

from ftr_align import align, clear_dam, meet
from ftr_align.cases import toy

CLEAR = {"solver": "CLARABEL"}
TOL = 1e-9


def nesting(f, g) -> str:
    """``"f"`` if Q(f) is inside Q(g), ``"g"`` if the reverse, ``"cross"`` if
    neither contains the other.  Equal models report ``"f"``."""
    f_u, g_u = align(f, g)
    f_in_g = bool(np.all(f_u.b <= g_u.b + TOL))
    g_in_f = bool(np.all(g_u.b <= f_u.b + TOL))
    if f_in_g:
        return "f"
    return "g" if g_in_f else "cross"


def uniform_scale(f, g) -> float | None:
    """``alpha`` if ``f``'s limits are a uniform ``alpha`` times ``g``'s on every
    enforced row, else ``None``."""
    f_u, g_u = align(f, g)
    enforced = np.isfinite(f_u.b)
    # Both must enforce exactly the same rows: a coverage difference is not a
    # scaling however uniform the shared rows look.
    if not enforced.any() or not np.array_equal(enforced, np.isfinite(g_u.b)):
        return None
    ratio = f_u.b[enforced] / g_u.b[enforced]
    return float(ratio[0]) if np.allclose(ratio, ratio[0], atol=1e-12) else None


def direction(g, scenario="(a)"):
    return clear_dam(g, toy.SCENARIOS[scenario], solver=CLEAR).direction


def find_case(predicate, models=None, scenario="(a)", what="a case"):
    """First ``(name, f, g)`` whose models satisfy ``predicate(f, g)``.

    Skips rather than fails when none matches: the property has gone out of the
    fixtures, which is a fact about the data, not a broken method."""
    models = toy.MODELS if models is None else models
    for name, (f, g) in models.items():
        if predicate(f, g):
            return name, f, g
    pytest.skip(f"no toy case currently provides {what}")


def find_zero_mode(models=None, scenario="(a)"):
    """First ``(name, looser, target, mode, direction)`` whose failure mode is
    numerically zero -- for the paths that only trigger on an empty mode.

    The direction comes back with it because it is cleared on ``g``; the target
    is an intersection, which need not admit a feasible dispatch at all.

    "Zero" is ``1e-6 * scale``: these are differences of support values of order
    1e4, so their absolute error is ~1e-4 whatever the difference itself is."""
    from ftr_align import gap_summary

    models = toy.MODELS if models is None else models
    for name, (f, g) in models.items():
        d = direction(g, scenario)
        row = gap_summary(f, g, d, solver=CLEAR)
        scale = 1e-6 * max(1.0, abs(row["h_f"]), abs(row["h_g"]))
        for mode, looser in (("U", f), ("V", g)):
            if abs(row[mode]) < scale:
                return name, looser, meet(f, g), mode, d
    pytest.skip("no toy case currently has a structurally zero failure mode")
