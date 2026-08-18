"""Dual-side properties, exercised on the 3-node toy.

Propositions only -- nothing here asserts a value that retuning a case would
move.  The paper's Table III lives in ``notebooks/reproduce_conference.py``.
"""

import numpy as np
import pytest

from ftr_align import SupportProblem, align, clear_dam, differences
from ftr_align.duality import robust_bounds
from ftr_align.cases import toy

CLEAR_SOLVER = {"solver": "CLARABEL"}

@pytest.mark.parametrize("case", list(toy.MODELS))
def test_robust_bounds_bracket_a_realized_certificate(case):
    """[mu_lo, mu_hi] ranges over the optimal dual face, so whatever certificate a
    solver returns must lie inside it, row by row."""
    f_model, g_model = toy.MODELS[case]
    dam = clear_dam(g_model, toy.SCENARIOS["(a)"], solver=CLEAR_SOLVER)
    for model in (f_model, g_model):
        problem = SupportProblem(model, dam.direction)
        lo, hi = robust_bounds(problem, solver=CLEAR_SOLVER)
        mu = problem.solve(solver=CLEAR_SOLVER).mu
        assert np.all(mu >= lo - 1e-4) and np.all(mu <= hi + 1e-4)


@pytest.mark.parametrize("case", list(toy.MODELS))
def test_differences_classifies_every_disagreeing_row(case):
    """prop:kinds, as a property of the classifier rather than a tally of which
    case contains what.

    For every case: the four buckets are disjoint, together they are exactly the
    rows where the aligned limits differ, and each row's bucket agrees with its
    own limits -- level vs coverage from finiteness, U vs V from which model is
    looser."""
    f, g = toy.MODELS[case]
    f_u, g_u = align(f, g)
    kinds = differences(f, g)

    rows = [i for key in kinds for i in kinds[key]]
    assert len(rows) == len(set(rows))  # disjoint
    assert set(rows) == set(np.where(f_u.b != g_u.b)[0].tolist())

    for key, index in kinds.items():
        level, mode = key.split("_")
        for i in index:
            both_finite = np.isfinite(f_u.b[i]) and np.isfinite(g_u.b[i])
            assert both_finite == (level == "level")
            assert (f_u.b[i] > g_u.b[i]) == (mode == "U")
