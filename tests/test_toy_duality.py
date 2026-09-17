"""Dual-side properties, exercised on the 3-node toy.

Propositions only -- nothing here asserts a value that retuning a case would
move.  The paper's Table III lives in ``notebooks/reproduce_conference.py``.
"""

import numpy as np
import pytest

from ftr_align import SupportProblem, clear_dam
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
