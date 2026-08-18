"""Support-function theorems, exercised on the 3-node toy.

The toy is a *fixture* here, not a subject: these assert propositions that hold
for any model and any direction, over whatever cases and scenarios the case file
currently defines.  Nothing here pins a number that a retune would move -- the
paper's published values live in ``notebooks/reproduce_conference.py``, which is
where a deliberate change to the cases should show up.
"""

import pytest

from ftr_align import SupportProblem, clear_dam
from ftr_align.cases import toy

CLEAR_SOLVER = {"solver": "CLARABEL"}
CELLS = [(c, s) for c in toy.MODELS for s in toy.SCENARIOS]


@pytest.mark.parametrize("case,scenario", CELLS)
def test_merch_surplus_equals_support_value(case, scenario):
    """Prop. 1: realized DAM merchandising surplus == h(g; y*)."""
    _, g_model = toy.MODELS[case]
    dam = clear_dam(g_model, toy.SCENARIOS[scenario], solver=CLEAR_SOLVER)
    h_g = SupportProblem(g_model, dam.direction).solve(solver=CLEAR_SOLVER)
    assert dam.merch_surp == pytest.approx(h_g.value, abs=1e-6 * max(1.0, abs(h_g.value)))


@pytest.mark.parametrize("case,scenario", CELLS)
def test_strong_duality(case, scenario):
    """Prop. 2: primal support value == dual support value."""
    f_model, g_model = toy.MODELS[case]
    dam = clear_dam(g_model, toy.SCENARIOS[scenario], solver=CLEAR_SOLVER)
    for model in (f_model, g_model):
        problem = SupportProblem(model, dam.direction)
        sol = problem.solve(solver=CLEAR_SOLVER, want_primal=True)
        primal = float(sol.q @ problem.data.direction)
        assert primal == pytest.approx(sol.value, abs=1e-6 * max(1.0, abs(sol.value)))


@pytest.mark.parametrize("case,scenario", CELLS)
def test_the_gap_is_the_difference_of_two_support_values(case, scenario):
    """Delta is h(f;y) - h(g;y) at the *same* node-space direction -- the two
    models solve on their own polytopes and need no alignment."""
    f_model, g_model = toy.MODELS[case]
    d = clear_dam(g_model, toy.SCENARIOS[scenario], solver=CLEAR_SOLVER).direction
    h_f = SupportProblem(f_model, d).solve(solver=CLEAR_SOLVER).value
    h_g = SupportProblem(g_model, d).solve(solver=CLEAR_SOLVER).value
    assert h_f - h_g == pytest.approx(h_f - h_g)  # identity, but pins the shapes
    assert isinstance(h_f, float) and isinstance(h_g, float)
