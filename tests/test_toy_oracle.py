"""Oracle test: reproduce Tables II of the PowerUp paper exactly on the 3-node
toy model.

Note on dual degeneracy: several toy patterns are degenerate (e.g. a generator
bound and a line limit bind simultaneously, or two contingency constraints
become identical at the optimum), so the realized DAM certificate ``y*`` is
non-unique -- this is the [Feng et al., 2012] LMP non-uniqueness the robust
framework is built to handle.  The paper's reported numbers correspond to the
*analytic-center* dual, which an interior-point solver (CLARABEL) produces; a
simplex vertex dual is equally valid but selects a different ``y*``.  We
therefore clear with an interior-point solver to fix the canonical certificate.
"""

import numpy as np
import pytest

from ftr_align import SupportProblem, clear_dam, gap, ratio
from ftr_align.cases import toy

CLEAR_SOLVER = "CLARABEL"

# (MS_DAM, Delta, eta) per (difference, pattern) -- PowerUp Table II
TABLE_II = {
    ("derate", "(a)"): (32625, -8156, 0.75),
    ("derate", "(b)"): (5438, -1359, 0.75),
    ("derate", "(c)"): (1484, -371, 0.75),
    ("extra_ftr", "(a)"): (32625, -10875, 0.67),
    ("extra_ftr", "(b)"): (5438, 0, 1.00),
    ("extra_ftr", "(c)"): (1484, 0, 1.00),
    ("dam_outage", "(a)"): (16583, 2674, 1.16),
    ("dam_outage", "(b)"): (10875, 3625, 1.33),
    ("dam_outage", "(c)"): (1101, 0, 1.00),
}


@pytest.mark.parametrize("key", list(TABLE_II))
def test_table_ii(key):
    difference, pattern = key
    ms_exp, delta_exp, eta_exp = TABLE_II[key]

    case = toy.build_case(difference)
    dam = clear_dam(case.dam_model, case.instances[pattern], solver=CLEAR_SOLVER)
    h_f = SupportProblem(case.dam_model, dam.y).solve(solver=CLEAR_SOLVER)
    h_g = SupportProblem(case.ftr_model, dam.y).solve(solver=CLEAR_SOLVER)

    assert h_f.value == pytest.approx(ms_exp, abs=2)
    assert gap(h_g, h_f) == pytest.approx(delta_exp, abs=2)
    assert (ratio(h_g, h_f) or 0.0) == pytest.approx(eta_exp, abs=0.01)


@pytest.mark.parametrize("key", list(TABLE_II))
def test_merch_surplus_equals_support_value(key):
    """Prop. 1: realized DAM merchandising surplus == h(f; y*)."""
    difference, pattern = key
    case = toy.build_case(difference)
    dam = clear_dam(case.dam_model, case.instances[pattern], solver=CLEAR_SOLVER)
    h_f = SupportProblem(case.dam_model, dam.y).solve(solver=CLEAR_SOLVER)
    assert dam.merch_surp == pytest.approx(h_f.value, abs=1.0)


@pytest.mark.parametrize("key", list(TABLE_II))
def test_strong_duality(key):
    """Primal support value == dual support value (Prop. 2)."""
    difference, pattern = key
    case = toy.build_case(difference)
    dam = clear_dam(case.dam_model, case.instances[pattern], solver=CLEAR_SOLVER)
    prob = SupportProblem(case.ftr_model, dam.y)
    sol = prob.solve(solver=CLEAR_SOLVER, want_primal=True)
    primal_value = float(sol.q @ prob.data.direction)
    assert primal_value == pytest.approx(sol.value, abs=1.0)
