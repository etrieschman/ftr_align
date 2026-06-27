"""Slice 2 oracle: reproduce Table III (support-function dual values) and
exercise the robust-bound / classification / discrepancy machinery on the toy.
"""

import numpy as np
import pytest

from ftr_align import SupportProblem, clear_dam
from ftr_align.duality import classify, discrepancy, net_dual, robust_bounds
from ftr_align.cases import toy

CLEAR_SOLVER = {"solver": "CLARABEL"}

# signed net dual mu = mu_upper - mu_lower, keyed (contingency_label, element)
# PowerUp Table III; "B" -> base.  Values rounded as printed; compared with abs=2.
TABLE_III = {
    ("derate", "(a)"): ({("base", "SL"): 435}, {("base", "SL"): 435}),
    ("derate", "(b)"): ({("base", "SC"): 217}, {("base", "SC"): 217}),
    ("derate", "(c)"): ({("base", "SC"): -59}, {("base", "SC"): -59}),
    ("extra_ftr", "(a)"): (
        {("base", "SL"): 435},
        {("base", "SC"): -435, ("SL", "SC"): 435},
    ),
    ("extra_ftr", "(b)"): ({("base", "SC"): 217}, {("base", "SC"): 217}),
    ("extra_ftr", "(c)"): ({("base", "SC"): -59}, {("base", "SC"): -59}),
    ("dam_outage", "(a)"): (
        {("base", "SL"): 114, ("SC", "SL"): 107},
        {("base", "SL"): 221, ("base", "SC"): 107},
    ),
    ("dam_outage", "(b)"): (
        {("SC", "SL"): 145},
        {("base", "SL"): 145, ("base", "SC"): 145},
    ),
    ("dam_outage", "(c)"): ({("base", "SC"): -44}, {("base", "SC"): -44}),
}


def _as_dict(df) -> dict:
    return {(r["contingency"], r["element"]): r["mu"] for r in df.iter_rows(named=True)}


@pytest.mark.parametrize("key", list(TABLE_III))
def test_table_iii(key):
    variation, scenario = key
    exp_f, exp_g = TABLE_III[key]
    dam_model, ftr_model = toy.MODELS[variation]
    dam = clear_dam(dam_model, toy.SCENARIOS[scenario], solver=CLEAR_SOLVER)

    f = SupportProblem(dam_model, dam.direction).solve(solver=CLEAR_SOLVER)
    g = SupportProblem(ftr_model, dam.direction).solve(solver=CLEAR_SOLVER)
    got_f = _as_dict(net_dual(dam_model, f.mu))
    got_g = _as_dict(net_dual(ftr_model, g.mu))

    for exp, got in [(exp_f, got_f), (exp_g, got_g)]:
        assert set(got) == set(exp)
        for cell, val in exp.items():
            assert got[cell] == pytest.approx(val, abs=2)


def test_toy_duals_are_unique():
    """The 2-D toy has a unique support dual: robust ranges collapse."""
    dam_model, ftr_model = toy.MODELS["dam_outage"]
    dam = clear_dam(dam_model, toy.SCENARIOS["(a)"], solver=CLEAR_SOLVER)
    for model in (dam_model, ftr_model):
        lo, hi = robust_bounds(SupportProblem(model, dam.direction), solver=CLEAR_SOLVER)
        assert np.allclose(lo, hi, atol=1e-4)


def test_classification():
    dam_model, _ = toy.MODELS["derate"]
    dam = clear_dam(dam_model, toy.SCENARIOS["(a)"], solver=CLEAR_SOLVER)
    lo, hi = robust_bounds(SupportProblem(dam_model, dam.direction), solver=CLEAR_SOLVER)
    classes = classify(lo, hi)
    # scenario (a): exactly the base:SL upper row binds, nothing degenerate
    assert classes.count("binding") == 1
    assert "degenerate" not in classes


def test_discrepancy_signs():
    # extra_ftr: FTR enforces an extra contingency -> tighter -> D_minus
    dam_model, ftr_model = toy.MODELS["extra_ftr"]
    d = discrepancy(dam_model, ftr_model)
    assert len(d["D_minus"]) > 0 and len(d["D_plus"]) == 0

    # dam_outage: DAM enforces an extra contingency -> FTR looser -> D_plus
    dam_model, ftr_model = toy.MODELS["dam_outage"]
    d = discrepancy(dam_model, ftr_model)
    assert len(d["D_plus"]) > 0 and len(d["D_minus"]) == 0
