"""Slice 2 oracle: reproduce Table III (support-function dual values) and
exercise the robust-bound and discrepancy machinery on the toy.
"""

import numpy as np
import pytest

from ftr_align import SupportProblem, clear_dam, differences, net_dual
from ftr_align.duality import robust_bounds
from ftr_align.cases import toy

CLEAR_SOLVER = {"solver": "CLARABEL"}

# signed net dual mu = mu_upper - mu_lower, keyed (contingency_label, element).
# PowerUp Table III; "B" -> base.  Values rounded as printed; compared with abs=2.
# Each entry is (DAM model g, FTR model f) -- the paper's column order.
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
    exp_g, exp_f = TABLE_III[key]
    f_model, g_model = toy.MODELS[variation]
    dam = clear_dam(g_model, toy.SCENARIOS[scenario], solver=CLEAR_SOLVER)

    sol_g = SupportProblem(g_model, dam.direction).solve(solver=CLEAR_SOLVER)
    sol_f = SupportProblem(f_model, dam.direction).solve(solver=CLEAR_SOLVER)
    got_g = _as_dict(net_dual(g_model, sol_g.mu))
    got_f = _as_dict(net_dual(f_model, sol_f.mu))

    for exp, got in [(exp_g, got_g), (exp_f, got_f)]:
        assert set(got) == set(exp)
        for cell, val in exp.items():
            assert got[cell] == pytest.approx(val, abs=2)


def test_toy_duals_are_unique():
    """The 2-D toy has a unique support dual: robust ranges collapse."""
    f_model, g_model = toy.MODELS["dam_outage"]
    dam = clear_dam(g_model, toy.SCENARIOS["(a)"], solver=CLEAR_SOLVER)
    for model in (f_model, g_model):
        lo, hi = robust_bounds(SupportProblem(model, dam.direction), solver=CLEAR_SOLVER)
        assert np.allclose(lo, hi, atol=1e-4)


def test_differences_kinds_and_modes():
    """prop:kinds -- each disagreeing row is a level or coverage difference, and
    feeds exactly one failure mode (the looser model's)."""
    # extra_ftr: f enforces a contingency g does not -> coverage difference, and
    # f is the tighter model there, so it feeds V.
    d = differences(*toy.MODELS["extra_ftr"])
    assert len(d["coverage_V"]) > 0
    assert all(len(d[k]) == 0 for k in ("coverage_U", "level_U", "level_V"))

    # dam_outage: g enforces a contingency f does not -> f looser -> feeds U.
    d = differences(*toy.MODELS["dam_outage"])
    assert len(d["coverage_U"]) > 0
    assert all(len(d[k]) == 0 for k in ("coverage_V", "level_U", "level_V"))

    # derate: both enforce the base case, f at 0.75 of g -> level difference
    # feeding V, with no coverage difference anywhere.
    d = differences(*toy.MODELS["derate"])
    assert len(d["level_V"]) > 0
    assert all(len(d[k]) == 0 for k in ("level_U", "coverage_U", "coverage_V"))

    # mixed: a level difference feeding V stacked on a coverage difference
    # feeding U -- both failure modes positive at once (the T2 headline).
    d = differences(*toy.MODELS["mixed"])
    assert len(d["level_V"]) > 0 and len(d["coverage_U"]) > 0
