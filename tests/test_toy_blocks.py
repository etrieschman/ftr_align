"""Slice 3 oracle: trade space D(b;y) and attribution blocks.

The double-circuit variant (parallel SLa, SLb) is the smallest instance with a
genuinely non-singleton optimal dual face: SLa and SLb have identical PTDF rows,
so mu trades between them.  This exercises every new object -- robust ranges
that don't collapse, a 1-D trade space, a size-2 block, and a face-invariant
block total.
"""

import numpy as np
import pytest

from ftr_align import SupportProblem, align, clear_dam
from ftr_align.duality import (
    attribution_blocks,
    classify,
    connected_blocks,
    marginal_repair,
    robust_bounds,
    shapley_repair,
    support_index,
    trade_matrix,
    trade_space,
)
from ftr_align.cases import toy

CLEAR = "CLARABEL"


def _gap(dam, ftr, scenario="(a)"):
    """Δ = h(g) - h(f) and the DAM congestion direction, for a model pair."""
    d = clear_dam(dam, toy.SCENARIOS[scenario], solver=CLEAR).direction
    dam_u, ftr_u = align(dam, ftr)
    delta = (SupportProblem(ftr_u, d).solve(solver=CLEAR).value
             - SupportProblem(dam_u, d).solve(solver=CLEAR).value)
    return delta, d


def test_shapley_repair_sums_to_gap():
    """Shapley block repairs are additive: they reconstruct Δ, even in the
    mixed case where underfunding and hedging drivers coexist."""
    for case in ("dam_outage", "mixed"):
        dam, ftr = toy.MODELS[case]
        delta, d = _gap(dam, ftr)
        total = shapley_repair(dam, ftr, d, solver=CLEAR)["repair"].sum()
        assert total == pytest.approx(delta, abs=2)


def test_marginal_repair_additive_only_for_single_driver():
    """With one driver, the marginal repair equals Δ; with both drivers present
    the marginals are masked and do NOT sum to Δ (Shapley is needed)."""
    dam, ftr = toy.MODELS["dam_outage"]            # single (underfunding) driver
    delta, d = _gap(dam, ftr)
    assert marginal_repair(dam, ftr, d, solver=CLEAR)["repair"].sum() == pytest.approx(delta, abs=2)

    dam, ftr = toy.MODELS["mixed"]                 # both drivers -> masking
    delta, d = _gap(dam, ftr)
    assert abs(marginal_repair(dam, ftr, d, solver=CLEAR)["repair"].sum() - delta) > 1.0


def test_redundant_face_and_trade():
    sys = toy.REDUNDANT_MODELS["derate"][0]
    # the two parallel circuits are electrically identical
    assert np.allclose(sys.A[sys.rows_upper(None)[toy.SL]], sys.A[sys.rows_upper(None)[1]])

    dam = clear_dam(sys, toy.SCENARIOS["(a)"], solver=CLEAR)
    prob = SupportProblem(sys, dam.direction)
    # value still matches the oracle (electrically the base toy)
    assert prob.solve(solver=CLEAR).value == pytest.approx(32625, abs=2)

    lo, hi = robust_bounds(prob, solver=CLEAR)
    index = support_index(hi)
    klass = classify(lo, hi)
    # exactly the two SLa/SLb upper rows carry the support, both degenerate
    assert index.tolist() == [sys.rows_upper(None)[0], sys.rows_upper(None)[1]]
    assert all(klass[i] == "degenerate" for i in index)
    assert all(lo[i] == pytest.approx(0, abs=1e-3) for i in index)

    # 1-D trade space, the (1, -1) weight shift between the twins
    C = trade_matrix(prob, index)
    D = trade_space(C)
    assert D.shape[1] == 1
    d = D[:, 0]
    assert abs(d[0]) == pytest.approx(abs(d[1]), rel=1e-6)
    assert d[0] * d[1] < 0


def test_redundant_single_block():
    sys = toy.REDUNDANT_MODELS["derate"][0]
    dam = clear_dam(sys, toy.SCENARIOS["(a)"], solver=CLEAR)
    blocks = attribution_blocks(SupportProblem(sys, dam.direction), solver=CLEAR)
    assert blocks.height == 1
    assert blocks["size"][0] == 2
    assert blocks["W"][0] == pytest.approx(32625, abs=2)


def test_block_total_is_face_invariant():
    """W_{G_r} is the same for any optimal certificate, even though individual
    multipliers differ (CLARABEL spreads weight, HiGHS puts it on one twin)."""
    sys = toy.REDUNDANT_MODELS["derate"][0]
    dam = clear_dam(sys, toy.SCENARIOS["(a)"], solver=CLEAR)
    prob = SupportProblem(sys, dam.direction)

    mu_clarabel = prob.solve(solver="CLARABEL").mu
    mu_highs = prob.solve(solver="HIGHS").mu
    sl = sys.rows_upper(None)[:2]
    # the split genuinely differs between solvers...
    assert not np.allclose(mu_clarabel[sl], mu_highs[sl], atol=1.0)

    w_clarabel = attribution_blocks(prob, mu=mu_clarabel, solver=CLEAR)["W"][0]
    w_highs = attribution_blocks(prob, mu=mu_highs, solver=CLEAR)["W"][0]
    # ...but the block total does not
    assert w_clarabel == pytest.approx(w_highs, abs=2)


def test_unique_dual_gives_singletons():
    """When the dual is unique (standard toy), there are no trades and every
    binding constraint is its own block."""
    dam_model, _ = toy.MODELS["derate"]
    dam = clear_dam(dam_model, toy.SCENARIOS["(a)"], solver=CLEAR)
    prob = SupportProblem(dam_model, dam.direction)

    _, hi = robust_bounds(prob, solver=CLEAR)
    index = support_index(hi)
    C = trade_matrix(prob, index)
    assert trade_space(C).shape[1] == 0          # no trades
    assert all(len(g) == 1 for g in connected_blocks(C))  # all singletons

    blocks = attribution_blocks(prob, solver=CLEAR)
    assert blocks["W"].sum() == pytest.approx(prob.solve(solver=CLEAR).value, abs=2)
