"""Slice 3 oracle: trade space D(b;y) and attribution blocks.

The double-circuit variant (parallel SLa, SLb) is the smallest instance with a
genuinely non-singleton optimal dual face: SLa and SLb have identical PTDF rows,
so mu trades between them.  This exercises every new object -- robust ranges
that don't collapse, a 1-D trade space, a size-2 block, and a face-invariant
block total.
"""

import numpy as np
import pytest

from ftr_align import SupportProblem, clear_dam
from ftr_align.duality import (
    attribution_blocks,
    classify,
    connected_blocks,
    robust_bounds,
    support_index,
    trade_matrix,
    trade_space,
)
from ftr_align.cases import toy

CLEAR = "CLARABEL"


def test_redundant_face_and_trade():
    case = toy.build_redundant_case()
    sys = case.dam_model.system
    # the two parallel circuits are electrically identical
    assert np.allclose(sys.A[sys.rows_upper(None)[toy.SL]], sys.A[sys.rows_upper(None)[1]])

    dam = clear_dam(case.dam_model, case.instances["(a)"], solver=CLEAR)
    prob = SupportProblem(case.dam_model, dam.y)
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
    case = toy.build_redundant_case()
    dam = clear_dam(case.dam_model, case.instances["(a)"], solver=CLEAR)
    blocks = attribution_blocks(SupportProblem(case.dam_model, dam.y), solver=CLEAR)
    assert blocks.height == 1
    assert blocks["size"][0] == 2
    assert blocks["W"][0] == pytest.approx(32625, abs=2)


def test_block_total_is_face_invariant():
    """W_{G_r} is the same for any optimal certificate, even though individual
    multipliers differ (CLARABEL spreads weight, HiGHS puts it on one twin)."""
    case = toy.build_redundant_case()
    dam = clear_dam(case.dam_model, case.instances["(a)"], solver=CLEAR)
    prob = SupportProblem(case.dam_model, dam.y)

    mu_clarabel = prob.solve(solver="CLARABEL").mu
    mu_highs = prob.solve(solver="HIGHS").mu
    sl = case.dam_model.system.rows_upper(None)[:2]
    # the split genuinely differs between solvers...
    assert not np.allclose(mu_clarabel[sl], mu_highs[sl], atol=1.0)

    w_clarabel = attribution_blocks(prob, mu=mu_clarabel, solver=CLEAR)["W"][0]
    w_highs = attribution_blocks(prob, mu=mu_highs, solver=CLEAR)["W"][0]
    # ...but the block total does not
    assert w_clarabel == pytest.approx(w_highs, abs=2)


def test_unique_dual_gives_singletons():
    """When the dual is unique (standard toy), there are no trades and every
    binding constraint is its own block."""
    case = toy.build_case("derate")
    dam = clear_dam(case.dam_model, case.instances["(a)"], solver=CLEAR)
    prob = SupportProblem(case.dam_model, dam.y)

    _, hi = robust_bounds(prob, solver=CLEAR)
    index = support_index(hi)
    C = trade_matrix(prob, index)
    assert trade_space(C).shape[1] == 0          # no trades
    assert all(len(g) == 1 for g in connected_blocks(C))  # all singletons

    blocks = attribution_blocks(prob, solver=CLEAR)
    assert blocks["W"].sum() == pytest.approx(prob.solve(solver=CLEAR).value, abs=2)
