"""Slice 3 oracle: trade space D(b;y) and attribution blocks.

The double-circuit variant (parallel SLa, SLb) is the smallest instance with a
genuinely non-singleton optimal dual face: SLa and SLb have identical PTDF rows,
so mu trades between them.  This exercises every new object -- robust ranges
that don't collapse, a 1-D trade space, a size-2 block, and a face-invariant
block total.
"""

import numpy as np
import pytest

from ftr_align import SupportProblem, align, clear_dam, meet
from ftr_align.attribution import repair_value
from ftr_align.duality import (
    J_star,
    attribution_blocks,
    block_totals,
    connected_blocks,
    robust_bounds,
    trade_matrix,
    trade_space,
)
from ftr_align.cases import toy

CLEAR = {"solver": "CLARABEL"}


def _gap(f, g, scenario="(a)"):
    """Δ(f,g;y) = h(f;y) - h(g;y) and the DAM congestion direction."""
    d = clear_dam(g, toy.SCENARIOS[scenario], solver=CLEAR).direction
    f_u, g_u = align(f, g)
    delta = (SupportProblem(f_u, d).solve(solver=CLEAR).value
             - SupportProblem(g_u, d).solve(solver=CLEAR).value)
    return delta, d


def test_full_repair_recovers_the_whole_failure_mode():
    """prop:repair_basic -- U^(full index) == U, for every case."""
    from ftr_align import failure_modes

    for case in toy.MODELS:
        f, g = toy.MODELS[case]
        _, d = _gap(f, g)
        modes = failure_modes(f, g, d, solver=CLEAR)
        every_row = np.arange(len(meet(f, g).b))
        assert repair_value(f, g, d, every_row, mode="U", solver=CLEAR) == pytest.approx(
            modes["U"], abs=1e-3
        )
        assert repair_value(f, g, d, every_row, mode="V", solver=CLEAR) == pytest.approx(
            modes["V"], abs=1e-3
        )


def test_redundant_face_and_trade():
    sys = toy.REDUNDANT_MODELS["derate"][1]
    # the two parallel circuits are electrically identical
    assert np.allclose(sys.K[sys.rows_upper(None)[toy.SL]], sys.K[sys.rows_upper(None)[1]])

    dam = clear_dam(sys, toy.SCENARIOS["(a)"], solver=CLEAR)
    prob = SupportProblem(sys, dam.direction)
    # value still matches the oracle (electrically the base toy)
    assert prob.solve(solver=CLEAR).value == pytest.approx(32625, abs=2)

    lo, hi = robust_bounds(prob, solver=CLEAR)
    index = J_star(prob)
    # exactly the two SLa/SLb upper rows carry the support, and both are
    # *degenerate*: each can take the whole weight or none of it.
    assert index.tolist() == [sys.rows_upper(None)[0], sys.rows_upper(None)[1]]
    assert all(lo[i] == pytest.approx(0, abs=1e-3) for i in index)
    assert all(hi[i] > 1.0 for i in index)

    # 1-D trade space, the (1, -1) weight shift between the twins
    C = trade_matrix(prob, index)
    D = trade_space(C)
    assert D.shape[1] == 1
    d = D[:, 0]
    assert abs(d[0]) == pytest.approx(abs(d[1]), rel=1e-6)
    assert d[0] * d[1] < 0


def test_redundant_single_block():
    sys = toy.REDUNDANT_MODELS["derate"][1]
    dam = clear_dam(sys, toy.SCENARIOS["(a)"], solver=CLEAR)
    prob = SupportProblem(sys, dam.direction)
    blocks = attribution_blocks(prob)
    assert len(blocks) == 1
    assert len(blocks[0]) == 2
    W = block_totals(prob.data.b, prob.solve(solver=CLEAR).mu, blocks)
    assert W[0] == pytest.approx(32625, abs=2)


def test_block_total_is_face_invariant():
    """W_{G_r} is the same for any optimal certificate, even though individual
    multipliers differ (CLARABEL spreads weight, HiGHS puts it on one twin)."""
    sys = toy.REDUNDANT_MODELS["derate"][1]
    dam = clear_dam(sys, toy.SCENARIOS["(a)"], solver=CLEAR)
    prob = SupportProblem(sys, dam.direction)

    mu_clarabel = prob.solve(solver={"solver": "CLARABEL"}).mu
    mu_highs = prob.solve(solver={"solver": "HIGHS"}).mu
    sl = sys.rows_upper(None)[:2]
    # the split genuinely differs between solvers...
    assert not np.allclose(mu_clarabel[sl], mu_highs[sl], atol=1.0)

    blocks = attribution_blocks(prob)
    w_clarabel = block_totals(prob.data.b, mu_clarabel, blocks)[0]
    w_highs = block_totals(prob.data.b, mu_highs, blocks)[0]
    # ...but the block total does not
    assert w_clarabel == pytest.approx(w_highs, abs=2)


def test_unique_dual_gives_singletons():
    """When the dual is unique (standard toy), there are no trades and every
    binding constraint is its own block."""
    _, g_model = toy.MODELS["derate"]
    dam = clear_dam(g_model, toy.SCENARIOS["(a)"], solver=CLEAR)
    prob = SupportProblem(g_model, dam.direction)

    index = J_star(prob)
    C = trade_matrix(prob, index)
    assert trade_space(C).shape[1] == 0          # no trades
    assert all(len(g) == 1 for g in connected_blocks(C))  # all singletons

    blocks = attribution_blocks(prob, index=index)
    W = block_totals(prob.data.b, prob.solve(solver=CLEAR).mu, blocks)
    assert W.sum() == pytest.approx(prob.solve(solver=CLEAR).value, abs=2)
