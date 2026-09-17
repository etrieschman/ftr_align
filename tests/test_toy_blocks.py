"""Slice 3 oracle: shift space S(v) and attribution blocks.

The double-circuit variant (parallel SLa, SLb) is the smallest instance with a
genuinely non-singleton optimal dual face: SLa and SLb have identical PTDF rows,
so mu shifts between them.  This exercises every new object -- robust ranges
that don't collapse, a 1-D shift space, a size-2 block, and a face-invariant
block total.
"""

import numpy as np
import pytest

from ftr_align import SupportProblem, clear_dam
from ftr_align.duality import (
    J_star,
    attribution_blocks,
    block_totals,
    connected_blocks,
    robust_bounds,
    shift_matrix,
    shift_space,
)
from ftr_align.cases import toy

CLEAR = {"solver": "CLARABEL"}


def _gap(f, g, scenario="(a)"):
    """Δ(f,g;y) = h(f;y) - h(g;y) and the DAM congestion direction."""
    d = clear_dam(g, toy.SCENARIOS[scenario], solver=CLEAR).direction
    delta = (SupportProblem(f, d).solve(solver=CLEAR).value
             - SupportProblem(g, d).solve(solver=CLEAR).value)
    return delta, d


def test_redundant_face_and_shift():
    sys = toy.REDUNDANT_MODELS["derate"][1]
    # the two parallel circuits are electrically identical
    assert np.allclose(sys.K[sys.rows_upper(None)[toy.SL]], sys.K[sys.rows_upper(None)[1]])

    prob = SupportProblem(sys, _gap(*toy.REDUNDANT_MODELS["derate"])[1])
    # value still matches the oracle (electrically the base toy)
    assert prob.solve(solver=CLEAR).value == pytest.approx(32625, abs=2)

    lo, hi = robust_bounds(prob, solver=CLEAR)
    index = J_star(prob)
    # exactly the two SLa/SLb upper rows carry the support, and both are
    # *degenerate*: each can take the whole weight or none of it.
    assert index.tolist() == [sys.rows_upper(None)[0], sys.rows_upper(None)[1]]
    assert all(lo[i] == pytest.approx(0, abs=1e-3) for i in index)
    assert all(hi[i] > 1.0 for i in index)

    # 1-D shift space, the (1, -1) weight shift between the twins
    C = shift_matrix(prob, index)
    D = shift_space(C)
    assert D.shape[1] == 1
    d = D[:, 0]
    assert abs(d[0]) == pytest.approx(abs(d[1]), rel=1e-6)
    assert d[0] * d[1] < 0


def test_redundant_single_block():
    sys = toy.REDUNDANT_MODELS["derate"][1]
    prob = SupportProblem(sys, _gap(*toy.REDUNDANT_MODELS["derate"])[1])
    blocks = attribution_blocks(prob)
    assert len(blocks) == 1
    assert len(blocks[0]) == 2
    W = block_totals(prob.data.b, prob.solve(solver=CLEAR).mu, blocks)
    assert W[0] == pytest.approx(32625, abs=2)


def test_block_total_is_face_invariant():
    """W_{G_r} is the same for any optimal certificate, even though individual
    multipliers differ (CLARABEL spreads weight, HiGHS puts it on one twin)."""
    sys = toy.REDUNDANT_MODELS["derate"][1]
    prob = SupportProblem(sys, _gap(*toy.REDUNDANT_MODELS["derate"])[1])

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
    """When the dual is unique (standard toy), there are no shifts and every
    binding constraint is its own block."""
    _, g_model = toy.MODELS["derate"]
    dam = clear_dam(g_model, toy.SCENARIOS["(a)"], solver=CLEAR)
    prob = SupportProblem(g_model, dam.direction)

    index = J_star(prob)
    C = shift_matrix(prob, index)
    assert shift_space(C).shape[1] == 0          # no shifts
    assert all(len(g) == 1 for g in connected_blocks(C))  # all singletons

    blocks = attribution_blocks(prob, index=index)
    W = block_totals(prob.data.b, prob.solve(solver=CLEAR).mu, blocks)
    assert W.sum() == pytest.approx(prob.solve(solver=CLEAR).value, abs=2)
