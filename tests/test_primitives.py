"""Step-2 primitives: the intersection model, the primal optimal face, and the
span test that both invariance conditions run on.

These are the three objects that genuinely solve or factor; everything the
attribution layer adds on top of them is arithmetic.
"""

import numpy as np
import pytest

from ftr_align import SupportProblem, clear_dam, meet
from ftr_align.duality import (
    face_leak,
    J_star,
    in_span,
    primal_face_range,
    robust_bounds,
    trade_matrix,
    trade_space,
)
from ftr_align.cases import toy

CLEAR = {"solver": "CLARABEL"}


def _direction(g, scenario="(a)"):
    return clear_dam(g, toy.SCENARIOS[scenario], solver=CLEAR).direction


def _h(model, d):
    return SupportProblem(model, d).solve(solver=CLEAR).value


# ---------------------------------------------------------------------------
# meet: the intersection model f ^ g
# ---------------------------------------------------------------------------
def test_meet_is_the_tighter_limit_rowwise():
    """def:intersection -- b(f^g) = min(f, g) on a common index."""
    f, g = toy.MODELS["mixed"]
    m = meet(f, g)
    from ftr_align import align

    f_u, g_u = align(f, g)
    assert np.array_equal(m.b, np.minimum(f_u.b, g_u.b))
    # contingency objects stay consistent with b -- the model is safe to clear
    # or inspect, not just to hand to a SupportProblem.
    rebuilt = np.concatenate(
        [
            np.concatenate([c.upper for c in m.contingencies]),
            np.concatenate([c.lower for c in m.contingencies]),
        ]
    )
    assert np.array_equal(rebuilt, m.b)


def test_meet_enforced_rows_are_the_union():
    """prop:intersection_polytope -- J(f^g) = J(f) u J(g)."""
    from ftr_align import align

    f, g = toy.MODELS["mixed"]
    f_u, g_u = align(f, g)
    m = meet(f, g)
    expected = np.isfinite(f_u.b) | np.isfinite(g_u.b)
    assert np.array_equal(np.isfinite(m.b), expected)


def test_meet_is_idempotent_and_commutative():
    f, g = toy.MODELS["mixed"]
    assert np.array_equal(meet(f, f).b, meet(f, f).b)
    assert np.array_equal(meet(f, g).b, meet(g, f).b)
    ff = meet(f, f)
    assert np.array_equal(ff.b, f.b)


@pytest.mark.parametrize("case", ["derate", "extra_ftr", "dam_outage", "mixed"])
@pytest.mark.parametrize("scenario", ["(a)", "(b)", "(c)"])
def test_meet_support_is_below_both(case, scenario):
    """Q(f^g) is contained in both, so its support value cannot exceed either --
    equivalently U >= 0 and V >= 0 (def:failure_modes).

    The tolerance is relative because the containment is frequently *tight*: in
    several of these cases a failure mode is exactly zero, so the two support
    values agree mathematically and differ only at the interior-point solver's
    accuracy (~1e-8 relative on values of 1e4)."""
    f, g = toy.MODELS[case]
    d = _direction(g, scenario)
    h_meet, h_f, h_g = _h(meet(f, g), d), _h(f, d), _h(g, d)
    slack = 1e-7 * max(1.0, abs(h_f), abs(h_g))
    assert h_meet <= h_f + slack
    assert h_meet <= h_g + slack
    # prop:alignment_gap_decomposition: Delta = U - V, identically
    assert h_f - h_g == pytest.approx((h_f - h_meet) - (h_g - h_meet), rel=1e-9)


@pytest.mark.parametrize(
    "case,tighter",
    [("derate", "f"), ("extra_ftr", "f"), ("dam_outage", "g")],
)
def test_one_signed_cases_collapse_to_the_tighter_model(case, tighter):
    """cor:canonical items 1, 4, 5 -- when the difference is one-signed the
    intersection *is* the tighter model, so the opposite failure mode vanishes."""
    f, g = toy.MODELS[case]
    from ftr_align import align

    f_u, g_u = align(f, g)
    m = meet(f, g)
    target = f_u if tighter == "f" else g_u
    assert np.array_equal(m.b, target.b)

    d = _direction(g)
    h_meet, h_f, h_g = _h(m, d), _h(f, d), _h(g, d)
    if tighter == "f":  # f ^ g == f  =>  U == 0, and V == -Delta
        assert h_meet == pytest.approx(h_f, abs=1e-6)
        assert h_g - h_meet == pytest.approx(h_g - h_f, abs=1e-6)
    else:  # f ^ g == g  =>  V == 0, and U == Delta
        assert h_meet == pytest.approx(h_g, abs=1e-6)


def test_meet_rejects_mismatched_geometry():
    """Assumption 1 is a precondition, and its failure is loud, not silent."""
    f, _ = toy.MODELS["derate"]
    other, _ = toy.REDUNDANT_MODELS["derate"]  # different network, different K
    with pytest.raises(Exception):
        meet(f, other)


def test_uniform_derate_gap_is_closed_form():
    """cor:canonical item 1 -- for f = alpha*g the whole gap is V, at exactly
    (1 - alpha) h(g;y), for every certificate."""
    f, g = toy.MODELS["derate"]
    for scenario in toy.SCENARIOS:
        d = _direction(g, scenario)
        h_f, h_g, h_meet = _h(f, d), _h(g, d), _h(meet(f, g), d)
        U = h_f - h_meet
        V = h_g - h_meet
        assert U == pytest.approx(0.0, abs=1e-6)
        assert V == pytest.approx(0.25 * h_g, rel=1e-6)
        assert h_f - h_g == pytest.approx(U - V, abs=1e-6)  # prop:alignment_gap


# ---------------------------------------------------------------------------
# primal_face: the range of a linear functional over argmax_{q in Q} d^T q
# ---------------------------------------------------------------------------
def test_primal_face_collapses_when_two_constraints_bind():
    """dam_outage/(a) binds two rows in the DAM model, so their normals span the
    2-D balanced subspace, d sits in the interior of that cone, and the exposed
    face is a single vertex: every functional is identified there."""
    _, g = toy.MODELS["dam_outage"]
    d = _direction(g)
    prob = SupportProblem(g, d)
    value = prob.solve(solver={"solver": "HIGHS"}).value
    for w in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, -1.0])):
        rng = primal_face_range(prob, w, solver=CLEAR)
        assert rng.width <= face_leak(value, w)
        assert np.allclose(rng.q_lo, rng.q_hi, atol=1e-3)


def test_primal_face_opens_up_when_one_constraint_binds():
    """derate/(a) binds a single row, so d = K^T y* is *parallel* to that row's
    normal and the exposed face is the whole edge.  A DAM-derived direction is
    never generic -- it always lies in the normal cone of the face it exposes --
    so this is the ordinary case, not a contrived one."""
    _, g = toy.MODELS["derate"]
    d = _direction(g)
    prob = SupportProblem(g, d)
    value = prob.solve(solver={"solver": "HIGHS"}).value

    w = np.array([1.0, -1.0, 0.0])  # varies along the edge
    rng = primal_face_range(prob, w, solver=CLEAR)
    assert rng.width > 100 * face_leak(value, w)  # genuinely unidentified

    for q in (rng.q_lo, rng.q_hi):
        assert d @ q == pytest.approx(value, rel=1e-6)  # on the optimal face
        assert np.all(g.K @ q <= g.b + 1e-6)  # primal feasible
        assert q.sum() == pytest.approx(0.0, abs=1e-6)  # balanced
    assert not np.allclose(rng.q_lo, rng.q_hi, atol=1e-3)  # two distinct optima


def test_primal_face_range_is_zero_for_the_direction_itself():
    """d^T q is constant on the optimal face by construction, whatever the
    face's dimension -- so the one functional guaranteed to be identified is d."""
    _, g = toy.MODELS["derate"]
    d = _direction(g)
    prob = SupportProblem(g, d)
    value = prob.solve(solver={"solver": "HIGHS"}).value
    rng = primal_face_range(prob, d, solver=CLEAR)
    assert rng.width <= face_leak(value, d)


# ---------------------------------------------------------------------------
# in_span: the shared engine for both invariance conditions
# ---------------------------------------------------------------------------
def test_in_span_basic_algebra():
    rows = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert in_span(rows, np.array([3.0, -2.0, 0.0]))
    assert not in_span(rows, np.array([0.0, 0.0, 1.0]))
    assert not in_span(rows, np.array([1.0, 1.0, 1e-3]))
    # empty span contains only the origin
    assert in_span(np.zeros((0, 3)), np.zeros(3))
    assert not in_span(np.zeros((0, 3)), np.array([1.0, 0.0, 0.0]))


def test_invariance_test_matches_the_block_partition():
    """prop:invariant_subset on the redundant twins: the pair {SLa, SLb} has a
    certified total, either twin alone does not.  This is the block partition
    derived a second way -- from the row space of C rather than from matroid
    connectivity -- so agreement is a real check, not a restatement."""
    g = toy.REDUNDANT_MODELS["derate"][1]
    d = _direction(g)
    prob = SupportProblem(g, d)
    index = J_star(prob)
    C = trade_matrix(prob, index)
    b = prob.data.b

    assert len(index) == 2  # both parallel circuits carry weight

    # b_S restricted to J*, zero off S -- S given by position within J*
    def b_S(positions):
        v = np.zeros(len(index))
        v[positions] = b[index[positions]]
        return v

    assert in_span(C, b_S(np.array([0, 1])))  # the pair: certified
    assert not in_span(C, b_S(np.array([0])))  # one twin alone: not
    assert not in_span(C, b_S(np.array([1])))


def test_singleton_blocks_have_certified_totals():
    """prop:robust_bounds' last claim, from the other side: with a unique dual
    every row is its own block, so every singleton passes the invariance test."""
    _, g = toy.MODELS["derate"]
    d = _direction(g)
    prob = SupportProblem(g, d)
    index = J_star(prob)
    C = trade_matrix(prob, index)
    assert trade_space(C).shape[1] == 0  # no trades at all

    lo, hi = robust_bounds(prob, solver=CLEAR)
    b = prob.data.b
    for pos, row in enumerate(index):
        v = np.zeros(len(index))
        v[pos] = b[row]
        assert in_span(C, v)
        assert lo[row] == pytest.approx(hi[row], abs=1e-4)


def test_primal_invariance_condition_predicts_the_observed_range():
    """prop:primal_invariance -- a block share is independent of which
    intersection optimum q^ is chosen iff w = sum_B mu_i k_i lies in
    span{1} + row(K_{J*(f^g;y)}).  The algebraic condition and the LP range must
    agree, in *both* directions.

    Uses derate, whose intersection optimum is an edge rather than a vertex.  On
    a vertex the range is zero for every w and the test would be vacuous -- which
    is exactly what dam_outage does, since there span{1} + row(K_J) is all of
    R^n and the condition holds trivially."""
    f, g = toy.MODELS["derate"]
    m = meet(f, g)
    d = _direction(g)
    prob_m = SupportProblem(m, d)
    value = prob_m.solve(solver={"solver": "HIGHS"}).value

    n = m.K.shape[1]
    basis = np.vstack([np.ones(n), m.K[J_star(prob_m)]])

    w_in = m.K[J_star(prob_m)[0]]  # in the span by construction
    w_out = np.array([1.0, -1.0, 0.0])  # off it
    assert in_span(basis, w_in)
    assert not in_span(basis, w_out)

    assert primal_face_range(prob_m, w_in, solver=CLEAR).width <= face_leak(value, w_in)
    assert primal_face_range(prob_m, w_out, solver=CLEAR).width > 100 * face_leak(
        value, w_out
    )
