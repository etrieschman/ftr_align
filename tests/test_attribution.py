"""T0 plumbing: the invariants that must hold for every case, at every direction.

These run alongside every later test.  A failure here is either a bug or a false
proposition -- they are the only tests in the backlog that fail *informatively*,
because each one is a claim from the memos rather than a number from a table.
"""

import numpy as np
import pytest

from ftr_align import SupportProblem, align, clear_dam, meet
from ftr_align.attribution import (
    block_share_range,
    block_shares,
    ceiling,
    differences,
    failure_modes,
    floor,
    primal_invariant,
    repair_value,
    row_shares,
)
from ftr_align.duality import J_star, attribution_blocks
from ftr_align.cases import toy

CLEAR = {"solver": "CLARABEL"}
CASES = list(toy.MODELS)
SCENARIOS = list(toy.SCENARIOS)


def _tol(modes: dict) -> float:
    """Absolute slack for comparisons between failure-mode quantities.

    Scaled by the *support values*, not by the quantity being compared.  Every
    object here -- U, V, a repair value, a floor, a block share -- is a
    difference of support values of order 1e4, so its absolute error is ~1e-4
    however small the difference itself is.  U == 0 exactly in several toy cases,
    and a tolerance proportional to U would then demand more precision than the
    inputs carry.

    It has to be a tolerance and not exact equality because these bounds are
    *attained*: cor:canonical item 1 makes the floor exactly tight for a uniform
    derate, and the ceiling closes at q^ by construction, so the comparisons are
    routinely between two computations of the same number."""
    return 1e-6 * max(1.0, abs(modes["h_f"]), abs(modes["h_g"]))


def _setup(case, scenario, mode="U"):
    """Everything the attribution formulas need at one (case, scenario)."""
    f, g = toy.MODELS[case]
    d = clear_dam(g, toy.SCENARIOS[scenario], solver=CLEAR).direction
    f_u, g_u = align(f, g)
    model = f_u if mode == "U" else g_u
    mu = SupportProblem(model, d).solve(solver=CLEAR).mu
    q_meet = SupportProblem(meet(f, g), d).solve(solver=CLEAR, want_primal=True).q
    return f, g, d, mu, q_meet


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("mode", ["U", "V"])
def test_failure_modes_are_nonnegative_and_decompose_the_gap(case, scenario, mode):
    """def:failure_modes and prop:alignment_gap_decomposition."""
    f, g, d, _, _ = _setup(case, scenario, mode)
    m = failure_modes(f, g, d, solver=CLEAR)
    assert m["U"] >= -_tol(m)
    assert m["V"] >= -_tol(m)
    assert m["Delta"] == pytest.approx(m["U"] - m["V"], rel=1e-9)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("mode", ["U", "V"])
def test_exact_split_reconstructs_the_failure_mode(case, scenario, mode):
    """cor:exact_split -- summing mu_i [b_i - (K q^)_i] over the rows the model
    prices reproduces the failure mode exactly.

    The sharpest consistency check available: it ties the certificate, the
    intersection optimum and three separate support values into one identity."""
    f, g, d, mu, q_meet = _setup(case, scenario, mode)
    m = failure_modes(f, g, d, solver=CLEAR)
    total = row_shares(f, g, mu, q_meet, mode=mode).sum()
    assert total == pytest.approx(m[mode], abs=1e3 * _tol(m))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("mode", ["U", "V"])
def test_repair_is_normalised_monotone_and_complete(case, scenario, mode):
    """prop:repair_basic -- U^(empty) = 0, U^(S) <= U^(S') for S in S', and
    U^(full) = U.  Monotonicity is the property that fails if the repair target
    is g rather than f ^ g."""
    f, g, d, _, _ = _setup(case, scenario, mode)
    n_rows = len(meet(f, g).b)
    every = np.arange(n_rows)
    modes = failure_modes(f, g, d, solver=CLEAR)

    tol = _tol(modes)
    assert repair_value(f, g, d, [], mode=mode, solver=CLEAR) == pytest.approx(0, abs=tol)
    assert repair_value(f, g, d, every, mode=mode, solver=CLEAR) == pytest.approx(
        modes[mode], abs=1e3 * tol
    )

    prev = 0.0
    for cut in (n_rows // 4, n_rows // 2, 3 * n_rows // 4, n_rows):
        value = repair_value(f, g, d, every[:cut], mode=mode, solver=CLEAR)
        assert value >= prev - tol
        assert value >= -tol
        prev = value


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("mode", ["U", "V"])
def test_floor_bounds_every_repair_and_is_additive(case, scenario, mode):
    """prop:floor -- the floor never exceeds the repair value it bounds, at any
    S, and is additive in S (unlike the repair value itself)."""
    f, g, d, mu, _ = _setup(case, scenario, mode)
    n_rows = len(meet(f, g).b)
    every = np.arange(n_rows)
    tol = _tol(failure_modes(f, g, d, solver=CLEAR))

    for cut in (0, n_rows // 3, 2 * n_rows // 3, n_rows):
        rows = every[:cut]
        lower = floor(f, g, mu, rows, mode=mode)
        actual = repair_value(f, g, d, rows, mode=mode, solver=CLEAR)
        assert lower <= actual + tol

    left, right = every[: n_rows // 2], every[n_rows // 2 :]
    assert floor(f, g, mu, left, mode=mode) + floor(
        f, g, mu, right, mode=mode
    ) == pytest.approx(floor(f, g, mu, every, mode=mode), abs=1e-9)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_coverage_differences_carry_no_floor(case, scenario):
    """cor:diagnosable -- an infinite limit forces mu_i = 0, so a coverage
    difference contributes nothing to the floor.  Its whole effect is displaced
    value landing on other rows."""
    f, g, d, mu, _ = _setup(case, scenario, "U")
    tol = _tol(failure_modes(f, g, d, solver=CLEAR))
    diff = differences(f, g)
    for key in ("coverage_U", "coverage_V"):
        if len(diff[key]):
            assert floor(f, g, mu, diff[key], mode="U") == pytest.approx(0.0, abs=tol)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("mode", ["U", "V"])
def test_ceiling_bounds_the_failure_mode_from_above(case, scenario, mode):
    """prop:ceiling, and rem:injection_nesting on which q sharpens it.

    q = 0 is admissible for every S and returns h(f;y) -- the weakest possible
    bound.  A q attaining h(f^g;y) closes the bound on the *full* failure mode.
    Both must dominate the true value; only the second is tight."""
    f, g, d, mu, q_meet = _setup(case, scenario, mode)
    m = failure_modes(f, g, d, solver=CLEAR)
    h = m["h_f"] if mode == "U" else m["h_g"]

    loose = ceiling(f, g, mu, d, np.zeros_like(q_meet), mode=mode)
    tight = ceiling(f, g, mu, d, q_meet, mode=mode)

    tol = _tol(m)
    assert loose == pytest.approx(h, abs=1e3 * tol)  # uninformative, as advertised
    assert tight >= m[mode] - tol
    assert tight == pytest.approx(m[mode], abs=1e3 * tol)  # closed at q^
    assert tight <= loose + tol


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("mode", ["U", "V"])
def test_block_shares_sum_to_the_failure_mode(case, scenario, mode):
    """prop:block_underfunding -- U decomposes over the blocks of the FTR support
    problem and V over those of the DAM one."""
    f, g, d, _, _ = _setup(case, scenario, mode)
    _, shares = block_shares(f, g, d, mode=mode, solver=CLEAR)
    m = failure_modes(f, g, d, solver=CLEAR)
    assert shares.sum() == pytest.approx(m[mode], abs=1e3 * _tol(m))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_block_shares_are_invariant_across_the_dual_face(case, scenario):
    """prop:dual_invariance -- every block total is the same at every optimal
    certificate.  CLARABEL and HiGHS return genuinely different points of the
    face, so agreeing on the blocks is a real test of the claim."""
    f, g = toy.MODELS[case]
    d = clear_dam(g, toy.SCENARIOS[scenario], solver=CLEAR).direction
    f_u, _ = align(f, g)
    problem = SupportProblem(f_u, d)
    q_meet = SupportProblem(meet(f, g), d).solve(solver=CLEAR, want_primal=True).q
    blocks = attribution_blocks(problem)

    mus = [problem.solve(solver={"solver": s}).mu for s in ("CLARABEL", "HIGHS")]
    shares = [
        np.array(
            [row_shares(f, g, mu, q_meet, mode="U")[rows].sum() for rows in blocks]
        )
        for mu in mus
    ]
    assert np.allclose(shares[0], shares[1], atol=1e-3)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_primal_invariance_condition_matches_the_measured_range(case, scenario):
    """prop:primal_invariance -- the algebraic condition on a block holds exactly
    when that block's share does not move with the intersection optimum."""
    f, g, d, mu, _ = _setup(case, scenario, "U")
    f_u, _ = align(f, g)
    blocks = attribution_blocks(SupportProblem(f_u, d))
    for rows in blocks:
        invariant = primal_invariant(f, g, d, mu, rows, mode="U", solver=CLEAR)
        lo, hi = block_share_range(f, g, d, mu, rows, mode="U", solver=CLEAR)
        # the leak scale of the face cut, carried through the block coefficient
        w = mu[rows] @ f_u.K[rows]
        leak = 1e-4 * max(1.0, abs(lo)) + 1e-6 * float(np.linalg.norm(w)) * 1e5
        if invariant:
            assert hi - lo <= leak
        else:
            assert hi - lo > 0.0


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_only_priced_rows_carry_a_share(case, scenario):
    """A row outside J*(b;y) has mu_i = 0 in every optimal certificate, so it can
    carry no attributed value -- the support is what bounds attribution.

    Stated as a mass check rather than a per-row one: an interior-point
    certificate leaves ~1e-7 multipliers on unpriced rows, which against a limit
    of a few hundred MW is dollars of numerical dust.  What must hold is that the
    dust is negligible against the failure mode, not that it is bit-zero."""
    f, g, d, mu, q_meet = _setup(case, scenario, "U")
    f_u, _ = align(f, g)
    share = row_shares(f, g, mu, q_meet, mode="U")
    priced = J_star(SupportProblem(f_u, d))
    off = np.setdiff1d(np.arange(len(share)), priced)
    m = failure_modes(f, g, d, solver=CLEAR)
    assert abs(share[off]).sum() <= 1e3 * _tol(m)
