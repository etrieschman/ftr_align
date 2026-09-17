"""T0 plumbing: the invariants that must hold for every case, at every direction.

These run alongside every later test.  A failure here is either a bug or a false
proposition -- they are the only tests in the backlog that fail *informatively*,
because each one is a claim from the memos rather than a number from a table.
"""

import numpy as np
import pytest

from ftr_align import SupportProblem, clear_dam, intersection
from ftr_align.attribution import (
    block_share_range,
    primal_invariant,
    row_shares,
)
from ftr_align.duality import J_star, attribution_blocks
from ftr_align.metrics import gap_summary
from ftr_align.cases import toy

CLEAR = {"solver": "CLARABEL"}


def _block_shares(model, target, direction):
    """What `attribution.block_shares` used to be, inlined here: row_shares
    summed over attribution_blocks.  It was deleted from the library for the same
    reason `exact_split` was -- summing a co-indexed vector over an index set is
    the package's idiom, not a function -- and `metrics.misalignment_blocks` is
    the supported way to get these numbers with their ranges attached."""
    problem = SupportProblem(model, direction)
    sol = problem.solve(solver=CLEAR)
    q_target = SupportProblem(target, direction).solve(
        solver=CLEAR, want_primal=True
    ).q
    share = row_shares(model, target, sol.mu, q_target)
    blocks = attribution_blocks(problem, J_star(problem, sol))
    return np.array([float(share[rows].sum()) for rows in blocks])


def _loose(f, g, mode):
    """The model that *loses* the value in a mode: U is what f loses on
    adopting f ^ g, V is what g loses.  With the pair form there is no mode
    argument -- the mode is which model you pass first -- so the tests that
    still sweep both modes pick the model here."""
    return f if mode == "U" else g
CASES = list(toy.MODELS)
SCENARIOS = list(toy.SCENARIOS)


def _tol(modes: dict) -> float:
    """Absolute slack for comparisons between failure-mode quantities.

    Scaled by the *support values*, not by the quantity being compared.  Every
    object here -- U, V, a block share -- is a
    difference of support values of order 1e4, so its absolute error is ~1e-4
    however small the difference itself is.  U == 0 exactly in several toy cases,
    and a tolerance proportional to U would then demand more precision than the
    inputs carry.

    It has to be a tolerance and not exact equality because the comparisons are
    routinely between two computations of the same number."""
    return 1e-6 * max(1.0, abs(modes["h_ftr"]), abs(modes["h_dam"]))


def _setup(case, scenario, mode="U"):
    """Everything the attribution formulas need at one (case, scenario)."""
    f, g = toy.MODELS[case]
    d = clear_dam(g, toy.SCENARIOS[scenario], solver=CLEAR).direction
    model = f if mode == "U" else g
    mu = SupportProblem(model, d).solve(solver=CLEAR).mu
    q_int = SupportProblem(intersection(f, g), d).solve(solver=CLEAR, want_primal=True).q
    return f, g, d, mu, q_int


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("mode", ["U", "V"])
def test_gap_summary_modes_are_nonnegative_and_decompose_the_gap(case, scenario, mode):
    """def:failures and prop:uv."""
    f, g, d, _, _ = _setup(case, scenario, mode)
    m = gap_summary(f, g, d, solver=CLEAR)
    assert m["U"] >= -_tol(m)
    assert m["V"] >= -_tol(m)
    assert m["Delta"] == pytest.approx(m["U"] - m["V"], rel=1e-9)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("mode", ["U", "V"])
def test_exact_split_reconstructs_the_failure_mode(case, scenario, mode):
    """prop:exact_split -- summing mu_i [b_i - (K q^)_i] over the rows the model
    prices reproduces the failure mode exactly.

    The sharpest consistency check available: it ties the certificate, the
    intersection optimum and three separate support values into one identity."""
    f, g, d, mu, q_int = _setup(case, scenario, mode)
    m = gap_summary(f, g, d, solver=CLEAR)
    total = row_shares(_loose(f, g, mode), intersection(f, g), mu, q_int).sum()
    assert total == pytest.approx(m[mode], abs=1e3 * _tol(m))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("mode", ["U", "V"])
def test_block_shares_sum_to_the_failure_mode(case, scenario, mode):
    """thm:failure_blocks -- U decomposes over the blocks of the FTR support
    problem and V over those of the DAM one."""
    f, g, d, _, _ = _setup(case, scenario, mode)
    shares = _block_shares(_loose(f, g, mode), intersection(f, g), d)
    m = gap_summary(f, g, d, solver=CLEAR)
    assert shares.sum() == pytest.approx(m[mode], abs=1e3 * _tol(m))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_block_shares_are_invariant_across_the_dual_face(case, scenario):
    """thm:blocks(ii) -- every block total is the same at every optimal
    certificate.  CLARABEL and HiGHS return genuinely different points of the
    face, so agreeing on the blocks is a real test of the claim."""
    f, g = toy.MODELS[case]
    d = clear_dam(g, toy.SCENARIOS[scenario], solver=CLEAR).direction
    problem = SupportProblem(f, d)
    q_int = SupportProblem(intersection(f, g), d).solve(solver=CLEAR, want_primal=True).q
    blocks = attribution_blocks(problem)

    mus = [problem.solve(solver={"solver": s}).mu for s in ("CLARABEL", "HIGHS")]
    shares = [
        np.array(
            [row_shares(f, intersection(f, g), mu, q_int)[rows].sum() for rows in blocks]
        )
        for mu in mus
    ]
    assert np.allclose(shares[0], shares[1], atol=1e-3)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_primal_invariance_condition_matches_the_measured_range(case, scenario):
    """thm:failure_blocks(ii) -- the algebraic condition on a block holds exactly
    when that block's share does not move with the intersection optimum."""
    f, g, d, mu, _ = _setup(case, scenario, "U")
    blocks = attribution_blocks(SupportProblem(f, d))
    for rows in blocks:
        invariant = primal_invariant(f, intersection(f, g), d, mu, rows, solver=CLEAR)
        lo, hi = block_share_range(f, intersection(f, g), d, mu, rows, solver=CLEAR)
        # the leak scale of the face cut, carried through the block coefficient
        w = mu[rows] @ f.K[rows]
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
    f, g, d, mu, q_int = _setup(case, scenario, "U")
    share = row_shares(f, intersection(f, g), mu, q_int)
    priced = J_star(SupportProblem(f, d))
    off = np.setdiff1d(np.arange(len(share)), priced)
    m = gap_summary(f, g, d, solver=CLEAR)
    assert abs(share[off]).sum() <= 1e3 * _tol(m)
