"""Reporting layer: the tables agree with the numbers they report.

Deliberately thin.  These are views over quantities already tested in
``test_attribution.py``; what needs checking is that the view does not distort
them, not the quantities themselves.
"""

import numpy as np
import polars as pl
import pytest

from ftr_align import SupportProblem, align, clear_dam, meet, constraint_table, gap_summary
from ftr_align.attribution import differences, row_shares
from ftr_align.duality import J_star, attribution_blocks, block_totals
from ftr_align.metrics import block_table, summary
from ftr_align.cases import toy
from toy_facts import find_case, find_zero_mode, nesting, uniform_scale

CLEAR = {"solver": "CLARABEL"}


def _direction(g, scenario="(a)"):
    return clear_dam(g, toy.SCENARIOS[scenario], solver=CLEAR).direction


@pytest.mark.parametrize("case", list(toy.MODELS))
def test_gap_summary_matches_the_underlying_quantities(case):
    """gap_summary is now the *only* place the failure modes are computed, so it
    is checked against first principles rather than against a sibling function:
    three raw support solves, one per polytope, and the definitional subtraction.

    (It used to be checked against `attribution.failure_modes`, which did the same
    three solves and the same subtraction.  That made the test circular in
    substance and the function redundant in fact, so the function went.)"""
    f, g = toy.MODELS[case]
    d = _direction(g)
    row = gap_summary(f, g, d, labels={"case": case}, solver=CLEAR)

    h_f = SupportProblem(f, d).solve(solver=CLEAR).value
    h_g = SupportProblem(g, d).solve(solver=CLEAR).value
    h_meet = SupportProblem(meet(f, g), d).solve(solver=CLEAR).value

    assert row["case"] == case
    # gap_summary solves the *aligned* models (it needs mu on the aligned index
    # for the floor); the reference solves above use the unaligned ones.  Same
    # polytope, different LP -- the aligned one carries extra rows pinned to
    # mu == 0 -- so the interior-point solver lands at a slightly different
    # point.  Compare at the scale the support values carry, not bit-for-bit.
    scale = max(1.0, abs(h_f), abs(h_g))
    for key, expected in (
        ("h_f", h_f),
        ("h_g", h_g),
        ("h_meet", h_meet),
        ("U", h_f - h_meet),
        ("V", h_g - h_meet),
        ("Delta", h_f - h_g),
    ):
        assert row[key] == pytest.approx(expected, abs=1e-6 * scale)

    # both modes nonnegative: Q(f^g) sits inside both Q(f) and Q(g)
    assert row["U"] >= -1e-6 * scale
    assert row["V"] >= -1e-6 * scale
    assert row["Delta"] == pytest.approx(row["U"] - row["V"], rel=1e-9)

    for mode in ("U", "V"):
        if row[f"floor_ratio_{mode}"] is not None:
            assert row[f"floor_ratio_{mode}"] == pytest.approx(
                row[f"floor_{mode}"] / row[mode], abs=1e-6 * scale
            )
        assert row[f"dim_trade_space_{mode}"] >= 0
        assert row[f"max_block_{mode}"] >= 1 or row[f"n_blocks_{mode}"] == 0


def test_gap_summaries_stack_into_a_frame():
    """The point of returning a dict: a sweep is a list comprehension, and a new
    column is a new key."""
    rows = [
        gap_summary(f, g, _direction(g, s), {"case": c, "scenario": s}, solver=CLEAR)
        for c, (f, g) in toy.MODELS.items()
        for s in toy.SCENARIOS
    ]
    df = pl.DataFrame(rows)
    assert df.height == len(toy.MODELS) * len(toy.SCENARIOS)
    assert {"case", "scenario", "U", "V", "Delta"} <= set(df.columns)


@pytest.mark.parametrize("case", list(toy.MODELS))
@pytest.mark.parametrize("mode", ["U", "V"])
def test_constraint_table_shares_sum_to_the_failure_mode(case, mode):
    """The per-row shares in the table are the same numbers the blocks sum,
    so the table must reconstruct the mode too -- restricting the rows shown
    must not drop any that carry value."""
    f, g = toy.MODELS[case]
    d = _direction(g)
    table = constraint_table(f if mode == "U" else g, d, meet(f, g), solver=CLEAR)
    modes = gap_summary(f, g, d, solver=CLEAR)
    tol = 1e-3 * max(1.0, abs(modes["h_f"]), abs(modes["h_g"]))
    assert table["loss"].sum() == pytest.approx(modes[mode], abs=tol)


def test_constraint_table_reports_limits_and_difference_kinds():
    _, f, g = find_case(lambda f, g: len(differences(f, g)["level_V"]) > 0
                        or len(differences(f, g)["coverage_U"]) > 0,
                        what="a pair whose models disagree")
    d = _direction(g)
    table = constraint_table(g, d, meet(f, g), solver=CLEAR)

    # meet is the tighter limit, row by row, as reported
    assert np.all(
        table["target_limit"].to_numpy()
        <= np.minimum(table["limit"].to_numpy(), table["target_limit"].to_numpy()) + 1e-9
    )
    # every kind the models actually carry is named in the table
    expected = {k for k, rows in differences(g, meet(f, g)).items() if len(rows)}
    assert set(table["difference"].drop_nulls().to_list()) == expected


def test_block_table_joins_both_attributions(case="mixed"):
    """The joined report carries the support attribution and the misalignment
    attribution over the same partition, and neither is distorted by the join:
    W still sums to h(model), U_B still sums to the failure mode."""
    f, g = toy.REDUNDANT_MODELS[case]
    d = _direction(g)
    m = meet(f, g)
    modes = gap_summary(f, g, d, solver=CLEAR)
    table = block_table(g, d, m)
    scale = max(1.0, abs(modes["h_g"]))

    assert table["value"].sum() == pytest.approx(modes["h_g"], abs=1e-6 * scale)
    assert table["loss"].sum() == pytest.approx(modes["V"], abs=1e-6 * scale)
    # and it is exactly the two halves, joined
    assert table["value"].to_list() == pytest.approx(
        block_table(g, d)["value"].to_list(), abs=1e-6 * scale
    )
    assert table["loss"].to_list() == pytest.approx(
        block_table(g, d, m)["loss"].to_list(), abs=1e-6 * scale
    )


def test_block_table_fractions_sum_to_one_and_survive_a_zero_mode():
    """value_frac and loss_frac are shares of their own column totals.  A zero
    failure mode gives a *typed* null column, not an untyped one -- otherwise the
    frames refuse to stack."""
    _, looser, target, mode, d = find_zero_mode(toy.REDUNDANT_MODELS)
    table = block_table(looser, d, target, labels={"mode": mode})

    assert table["loss_frac"].dtype == pl.Float64
    assert table["value_frac"].sum() == pytest.approx(1.0)
    assert table["loss_frac"].null_count() == table.height


def test_block_table_fractions_sum_to_one_for_a_nonzero_mode():
    _, f, g = find_case(lambda f, g: nesting(f, g) != "cross",
                        models=toy.REDUNDANT_MODELS, what="a nested pair")
    d = _direction(g)
    looser = g if nesting(f, g) == "f" else f
    table = block_table(looser, d, meet(f, g))
    if table["loss"].sum() == 0.0:
        pytest.skip("this pair's failure mode is zero")
    assert table["loss_frac"].sum() == pytest.approx(1.0)


def test_summary_reports_attribution_shape():
    """The single-model counterpart of gap_summary.  On the redundant toy the
    parallel twins are one block of size 2 with a 1-dimensional trade space --
    the smallest case where constraint-level attribution is unidentified."""
    sys = toy.REDUNDANT_MODELS["derate"][1]
    prob = SupportProblem(sys, _direction(sys))
    row = summary(g_u, prob.data.direction, labels={"case": "redundant"}, solver=CLEAR)

    assert row["case"] == "redundant"
    assert row["h"] == pytest.approx(32625, abs=2)
    assert row["n_priced"] == 2
    assert row["n_blocks"] == 1
    assert row["max_block"] == 2
    assert row["dim_trade_space"] == 1

    # the plain toy has a unique dual: every priced row is its own block
    _, g = toy.MODELS["derate"]
    plain = summary(g, _direction(g), solver=CLEAR)
    assert plain["max_block"] == 1
    assert plain["dim_trade_space"] == 0


# ---------------------------------------------------------------------------
# block_table without a target: attribution of the SUPPORT VALUE (one model)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", list(toy.MODELS))
def test_block_table_support_sum_to_the_support_value(case):
    """cor:block_value_invariance -- W partitions h(model;y).  One model: no
    target, no failure mode, nothing about misalignment."""
    _, g = toy.MODELS[case]
    d = _direction(g)
    h = SupportProblem(g, d).solve(solver=CLEAR).value
    table = block_table(g, d)
    assert table["value"].sum() == pytest.approx(h, abs=1e-6 * max(1.0, abs(h)))


@pytest.mark.parametrize("case", list(toy.REDUNDANT_MODELS))
def test_block_table_support_trade_space_is_computed_not_assumed(case):
    """dim_trade_space is dim ker C restricted to the block, not `size - 1`.  It
    is 0 for every singleton; on the redundant variant the parallel SLa/SLb pair
    is the block where it is nonzero."""
    _, g = toy.REDUNDANT_MODELS[case]
    for row in block_table(g, _direction(g)).iter_rows(named=True):
        assert 0 <= row["dim_trade_space"] <= row["size"] - 1
        if row["size"] == 1:
            assert row["dim_trade_space"] == 0


# ---------------------------------------------------------------------------
# block_table with a target: attribution of a FAILURE MODE (a nested pair)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", list(toy.MODELS))
@pytest.mark.parametrize("mode", ["U", "V"])
def test_block_table_misalignment_sum_to_the_failure_mode(case, mode):
    """cor:exact_split at block granularity: the blocks partition J*, so their
    shares reproduce the mode exactly."""
    f, g = toy.MODELS[case]
    d = _direction(g)
    modes = gap_summary(f, g, d, solver=CLEAR)
    table = block_table(f if mode == "U" else g, d, meet(f, g))
    scale = max(1.0, abs(modes["h_f"]), abs(modes["h_g"]))
    assert table["loss"].sum() == pytest.approx(modes[mode], abs=1e-6 * scale)


@pytest.mark.parametrize("case", list(toy.MODELS))
@pytest.mark.parametrize("mode", ["U", "V"])
def test_block_table_misalignment_share_lies_in_its_own_range(case, mode):
    """U_B is read at one q; the range is over every q in target's optimal face.
    The former had better be inside the latter, or the two describe different
    faces."""
    f, g = toy.MODELS[case]
    d = _direction(g)
    scale = max(1.0, abs(gap_summary(f, g, d, solver=CLEAR)["h_g"]))
    looser = f if mode == "U" else g
    for row in block_table(looser, d, meet(f, g)).iter_rows(named=True):
        assert row["loss_lo"] <= row["loss"] + 1e-6 * scale
        assert row["loss"] <= row["loss_hi"] + 1e-6 * scale


def test_block_table_misalignment_agree_with_block_shares():
    """The table is a *view*: same blocks, same numbers as the function it
    composes.  On the redundant variant, where the parallel SLa/SLb pair makes a
    size-2 block, so the grouping is doing something."""
    f, g = toy.REDUNDANT_MODELS["mixed"]
    d = _direction(g)
    table = block_table(g, d, meet(f, g))

    problem = SupportProblem(g, d)
    sol = problem.solve(solver=CLEAR)
    blocks = attribution_blocks(problem, J_star(problem, sol))
    q = SupportProblem(meet(f, g), d).solve(solver=CLEAR, want_primal=True).q
    share = row_shares(g, meet(f, g), sol.mu, q)

    assert table["rows"].to_list() == [[int(i) for i in rows] for rows in blocks]
    assert table["loss"].to_list() == pytest.approx(
        [float(share[rows].sum()) for rows in blocks], abs=1e-4
    )
    # the redundant network has parallel elements with identical PTDF rows, so
    # at least one block must be non-singleton -- that is what redundancy means
    assert max(table["size"]) > 1


def test_block_table_carries_both_attributions():
    """The two attributions are different quantities over the *same* blocks --
    W sums to h(model), U_B sums to h(model) - h(target).  Joinable on `block`,
    which is why they are easy to confuse and why they are separate functions."""
    f, g = toy.REDUNDANT_MODELS["mixed"]
    d = _direction(g)
    m = meet(f, g)
    joined = block_table(g, d).join(
        block_table(g, d, m).drop("members", "rows", "size"), on="block"
    )
    modes = gap_summary(f, g, d, solver=CLEAR)
    scale = max(1.0, abs(modes["h_g"]))
    assert joined.height == block_table(g, d).height
    assert joined["value"].sum() == pytest.approx(modes["h_g"], abs=1e-6 * scale)
    assert joined["loss"].sum() == pytest.approx(modes["V"], abs=1e-6 * scale)


def test_block_table_misalignment_refuse_a_crossing_pair():
    """The quantity is what `model` loses on adopting `target`, which means
    nothing unless Q(target) is inside Q(model).  `mixed` crosses in both
    directions, so neither ordering is admissible."""
    _, f, g = find_case(lambda f, g: nesting(f, g) == "cross", what="a crossing pair")
    d = _direction(g)
    for a, b in ((f, g), (g, f)):
        with pytest.raises(ValueError, match="must be contained"):
            block_table(a, d, b)


def test_block_table_misalignment_accept_any_nested_pair_not_just_the_meet():
    """What the pair form buys: a nested pair needs no intersection.  Where f is
    inside g, (g, f) measures what the DAM loses on adopting the FTR limits --
    and f *is* f ^ g there, so that is exactly V."""
    _, f, g = find_case(lambda f, g: nesting(f, g) == "f", what="a nested pair")
    d = _direction(g)
    row = gap_summary(f, g, d, solver=CLEAR)
    table = block_table(g, d, f)
    assert table["loss"].sum() == pytest.approx(
        row["V"], abs=1e-6 * max(1.0, abs(row["h_g"]))
    )


def test_block_table_misalignment_label_and_stack_by_mode():
    """No mode column of its own -- the mode *is* which model you pass first, so
    naming it is the caller's job and `labels` is where that goes."""
    f, g = toy.MODELS["derate"]
    d = _direction(g)
    both = pl.concat(
        [
            block_table(looser, d, meet(f, g), labels={"mode": m})
            for m, looser in (("U", f), ("V", g))
        ]
    )
    assert set(both["mode"]) == {"U", "V"}
